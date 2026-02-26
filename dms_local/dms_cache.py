# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from nvidia/Qwen3-8B-DMS-8x (HuggingFace).
# Changes:
#   - get_contiguous_cache(): replaced Python batch-loop for boolean
#     compaction with Triton compact_by_bool kernel
#   - left_pad_one() calls replaced with Triton left_pad_2d kernel

import functools
import gc
import math
from typing import Any

import torch

from transformers import CacheLayerMixin
from transformers.cache_utils import Cache

from .triton_kernels import left_pad_2d, compact_by_bool


def ceil_int_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def float_ceil(a: float):
    return math.ceil(a)


def _aux_potential_eviction(
    vals_for_replacement: torch.Tensor,
    to_be_evicted_table_block_id: torch.Tensor,
    to_be_evicted_position_within_block: torch.Tensor,
    to_be_evicted_mask: torch.Tensor,
    block_table: torch.Tensor,
    blocks: torch.Tensor,
    page_batch_index: torch.Tensor,
    last_table_block_id: torch.Tensor,
    next_position_within_block: torch.Tensor,
):
    """Adding a new element to KV cache may lead to eviction of the last element in the DMS sliding window."""

    # For each batch element the block table contains a list of blocks allocated for this batch element
    block_ids = block_table[page_batch_index, to_be_evicted_table_block_id]

    # Override the last element of the sliding window with the new element if the last element of the sliding window
    # is marked for the eviction and the window is full
    blocks[block_ids, to_be_evicted_position_within_block, :, :] = (
        blocks[block_ids, to_be_evicted_position_within_block, :, :] * (1 - to_be_evicted_mask[:, None, None])
        + vals_for_replacement[:, 0, None, :] * to_be_evicted_mask[:, None, None]
    )

    # Otherwise write the new element to the next position within the last allocated block
    block_ids = block_table[page_batch_index, last_table_block_id]
    blocks[block_ids, next_position_within_block, :, :] = blocks[
        block_ids, next_position_within_block, :, :
    ] * to_be_evicted_mask[:, None, None] + vals_for_replacement[:, 0, None, :] * (
        1 - to_be_evicted_mask[:, None, None]
    )


@torch.compile()
def _aux_update_single(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    eviction_info: torch.Tensor,
    recent_info: torch.Tensor,
    recent_info_position: torch.Tensor,
    block_table: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    cache_seq_lenghts: torch.Tensor,
    page_batch_index: torch.Tensor,
) -> torch.Tensor:
    # page_batch, seq_len, head_dim = key_states.size()
    # page_batch, seq_len = eviction_info.size()
    # page_batch_index is a tensor of shape (page_batch,): 0, 1, 2, ... page_batch - 1
    block_size = key_blocks.size(1)

    # `recent_info_position` points to the next position in the sliding window; when the sliding window is full,
    # it points to the first position. Not-filled elements are not zeroed out and not marked for eviction
    # (see recent_info initialization).
    eviction_candidate_info_position = recent_info_position % recent_info.size(1)

    eviction_candidate_info = recent_info[
        page_batch_index, eviction_candidate_info_position
    ]  # Note that this is zeroed out in the beginning

    # `eviction_candidate_info[:, 1]` is 1 when the element is marked for eviction and 0 otherwise
    # `block_table[eviction_candidate_info[:, 0] // block_size]` is the block id where the element resides
    # and `eviction_candidate_info[:, 0] % block_size` is the position (offset) within the block
    to_be_evicted = eviction_candidate_info[:, 1] == 1
    to_be_evicted_kv = to_be_evicted.to(key_blocks.dtype)
    to_be_evicted_int = to_be_evicted.to(torch.int32)
    to_be_evicted_position = eviction_candidate_info[:, 0]
    to_be_evicted_table_block_id = to_be_evicted_position // block_size
    to_be_evicted_position_within_block = to_be_evicted_position % block_size

    last_table_block_id = cache_seq_lenghts // block_size
    next_position_within_block = cache_seq_lenghts % block_size

    _aux_potential_eviction(
        vals_for_replacement=key_states,
        to_be_evicted_table_block_id=to_be_evicted_table_block_id,
        to_be_evicted_position_within_block=to_be_evicted_position_within_block,
        to_be_evicted_mask=to_be_evicted_kv,
        block_table=block_table,
        blocks=key_blocks,
        page_batch_index=page_batch_index,
        last_table_block_id=last_table_block_id,
        next_position_within_block=next_position_within_block,
    )

    _aux_potential_eviction(
        vals_for_replacement=value_states,
        to_be_evicted_table_block_id=to_be_evicted_table_block_id,
        to_be_evicted_position_within_block=to_be_evicted_position_within_block,
        to_be_evicted_mask=to_be_evicted_kv,
        block_table=block_table,
        blocks=value_blocks,
        page_batch_index=page_batch_index,
        last_table_block_id=last_table_block_id,
        next_position_within_block=next_position_within_block,
    )

    final_position = to_be_evicted_position * to_be_evicted_int + (1 - to_be_evicted_int) * (cache_seq_lenghts)

    previous_recent_info_position = (recent_info_position + recent_info.size(1) - 1) % recent_info.size(1)

    # Update the eviction info for the previous element in the sliding window (if present)
    recent_info[page_batch_index, previous_recent_info_position, 1] = (
        eviction_info[:, 0] * (cache_seq_lenghts > 0).to(torch.int32)
    ).to(torch.int32)

    # No info about eviction yet for the new element
    recent_info[page_batch_index, eviction_candidate_info_position, 1] = 0
    recent_info[page_batch_index, eviction_candidate_info_position, 0] = final_position

    recent_info_position[...] += 1

    cache_seq_lenghts[...] = cache_seq_lenghts + (1 - to_be_evicted_int)

    # At the beginning of this function call block_table[cache_seq_lenghts // block_size] points to a block with
    # at least one free position; need to maintain this invariant by detecting filled blocks
    requires_free_page = torch.logical_and((cache_seq_lenghts % block_size) == 0, to_be_evicted_int == 0)

    return requires_free_page


def _aux_get_recent_position_size(cache_seq_lenghts: torch.Tensor, dms_window_size: int) -> torch.Tensor:
    return torch.clamp(cache_seq_lenghts, max=dms_window_size)


def _aux_get_first_recent_position(
    recent_info_position: torch.Tensor,
    cache_seq_lenghts: torch.Tensor,
    dms_window_size: int,
) -> torch.Tensor:
    return recent_info_position - _aux_get_recent_position_size(
        cache_seq_lenghts=cache_seq_lenghts, dms_window_size=dms_window_size
    )


def _aux_write_kv(
    block_table: torch.Tensor,
    blocks: torch.Tensor,
    write_positions: torch.Tensor,
    values: torch.Tensor,
    page_batch_index: torch.Tensor,
):
    page_batch, chunk_len = write_positions.size()
    block_size = blocks.size(1)
    block_table_id = write_positions // block_size
    position_within_block = write_positions % block_size

    block_id = block_table[page_batch_index[:, None], block_table_id]
    assert (block_id != -1).all()

    blocks[block_id, position_within_block, :, :] = values[:, :, None, :]


@torch.compile()
def _aux_update_many_handle_single_chunk(
    update_key_chunk: torch.Tensor,
    update_value_chunk: torch.Tensor,
    eviction_info_chunk: torch.Tensor,
    block_table: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    cache_seq_lenghts: torch.Tensor,
    recent_info: torch.Tensor,
    recent_info_position: torch.Tensor,
    page_batch_index: torch.Tensor,
    update_mask: torch.Tensor,
    true_update_size: torch.Tensor,
) -> torch.Tensor:
    """
    Used for prefilling the KV cache as each tensor has a fixed size.

    `true_update_size` represents the true number of elements to be added for each batch index.
    """

    assert update_key_chunk.size() == update_value_chunk.size()
    page_batch, chunk_len, head_dim = update_key_chunk.size()
    assert chunk_len < recent_info.size(1)

    assert eviction_info_chunk.size() == (page_batch, chunk_len)
    assert page_batch_index.size() == (page_batch,)

    block_size = key_blocks.size(1)

    device = update_key_chunk.device

    # First we update the eviction info for the previous element if present
    update_eviction_info_positions = (recent_info_position - 1) % recent_info.size(1)
    update_eviction_info_mask = (cache_seq_lenghts > 0).to(torch.int32)

    recent_info[page_batch_index, update_eviction_info_positions, 1] = (
        eviction_info_chunk[:, 0] * update_eviction_info_mask
        + (1 - update_eviction_info_mask) * recent_info[page_batch_index, update_eviction_info_positions, 1]
    ).to(torch.int32)

    chunk_indexer = torch.arange(chunk_len, dtype=torch.int32, device=device)

    # The following trick handles variable lens: if the index is longer than true_update_size, then pad the index
    # with the last element within the true_update_size, e.g., [0, 1, 2, 3, 4, 5] and true_update_size = [3]
    # means that we have [0, 1, 2, 2, 2, 2] . This will later be used to write the same element multiple times
    # while preserving the constant shapes of the tensors.
    potential_eviction_positions_in_recent_info = (
        recent_info_position[:, None] + torch.minimum(chunk_indexer[None, :], true_update_size[:, None] - 1)
    ) % recent_info.size(1)

    potential_eviction_positions_in_seq = recent_info[
        page_batch_index[:, None], potential_eviction_positions_in_recent_info, 0
    ]
    confirmed_evictions_mask = (
        recent_info[page_batch_index[:, None], potential_eviction_positions_in_recent_info, 1] == 1
    )

    # Account for the padding with the last element (as described above) to get a proper count of confirmed evictions
    confirmed_evictions_mask[:, 1:] = torch.logical_and(
        confirmed_evictions_mask[:, 1:],
        potential_eviction_positions_in_recent_info[:, 1:] != potential_eviction_positions_in_recent_info[:, :-1],
    )

    confirmed_evictions_cum_sum = confirmed_evictions_mask.to(torch.int32).cumsum(dim=-1)
    confirmed_evictions_mask = torch.logical_and(
        confirmed_evictions_mask,
        confirmed_evictions_cum_sum <= true_update_size[:, None],
    )

    # Count how many new positions are needed for each element of the batch
    num_confirmed_evictions = confirmed_evictions_mask.to(torch.int32).sum(dim=-1)
    new_positions_used = true_update_size - num_confirmed_evictions

    assert (new_positions_used >= 0).all()
    assert new_positions_used.size() == (page_batch,)

    new_free_positions = cache_seq_lenghts[:, None] + torch.clamp(
        torch.minimum(chunk_indexer[None, :], new_positions_used[:, None] - 1), min=0
    )

    assert new_free_positions.size() == (page_batch, chunk_len)
    assert new_free_positions.size() == potential_eviction_positions_in_seq.size()

    potential_eviction_positions_in_seq = torch.cat(
        [
            potential_eviction_positions_in_seq,
            new_free_positions,
        ],
        dim=-1,
    )

    # Padding below allows for constant shape ops to take prefix of length new_positions_used from new_free_positions
    confirmed_evictions_padding = torch.zeros_like(confirmed_evictions_mask)
    padding_chunk_size = chunk_len - num_confirmed_evictions[:, None]
    indexer = torch.minimum(chunk_indexer[None, :], torch.clamp(padding_chunk_size - 1, min=0))

    confirmed_evictions_padding[page_batch_index[:, None], indexer] = True
    # If only post eviction positions are used, then have writing padding that ends in the last of those positions,
    # instead of the next free position
    confirmed_evictions_padding = torch.logical_and(confirmed_evictions_padding, padding_chunk_size > 0)

    confirmed_evictions_mask = torch.cat([confirmed_evictions_mask, confirmed_evictions_padding], dim=-1)

    pad_selector = (new_positions_used > 0).to(torch.int32)[:, None]

    potential_eviction_positions_in_seq[:, chunk_len:] = (
        pad_selector * potential_eviction_positions_in_seq[:, chunk_len:]
        + (1 - pad_selector) * potential_eviction_positions_in_seq[:, [chunk_len - 1]]
    )

    new_write_positions = potential_eviction_positions_in_seq[confirmed_evictions_mask].reshape(page_batch, chunk_len)

    _aux_write_kv(
        block_table=block_table,
        blocks=key_blocks,
        write_positions=new_write_positions,
        values=update_key_chunk,
        page_batch_index=page_batch_index,
    )

    _aux_write_kv(
        block_table=block_table,
        blocks=value_blocks,
        write_positions=new_write_positions,
        values=update_value_chunk,
        page_batch_index=page_batch_index,
    )

    recent_indexer = torch.minimum(chunk_indexer[None, :], torch.clamp(true_update_size[:, None] - 1, min=0))

    recent_info_indexer = (recent_info_position[:, None] + recent_indexer) % recent_info.size(1)

    # Update the info about last window positions
    non_empty_update = (true_update_size[:, None] > 0).to(torch.int32)

    recent_info[page_batch_index[:, None], recent_info_indexer, 0] = (
        new_write_positions * non_empty_update
        + recent_info[page_batch_index[:, None], recent_info_indexer, 0] * (1 - non_empty_update)
    ).to(torch.int32)

    eviction_info_chunk = torch.cat(
        [
            eviction_info_chunk[:, 1:],
            torch.zeros_like(eviction_info_chunk[:, [0]]),
        ],
        dim=-1,
    )
    recent_info[page_batch_index[:, None], recent_info_indexer, 1] = (
        eviction_info_chunk[:, :] * non_empty_update
        + recent_info[page_batch_index[:, None], recent_info_indexer, 1] * (1 - non_empty_update)
    ).to(torch.int32)

    recent_info_position[...] += true_update_size
    cache_seq_lenghts[...] += new_positions_used

    require_free_pages = torch.logical_and(new_positions_used > 0, cache_seq_lenghts % block_size == 0)
    return require_free_pages


class DMSCache(Cache):
    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        accomodate_min_initial_context_length: int = 2048,
        block_size: int = 256,
    ):
        super().__init__(
            layer_class_to_replicate=functools.partial(
                DMSPagedCacheLayer,
                dms_window_size=dms_window_size,
                max_context_length=max_context_length,
                accomodate_min_initial_context_length=accomodate_min_initial_context_length,
                block_size=block_size,
            ),
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    def to_legacy_cache(self):
        raise NotImplementedError("Not Supported")

    @classmethod
    def from_legacy_cache(cls, *args, **kwargs):
        raise NotImplementedError("Not Supported")

    def early_initialization(self, *args, **kwargs):
        raise NotImplementedError("Not Supported")

    def __iter__(self):
        raise NotImplementedError("Not Supported")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override Cache.update() to skip built-in offloading.

        The base class offloads the layer to CPU immediately after update(),
        but DMS needs the cache on GPU for flash_attn_with_kvcache() which
        runs after update(). Offloading is handled in the decoder loop instead.
        """
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return keys, values

    def __getitem__(self, layer_idx: int):
        assert layer_idx < len(self.layers)
        return self.layers[layer_idx]


class DMSPagedCacheLayer(CacheLayerMixin):
    def __init__(
        self,
        dms_window_size: int,
        max_context_length: int,
        block_size: int = 256,
        growth_factor: float = 1.5,
        accomodate_min_initial_context_length: int = 4096,
    ):
        super().__init__()
        assert block_size <= dms_window_size
        self.block_size = block_size
        self.dms_window_size = dms_window_size
        self.prefill_chunk_size = max(self.dms_window_size - 2, block_size)
        assert self.prefill_chunk_size > 0
        self.growth_factor = growth_factor
        self.min_initial_context_length = accomodate_min_initial_context_length
        self.max_context_length = max_context_length

        self.max_blocks_per_sequence = ceil_int_div(self.max_context_length, self.block_size)

        self.key_blocks = None
        self.value_blocks = None
        self.block_table = None
        self.free_page_ids = None
        self.cache_seq_lengths = None
        self.recent_info = None  # Position and eviction info of last window_size keys/values
        self.recent_info_position = None

        self.device = None

        self.cumulative_length = 0

    def offload(self):
        if self.key_blocks is not None:
            self.key_blocks = self.key_blocks.to("cpu", non_blocking=True)
            self.value_blocks = self.value_blocks.to("cpu", non_blocking=True)
            self.block_table = self.block_table.to("cpu", non_blocking=True)
            self.free_page_ids = self.free_page_ids.to("cpu", non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to("cpu", non_blocking=True)
            self.recent_info = self.recent_info.to("cpu", non_blocking=True)
            self.recent_info_position = self.recent_info_position.to("cpu", non_blocking=True)

    def prefetch(self):
        if self.key_blocks is not None and self.key_blocks.device != self.device:
            self.key_blocks = self.key_blocks.to(self.device, non_blocking=True)
            self.value_blocks = self.value_blocks.to(self.device, non_blocking=True)
            self.block_table = self.block_table.to(self.device, non_blocking=True)
            self.free_page_ids = self.free_page_ids.to(self.device, non_blocking=True)
            self.cache_seq_lengths = self.cache_seq_lengths.to(self.device, non_blocking=True)
            self.recent_info = self.recent_info.to(self.device, non_blocking=True)
            self.recent_info_position = self.recent_info_position.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        print(f"reset {self.key_blocks is not None}")
        if self.key_blocks is not None:
            self.key_blocks = None
            self.value_blocks = None
            self.block_table = None
            self.free_page_ids = None
            self.cache_seq_lengths = None
            self.recent_info = None
            self.recent_info_position = None
            gc.collect()
            torch.cuda.empty_cache()
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        assert False  # No support for beam search at this point

    def _get_free_pages(self, num_pages: int):
        while len(self.free_page_ids) < num_pages:

            def expand_blocks(blocks: torch.Tensor):
                return torch.cat(
                    [
                        blocks,
                        blocks.new_zeros(
                            (
                                float_ceil(blocks.size(0) * self.growth_factor) - blocks.size(0),
                                blocks.size(1),
                                blocks.size(2),
                                blocks.size(3),
                            )
                        ),
                    ],
                    dim=0,
                )

            old_num_blocks = self.key_blocks.size(0)
            self.key_blocks = expand_blocks(self.key_blocks)
            self.value_blocks = expand_blocks(self.value_blocks)
            assert self.key_blocks.size(0) == self.value_blocks.size(0)
            self.free_page_ids = torch.cat(
                [
                    self.free_page_ids,
                    torch.arange(
                        old_num_blocks,
                        self.key_blocks.size(0),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        result = self.free_page_ids[:num_pages]
        assert result.size() == (num_pages,)
        self.free_page_ids = self.free_page_ids[num_pages:]
        return result

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads, _, self.head_dim = key_states.shape

        self.page_batch = self.batch_size * self.num_heads

        initial_num_blocks = max(
            ceil_int_div(self.min_initial_context_length, self.block_size) * self.page_batch,
            self.page_batch,
        )

        self.block_table = -torch.ones(
            self.page_batch,
            self.max_blocks_per_sequence + 1,  # +1 for handling full cache case
            dtype=torch.int32,
            device=self.device,
        )
        self.key_blocks = torch.zeros(
            (initial_num_blocks, self.block_size, 1, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.value_blocks = torch.zeros(
            (initial_num_blocks, self.block_size, 1, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.free_page_ids = torch.arange(0, initial_num_blocks, dtype=torch.int32, device=self.device)

        self.cache_seq_lengths = torch.zeros(self.page_batch, dtype=torch.int32, device=self.device)

        self.recent_info = torch.zeros(
            (self.page_batch, self.dms_window_size, 2),
            dtype=torch.int32,
            device=self.device,
        )

        self.recent_info_position = torch.zeros((self.page_batch,), dtype=torch.int32, device=self.device)

        self.block_table[:, 0] = self._get_free_pages(self.block_table.size(0))

    def _handle_page_allocation(self, requires_free_page: torch.Tensor, page_batch_index: torch.Tensor):
        if requires_free_page.any():
            req_free_pages = page_batch_index[requires_free_page]
            free_pages = self._get_free_pages(len(req_free_pages))

            self.block_table[
                req_free_pages,
                self.cache_seq_lengths[req_free_pages] // self.block_size,
            ] = free_pages

    def _update_single(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        eviction_info: torch.Tensor,
    ):
        batch_x_head, seq_len, head_dim = key_states.size()
        page_batch_index = torch.arange(batch_x_head, dtype=torch.int32, device=self.device)

        assert seq_len == 1

        requires_free_page = _aux_update_single(
            key_states=key_states,
            value_states=value_states,
            eviction_info=eviction_info,
            recent_info=self.recent_info,
            recent_info_position=self.recent_info_position,
            block_table=self.block_table,
            key_blocks=self.key_blocks,
            value_blocks=self.value_blocks,
            cache_seq_lenghts=self.cache_seq_lengths,
            page_batch_index=page_batch_index,
        )

        self._handle_page_allocation(requires_free_page=requires_free_page, page_batch_index=page_batch_index)

    # NOTE: Prefill is not yet optimized
    def _update_many(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        eviction_info: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ):
        # Assume key and value states are left padded, e.g., [_, _, _, 1, 2, 3, 4]
        page_batch, seq_len, head_dim = key_states.size()
        assert page_batch == self.page_batch
        assert head_dim == self.head_dim
        assert eviction_info.size() == (page_batch, seq_len)
        assert sequence_lengths.ndim == 1

        assert sequence_lengths.min() > 0, sequence_lengths

        start_positions = seq_len - sequence_lengths

        end_positions = start_positions + sequence_lengths

        page_batch_index = torch.arange(page_batch, dtype=torch.int32, device=self.device)

        while (start_positions < end_positions).any():
            chunk_indexer = torch.arange(self.prefill_chunk_size, dtype=torch.int32, device=self.device)[None, :]

            update_mask = chunk_indexer < (self.block_size - (self.cache_seq_lengths[:, None] % self.block_size))

            chunk_indexer = start_positions[:, None] + chunk_indexer

            update_mask = torch.logical_and(update_mask, chunk_indexer < end_positions[:, None])

            chunk_indexer = torch.clamp(torch.minimum(chunk_indexer, end_positions[:, None] - 1), min=0)

            true_update_size = update_mask.to(torch.int32).sum(dim=1)

            chunk_indexer = torch.clamp(
                torch.minimum(
                    chunk_indexer,
                    start_positions[:, None] + true_update_size[:, None] - 1,
                ),
                min=0,
            )

            key_chunk = key_states[page_batch_index[:, None], chunk_indexer]
            value_chunk = value_states[page_batch_index[:, None], chunk_indexer]
            eviction_info_chunk = eviction_info[page_batch_index[:, None], chunk_indexer]

            requires_free_page = _aux_update_many_handle_single_chunk(
                update_key_chunk=key_chunk,
                update_value_chunk=value_chunk,
                eviction_info_chunk=eviction_info_chunk,
                block_table=self.block_table,
                key_blocks=self.key_blocks,
                value_blocks=self.value_blocks,
                cache_seq_lenghts=self.cache_seq_lengths,
                recent_info=self.recent_info,
                recent_info_position=self.recent_info_position,
                page_batch_index=page_batch_index,
                update_mask=update_mask,
                true_update_size=true_update_size,
            )

            self._handle_page_allocation(requires_free_page=requires_free_page, page_batch_index=page_batch_index)

            start_positions[...] += true_update_size

    def get_contiguous_cache(
        self, right_padded: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.key_blocks is not None
        num_blocks_per_sequence = self.cache_seq_lengths.max().item() // self.block_size + 1
        blocks_to_retrieve = self.block_table[:, :num_blocks_per_sequence]

        # page_batch_index = torch.arange(self.batch_size * self.num_heads, dtype=torch.int32, device=self.device)[:, None]

        max_length = self.cache_seq_lengths.max().item()

        def handle_one(
            blocks: torch.Tensor,
            blocks_to_retrieve: torch.Tensor,
            max_length: int,
            num_blocks_per_sequence: int,
        ):
            retrieved = blocks[blocks_to_retrieve].reshape(
                self.page_batch,
                num_blocks_per_sequence * self.block_size,
                self.head_dim,
            )
            retrieved = retrieved[:, :max_length, :]

            recent_info_size = torch.clamp(self.cache_seq_lengths, max=self.dms_window_size)

            window_index = torch.arange(self.dms_window_size, device=self.device, dtype=torch.int32)
            adjusted_window_index = torch.minimum(
                window_index[None, :], torch.clamp(recent_info_size[:, None] - 1, min=0)
            )

            last_pos_data_ptr = (self.recent_info_position[:, None] - 1 - adjusted_window_index) % self.dms_window_size

            page_batch_index = torch.arange(self.page_batch, device=self.device, dtype=torch.int32)

            window_positions = self.recent_info[page_batch_index[:, None], last_pos_data_ptr, 0]

            permutation_index = self.cache_seq_lengths[:, None] - 1 - adjusted_window_index

            assert (permutation_index >= 0).all()

            permutation = torch.arange(max_length, device=blocks.device, dtype=torch.int32)[None, :]
            permutation = torch.minimum(permutation, self.cache_seq_lengths[:, None] - 1)
            permutation = torch.broadcast_to(permutation, (self.page_batch, max_length))
            # Mark which position *values* are window positions using scatter,
            # then look up each permutation value — O(page_batch × max_length)
            # instead of the old O(page_batch × max_length × window_size) broadcast.
            is_window = torch.zeros(
                self.page_batch, max_length, dtype=torch.bool, device=self.device
            )
            valid_window_pos = torch.clamp(window_positions.long(), 0, max_length - 1)
            is_window.scatter_(1, valid_window_pos, True)
            # gather by permutation values so padded positions (clamped to
            # cache_seq_lengths-1, which is always a window position) stay
            # correctly marked as window entries — matches old behaviour.
            non_window_positions = ~torch.gather(is_window, 1, permutation.long())

            result_permutation = torch.zeros_like(permutation)
            result_permutation[page_batch_index[:, None], permutation_index] = window_positions

            # Triton-accelerated: compact non-window positions to front, then assign vectorized
            compacted = compact_by_bool(permutation, non_window_positions)
            num_non_window = non_window_positions.to(torch.int32).sum(dim=-1)
            pos_range = torch.arange(max_length, device=self.device, dtype=torch.int32)[None, :]
            write_mask = pos_range < num_non_window[:, None]
            result_permutation[write_mask] = compacted[write_mask]

            return retrieved[page_batch_index[:, None], result_permutation]

        retrieved_keys = handle_one(
            blocks=self.key_blocks,
            blocks_to_retrieve=blocks_to_retrieve,
            max_length=max_length,
            num_blocks_per_sequence=num_blocks_per_sequence,
        )
        retrieved_values = handle_one(
            blocks=self.value_blocks,
            blocks_to_retrieve=blocks_to_retrieve,
            max_length=max_length,
            num_blocks_per_sequence=num_blocks_per_sequence,
        )
        cache_seq_lenghts = self.cache_seq_lengths

        page_batch_index = torch.arange(self.page_batch, device=self.device, dtype=torch.int32)

        eviction_info_indexer = torch.arange(self.dms_window_size, device=self.device, dtype=torch.int32)
        eviction_info_indexer = torch.minimum(
            eviction_info_indexer[None, :],
            _aux_get_recent_position_size(
                cache_seq_lenghts=self.cache_seq_lengths,
                dms_window_size=self.dms_window_size,
            )[:, None]
            - 1,
        )

        eviction_info_indexer = (
            _aux_get_first_recent_position(
                recent_info_position=self.recent_info_position,
                cache_seq_lenghts=self.cache_seq_lengths,
                dms_window_size=self.dms_window_size,
            )[:, None]
            + eviction_info_indexer
        )

        eviction_info_indexer = eviction_info_indexer % self.dms_window_size

        eviction_info = self.recent_info[page_batch_index[:, None], eviction_info_indexer, 1]

        if not right_padded:
            # Triton-accelerated left padding (replaces Python loop)
            retrieved_keys = left_pad_2d(retrieved_keys, cache_seq_lenghts)
            retrieved_values = left_pad_2d(retrieved_values, cache_seq_lenghts)
            cache_seq_lenghts = cache_seq_lenghts.reshape(self.batch_size, self.num_heads)
            eviction_info = left_pad_2d(
                eviction_info.unsqueeze(-1),
                _aux_get_recent_position_size(
                    cache_seq_lenghts=self.cache_seq_lengths,
                    dms_window_size=self.dms_window_size,
                ),
            ).squeeze(-1)

        retrieved_keys = retrieved_keys.reshape(
            self.batch_size, self.num_heads, max_length, self.head_dim
        ).contiguous()
        retrieved_values = retrieved_values.reshape(
            self.batch_size, self.num_heads, max_length, self.head_dim
        ).contiguous()
        cache_seq_lenghts = cache_seq_lenghts.reshape(self.batch_size, self.num_heads)
        eviction_info = eviction_info.reshape(self.batch_size, self.num_heads, -1)

        return retrieved_keys, retrieved_values, cache_seq_lenghts, eviction_info

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any],
    ):
        eviction_info = cache_kwargs["eviction_info"]
        sequence_lengths = cache_kwargs["sequence_lengths"]
        cumulative_length = cache_kwargs["cumulative_length"]

        if self.key_blocks is None:
            self.lazy_initialization(key_states)

        batch, head, seq_len, head_dim = key_states.size()
        assert key_states.size() == value_states.size()
        assert key_states.size()[:3] == eviction_info.size()
        assert sequence_lengths is None or sequence_lengths.size() == (batch, head)

        assert batch * head == self.page_batch
        assert self.head_dim == head_dim

        key_states = key_states.reshape(self.page_batch, seq_len, head_dim)
        value_states = value_states.reshape(self.page_batch, seq_len, head_dim)
        eviction_info = eviction_info.reshape(self.page_batch, seq_len)
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths.reshape(self.page_batch)

        if seq_len == 1:
            assert sequence_lengths is None or (sequence_lengths == 1).all()
            assert cumulative_length == 1
            self._update_single(
                key_states=key_states,
                value_states=value_states,
                eviction_info=eviction_info,
            )
        else:
            self._update_many(
                key_states=key_states,
                value_states=value_states,
                eviction_info=eviction_info,
                sequence_lengths=sequence_lengths,
            )

        self.cumulative_length += cumulative_length
        return None, None

    def get_block_table(self):
        return self.block_table

    def get_key_blocks(self):
        return self.key_blocks

    def get_value_blocks(self):
        return self.value_blocks

    def get_seq_lengths(self):
        return self.cache_seq_lengths

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Returns the length and offset of the cache, used to generate the mask."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return self.max_context_length
