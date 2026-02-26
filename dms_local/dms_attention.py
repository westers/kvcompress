# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from nvidia/Qwen3-8B-DMS-8x (HuggingFace).
# Changes:
#   - convert_to_left_padding(): replaced Python batch-loop with Triton
#     bool_gather_left_pad kernel and vectorized mask construction
#   - restore_order(): replaced Python batch-loop with Triton
#     scatter_by_index kernel
#   - dms_prefill_attention call replaced with dms_fused_prefill:
#     a single-pass Triton flash attention kernel with DMS eviction
#     masking, replacing the chunked SDPA loop

from collections.abc import Callable
from typing import Optional

import torch
from flash_attn import flash_attn_with_kvcache

from .dms_cache import DMSCache, DMSPagedCacheLayer
from .triton_kernels import bool_gather_left_pad, scatter_by_index, dms_fused_prefill


MASK_CONST = -50_000.0


def make_mask(attn_score_mask: torch.Tensor, q_seq_len: int, k_seq_len: int, ws: int) -> torch.Tensor:
    """
    Constructs a DMS attention mask for the (unoptimized) prefill phase.

    Args:
        attn_score_mask:
            DMS decisions (alpha) logits. This is the inference code, so decisions[b,h,t] = either 0 or MASK_CONST,
            where 0 denotes no eviction, and MASK_CONST denotes eviction.
        q_seq_len:
            Number of queries.
        k_seq_len:
            Number of key/value pairs.
        ws:
            Window size. The window starts before the considered token, that is, ws=1 means that for attention,
            we keep both the considered and the previous token.
    """
    batch, head, seq_len = attn_score_mask.shape
    ones = torch.ones(q_seq_len, k_seq_len, device=attn_score_mask.device, dtype=torch.bfloat16)
    # Decide to evict the token now, but execute the decision after ws future steps
    log_alphas = attn_score_mask

    if q_seq_len == k_seq_len:
        expanded_attn_score_mask = torch.tril(ones, diagonal=-1 - ws) * log_alphas[:, :, None]

        inf = torch.triu(ones * MASK_CONST, diagonal=1)[None, None]
        expanded_attn_score_mask += inf
    elif q_seq_len > 1:
        expanded_attn_score_mask = (
            torch.tril(ones, diagonal=-1 - ws + (k_seq_len - q_seq_len)) * log_alphas[:, :, None]
        )
        inf = torch.triu(ones * MASK_CONST, diagonal=1 + (k_seq_len - q_seq_len))[None, None]
        expanded_attn_score_mask += inf
    else:
        ones[:, -ws - 1 :] = 0
        expanded_attn_score_mask = ones * log_alphas[:, :, None]
    return expanded_attn_score_mask.contiguous()


def prepare_data_for_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    previous_evicted: torch.Tensor,
    new_decisions: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares Q, K and V for the unoptimized prefill case.

    Args:
        q:
            Queries for the current chunked attn step, shape: [batch, num_attn_heads, seq_len_q, head_dim].
        k:
            Keys for the current chunked attn step, shape: [batch, num_key_value_heads, seq_len_kv, head_dim].
        v:
            Values for the current chunked attn step, shape: [batch, num_key_value_heads, seq_len_kv, head_dim].
        previous_evicted:
            Contains information about the KV cache. That is, previous_evicted[b,h,t]=True means that the token `t`
            is evicted/masked. Shape: [batch, head_kv, seq_len_kv].
        new_decisions:
            Contains information about whether to evict the token or not. This tensor is shifted by one to the right,
            that is, the token t+1 predicts whether to evict token t. Shape: [batch, head_kv, seq_len_q].
    """

    assert k.shape == v.shape
    batch, head, seq_len, head_dim = q.shape
    _, head_kv, seq_len_kv, _ = k.shape
    assert seq_len_kv >= seq_len

    assert len(attn_mask.shape) == 4, f"attn_mask.shape: {attn_mask.shape}"

    assert attn_mask.shape[0] in [
        batch,
        1,
    ], f"attn_mask.shape: {attn_mask.shape}"
    assert attn_mask.shape[1] in [
        head_kv,
        1,
    ], f"attn_mask.shape: {attn_mask.shape}"
    assert attn_mask.shape[2] in [
        seq_len,
        1,
    ], f"attn_mask.shape: {attn_mask.shape}"
    assert attn_mask.shape[3] in [seq_len_kv], f"attn_mask.shape: {attn_mask.shape}"

    # From the attention mask, we need only info about the attention of the last token (except for some edge cases)
    last_token_attn_mask = attn_mask[:, :, -1]
    assert len(last_token_attn_mask.shape) == len(previous_evicted.shape), (
        f"last_token_attn_mask.shape: {last_token_attn_mask.shape}, previous_evicted.shape: {previous_evicted.shape}"
    )
    last_token_attn_mask = last_token_attn_mask.broadcast_to(previous_evicted.shape)

    previous_evicted = torch.logical_or(previous_evicted, torch.logical_not(last_token_attn_mask))

    decisions = new_decisions
    assert decisions.shape == (
        batch,
        head_kv,
        seq_len,
    ), f"decisions.shape: {decisions.shape}, batch: {batch}, head: {head_kv}, seq_len: {seq_len}"

    attn_bias = decisions * MASK_CONST

    # Carry bias is carried to the last element of the previous sequence
    if seq_len == seq_len_kv:
        carry_bias = None
    else:
        carry_bias = attn_bias[:, :, 0]
        assert attn_bias.shape[-1] == seq_len

    # The decision to evict token `t` is produced by token `t+1`, so shift one elem to the left (padding with 0).
    attn_bias = torch.nn.functional.pad(attn_bias, (-1, 1), value=0)

    if attn_bias.shape[-1] != seq_len_kv:
        attn_bias = torch.cat(
            [
                torch.zeros(
                    (batch, head_kv, seq_len_kv - seq_len),
                    device=attn_bias.device,
                    dtype=attn_bias.dtype,
                ),
                attn_bias,
            ],
            dim=-1,
        )

    if carry_bias is not None:
        assert attn_bias.shape[-1] >= seq_len + 1
        attn_bias[:, :, -seq_len - 1] = carry_bias

    attn_bias = torch.minimum(attn_bias, (previous_evicted.to(attn_bias.dtype)) * MASK_CONST)

    evicted = attn_bias < 0

    attn_mask = make_mask(
        attn_score_mask=attn_bias,
        ws=window_size,
        q_seq_len=seq_len,
        k_seq_len=seq_len_kv,
    )

    attn_mask = torch.minimum(
        attn_mask,
        (torch.logical_not(last_token_attn_mask).to(attn_mask.dtype)[:, :, None, :]) * MASK_CONST,
    )

    attn_mask = attn_mask >= 0

    assert attn_mask.shape[-3:] == (head_kv, seq_len, seq_len_kv)

    return attn_mask, evicted


def dms_prefill_attention(
    q: torch.Tensor,
    new_k: torch.Tensor,  # left padded
    new_v: torch.Tensor,
    old_k: torch.Tensor,  # left padded
    old_v: torch.Tensor,
    old_seq_lengths: torch.Tensor,
    previous_evicted: torch.Tensor,
    new_decisions: torch.Tensor,
    window_size: int,
    attn_mask: torch.Tensor,
    attn_scaling: float,
    # Prefill is not optimized so we chunk it
    prefill_chunk_size: int = 512,
    return_eviction_info: bool = False,
    custom_attn_fn: Optional[Callable] = None,
):
    batch, num_q_heads, new_seq_len, head_dim = q.shape
    _, num_kv_heads, old_seq_len, _ = old_k.shape

    assert old_seq_lengths.shape == (batch, num_kv_heads), old_seq_lengths.shape

    q_per_kv = num_q_heads // num_kv_heads

    page_batch = batch * num_kv_heads

    q = q.reshape(page_batch, q_per_kv, new_seq_len, head_dim)
    new_k = new_k.reshape(page_batch, 1, new_seq_len, head_dim)
    new_v = new_v.reshape(page_batch, 1, new_seq_len, head_dim)
    new_decisions = new_decisions.reshape(page_batch, 1, new_seq_len)

    old_k = old_k.reshape(page_batch, 1, old_seq_len, head_dim)
    old_v = old_v.reshape(page_batch, 1, old_seq_len, head_dim)
    previous_evicted = previous_evicted.reshape(page_batch, 1, old_seq_len).clone()

    old_seq_lengths = old_seq_lengths.reshape(page_batch)

    new_attn_mask = torch.ones((page_batch, 1, 1, old_seq_len + new_seq_len), dtype=torch.bool, device=q.device)

    if old_seq_len > 0:
        new_attn_mask[:, :, :, :old_seq_len] = (
            torch.arange(old_seq_len, device=q.device).flip(dims=(0,))[None, None, None, :]
            < old_seq_lengths[:, None, None, None]
        )

    if attn_mask is not None:
        assert attn_mask.ndim == 4
        attn_mask = attn_mask[:, :, -1, None, -new_seq_len:]
        assert attn_mask.ndim == 4

        attn_mask = attn_mask.broadcast_to((batch, num_kv_heads, 1, new_seq_len))
        attn_mask = attn_mask.reshape(page_batch, 1, 1, new_seq_len)
        new_attn_mask[:, :, :, -new_seq_len:] = attn_mask

    new_seq_lengths = (new_attn_mask[..., -new_seq_len:].to(torch.int32).sum(dim=-1)[:, 0, 0]).reshape(page_batch)

    assert new_seq_lengths.min() > 0, new_seq_lengths

    # The op below assumes that new_k/new_v can only be padded in the first prefill
    k = torch.cat([old_k, new_k], dim=2)
    v = torch.cat([old_v, new_v], dim=2)

    previous_evicted = torch.nn.functional.pad(previous_evicted, (0, new_seq_len), value=0)

    _, _, concat_seq_len, _ = k.size()

    attn_output_list = []

    for start_pos in range(concat_seq_len - new_seq_len, concat_seq_len, prefill_chunk_size):
        end_pos = min(start_pos + prefill_chunk_size, concat_seq_len)

        q_chunk = q[:, :, start_pos:end_pos]
        k_chunk = k[:, :, :end_pos]
        v_chunk = v[:, :, :end_pos]
        previous_evicted_chunk = previous_evicted[:, :, :end_pos]
        new_decisions_chunk = new_decisions[:, :, start_pos:end_pos]
        new_attn_mask_chunk = new_attn_mask[:, :, :, :end_pos]

        attn_mask, evicted = prepare_data_for_attention(
            q=q_chunk,
            k=k_chunk,
            v=v_chunk,
            attn_mask=new_attn_mask_chunk,
            previous_evicted=previous_evicted_chunk,
            new_decisions=new_decisions_chunk,
            window_size=window_size,
        )

        assert k_chunk.size() == v_chunk.size()
        kc_batch, kc_head, kc_seq_len, kc_head_dim = k_chunk.size()
        qc_batch, qc_head, qc_seq_len, qc_head_dim = q_chunk.size()
        ac_batch, ac_head, _, ac_seq_len = attn_mask.size()
        error_msg = f"q_chunk.size: {q_chunk.size()} k_chunk.size: {k_chunk.size()} attn_mask.size: {attn_mask.size()}"
        assert kc_batch == qc_batch, error_msg
        assert kc_head_dim == qc_head_dim, error_msg
        assert kc_seq_len >= qc_seq_len, error_msg
        assert ac_batch == kc_batch, error_msg
        assert ac_head in [1, kc_head], error_msg
        assert ac_seq_len == kc_seq_len, error_msg

        attn_fn = custom_attn_fn if custom_attn_fn is not None else torch.nn.functional.scaled_dot_product_attention

        attn_output = attn_fn(
            q_chunk.contiguous(),
            k_chunk.contiguous(),
            v_chunk.contiguous(),
            attn_mask=attn_mask.contiguous(),
            dropout_p=0.0,
            scale=attn_scaling,
            is_causal=False,  # to make mask work
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output_list.append(attn_output)

        previous_evicted[:, :, :end_pos] = evicted[:, :, :end_pos]

    attn_output = torch.cat(attn_output_list, dim=1)
    assert attn_output.size(1) == new_seq_len

    if return_eviction_info:
        return (
            attn_output,
            new_seq_lengths.reshape(batch, num_kv_heads),
            previous_evicted,
        )
    else:
        return (
            attn_output,
            new_seq_lengths.reshape(batch, num_kv_heads),
        )


def convert_to_left_padding(
    tensors: list[torch.Tensor], attn_mask: torch.Tensor, q_seq_len: int
) -> tuple[list[torch.Tensor], torch.Tensor, tuple[int, list[torch.Tensor]]]:
    """
    Used in the unoptimized prefill case.

    Non optimized code to convert attention mask to the left padding style attention mask, e.g.,
        [0, 1, 1, 0, 1, 1, 0]  ---> [0, 0, 0, 1, 1, 1, 1].
    The provided tensors of shape [batch, head, seq_len, head_dim] will be appropriately permuted across the sequence
    length dimension to match the new attention mask. Note that the code also compresses the attention mask
    by taking the mask from the last non-masked token.
    """
    if attn_mask is None:
        return tensors, None, None

    assert attn_mask.ndim == 4  # [batch, head/1, q_seq_len, kv_seq_len
    batch, _, _, kv_seq_len = attn_mask.shape
    if attn_mask.size(2) == 1:
        attn_mask = attn_mask.broadcast_to(-1, -1, q_seq_len, -1)
    # Take the part corresponding to the new data; previous data is already stored without masked tokens
    attn_mask = attn_mask[:, :, :, -q_seq_len:]
    assert attn_mask.size(-1) == attn_mask.size(-2), f"attn_mask.size: {attn_mask.size()}"
    assert attn_mask.size(2) == q_seq_len, f"attn_mask.size: {attn_mask.size()}"

    # Get the last non-masked element for each batch entry
    attn_mask_indexer = torch.arange(q_seq_len, device=attn_mask.device)

    attn_mask_diagonal = attn_mask[:, :, attn_mask_indexer, attn_mask_indexer]
    last_non_masked_token = q_seq_len - 1 - torch.argmax(attn_mask_diagonal.flip(dims=(-1,)).to(torch.int8), dim=-1)

    last_non_masked_token_attn_mask = attn_mask.gather(
        dim=2,
        index=last_non_masked_token[:, :, None, None].broadcast_to(-1, -1, -1, q_seq_len),
    )
    assert last_non_masked_token_attn_mask.shape == (
        batch,
        attn_mask.shape[1],
        1,
        q_seq_len,
    ), f"last_non_masked_token_attn_mask: {last_non_masked_token_attn_mask.shape}"
    true_seq_len_t = last_non_masked_token_attn_mask.to(torch.int32).sum(dim=-1)[:, :, 0]
    assert true_seq_len_t.ndim == 2, f"true_seq_len: {true_seq_len_t.shape}"
    assert (true_seq_len_t > 0).all(), f"true_seq_len: {true_seq_len_t}"
    true_seq_len = true_seq_len_t[:, 0].cpu().tolist()

    # Boolean mask: [batch, seq_len]
    bool_mask = last_non_masked_token_attn_mask[:, 0, 0, :]  # [batch, q_seq_len]

    # Triton-accelerated: gather + left-pad all tensors in parallel
    new_tensors = []
    gather_indices = None
    counts = None
    for t in tensors:
        new_t, gather_indices, counts = bool_gather_left_pad(t, bool_mask)
        new_tensors.append(new_t)

    # Build restore_order_index as padded tensor (for Triton scatter_by_index)
    # gather_indices[b, :counts[b]] contains the original positions
    restore_order_index = []
    for b in range(batch):
        c = int(counts[b].item())
        restore_order_index.append(gather_indices[b, :c])

    # Build new attention mask vectorized
    seq_range = torch.arange(q_seq_len, device=attn_mask.device)[None, :]  # [1, q_seq_len]
    counts_expanded = counts[:, None]  # [batch, 1]
    new_attn_mask = (seq_range >= (q_seq_len - counts_expanded)).to(attn_mask.dtype)
    new_attn_mask = new_attn_mask[:, None, None, :]  # [batch, 1, 1, q_seq_len]

    return new_tensors, new_attn_mask, (q_seq_len, restore_order_index)


def restore_order(attn_output: torch.Tensor, restore_order_index_info: tuple[int, list[torch.Tensor]]) -> torch.Tensor:
    """
    Used in the unoptimized prefill case.

    For reverting the permutation across the sequence length dimension induced by the convert_to_left_padding function.
    That is to change the order from left-padding style to the original order.
    Accelerated with Triton scatter kernel.
    """
    assert attn_output.ndim == 3  # batch, seq_len, dim
    batch, _, _ = attn_output.size()
    org_seq_len = restore_order_index_info[0]
    restore_order_indices = restore_order_index_info[1]

    # Pad variable-length index lists into a single tensor for Triton
    idx_lens = torch.tensor([roi.size(0) for roi in restore_order_indices],
                            device=attn_output.device, dtype=torch.int32)
    max_len = int(idx_lens.max().item())
    padded_indices = torch.zeros((batch, max_len), device=attn_output.device, dtype=torch.int64)
    for b, roi in enumerate(restore_order_indices):
        padded_indices[b, :roi.size(0)] = roi

    return scatter_by_index(attn_output, padded_indices, idx_lens, org_seq_len)


def dms_attention(
    new_q: torch.Tensor,
    new_q_flash: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    decisions: torch.Tensor,
    attn_mask: torch.Tensor,
    layer_idx: int,
    dms_cache: DMSCache,
    attn_scaling: float,
    window_size: int,
    profiler=None,
) -> tuple[torch.Tensor, Optional[tuple[int, list[torch.Tensor]]]]:
    """
    Picks the attention implementation to use based on the new sequence length.

    For prefill an unoptimized code path is used that supports both left and right padding. For decode the optimized
    code path is used. Currently only single prefill is supported. In case of multiple prefills, e.g.,
    (prefill -> generate -> prefill -> ...), attention masks for consecutive prefills are not supported.
    """
    batch, q_head, new_seq_len, head_dim = new_q.size()
    _, k_head, _, _ = new_k.size()

    if new_seq_len == 1:
        # Decode
        dms_cache.update(
            key_states=new_k,
            value_states=new_v,
            layer_idx=layer_idx,
            cache_kwargs={
                "eviction_info": decisions,
                "sequence_lengths": None,
                "cumulative_length": 1,
            },
        )

        layer_cache: DMSPagedCacheLayer = dms_cache[layer_idx]

        if profiler is not None:
            k_cache, _, cache_seq_lengths, _ = layer_cache.get_contiguous_cache()
            profiler.capture(new_q, k_cache, cache_seq_lengths, layer_idx, decisions)

        attn_output = flash_attn_with_kvcache(
            new_q_flash,
            layer_cache.get_key_blocks(),
            layer_cache.get_value_blocks(),
            k=None,
            v=None,
            cache_seqlens=layer_cache.get_seq_lengths(),
            causal=True,
            softmax_scale=attn_scaling,
            block_table=layer_cache.get_block_table(),
        )

        return attn_output, None

    else:
        # Prefill; we want the input to be padded from the left
        (new_q, new_k, new_v, decisions), attn_mask, restore_order_info = convert_to_left_padding(
            tensors=[new_q, new_k, new_v, decisions[..., None]],
            attn_mask=attn_mask,
            q_seq_len=new_seq_len,
        )
        decisions = decisions.squeeze(-1)

        if len(dms_cache) > layer_idx:
            dms_cache_layer: DMSPagedCacheLayer = dms_cache[layer_idx]

        else:
            dms_cache_layer = None

        if dms_cache_layer is not None:
            old_k, old_v, old_seq_lengths, old_eviction_info = dms_cache_layer.get_contiguous_cache(right_padded=False)

            _, _, old_seq_len, _ = old_k.size()

            old_eviction_info = torch.nn.functional.pad(
                old_eviction_info, (old_seq_len - old_eviction_info.size(2), 0)
            )
            assert old_eviction_info.size() == (
                batch,
                k_head,
                old_seq_len,
            ), (
                f"old_eviction_info: {old_eviction_info.shape} batch: {batch} k_head: {k_head} old_seq_len: {old_seq_len}"
            )
        else:
            old_k = torch.empty((batch, k_head, 0, head_dim), device=new_k.device, dtype=new_k.dtype)
            old_v = torch.empty((batch, k_head, 0, head_dim), device=new_v.device, dtype=new_v.dtype)
            old_seq_lengths = torch.zeros((batch, k_head), device=new_k.device, dtype=new_k.dtype)
            old_eviction_info = torch.empty((batch, k_head, 0), device=new_k.device, dtype=new_k.dtype)

        attn_output, new_seq_lengths = dms_fused_prefill(
            q=new_q,
            new_k=new_k,
            new_v=new_v,
            old_k=old_k,
            old_v=old_v,
            old_seq_lengths=old_seq_lengths,
            previous_evicted=old_eviction_info,
            new_decisions=decisions,
            window_size=window_size,
            attn_mask=attn_mask,
            attn_scaling=attn_scaling,
        )

        dms_cache.update(
            key_states=new_k,
            value_states=new_v,
            layer_idx=layer_idx,
            cache_kwargs={
                "eviction_info": decisions,
                "sequence_lengths": new_seq_lengths,
                "cumulative_length": new_seq_len,
            },
        )

        return attn_output, restore_order_info
