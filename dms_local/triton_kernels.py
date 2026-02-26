# SPDX-License-Identifier: Apache-2.0

"""
Triton CUDA kernels replacing Python batch-loops in DMS inference.

These replace serial CPU-GPU-sync loops with parallel GPU kernels,
targeting the prefill path of NVIDIA's DMS (Dynamic Memory Sparsification)
implementation from nvidia/Qwen3-8B-DMS-8x.

Kernels:
    1. left_pad_2d       - replaces left_pad_one() in dms_cache.py
    2. scatter_by_index   - replaces restore_order() in dms_attention.py
    3. bool_gather_left_pad - replaces convert_to_left_padding() inner loop
    4. compact_by_bool    - replaces get_contiguous_cache() compaction loop
    5. dms_fused_prefill  - replaces the chunked dms_prefill_attention() loop
                            with single-pass flash attention + DMS masking
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: left_pad_2d
# Replaces left_pad_one() in dms_cache.py get_contiguous_cache
# Given x[N, S, D] and lens[N], produce output where each row is right-aligned
# ---------------------------------------------------------------------------

@triton.jit
def _left_pad_2d_kernel(
    src_ptr, dst_ptr, lens_ptr,
    S, D,
    src_stride_n, src_stride_s,
    dst_stride_n, dst_stride_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    d_start = tl.program_id(1) * BLOCK_D

    length = tl.load(lens_ptr + n)
    pad = S - length

    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    for s_start in range(0, S, BLOCK_S):
        s_offs = tl.arange(0, BLOCK_S) + s_start
        s_mask = s_offs < length

        src_idx = n * src_stride_n + s_offs[:, None] * src_stride_s + d_offs[None, :]
        vals = tl.load(src_ptr + src_idx, mask=(s_mask[:, None] & d_mask[None, :]), other=0.0)

        dst_s = s_offs + pad
        dst_idx = n * dst_stride_n + dst_s[:, None] * dst_stride_s + d_offs[None, :]
        tl.store(dst_ptr + dst_idx, vals, mask=(s_mask[:, None] & d_mask[None, :]))


def left_pad_2d(x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Left-pad each row of x[N, S, D] according to lens[N].

    x[n, :lens[n], :] is moved to output[n, S-lens[n]:, :], rest is zero.
    """
    N, S, D = x.shape
    assert lens.shape == (N,)
    dst = torch.zeros_like(x)
    if N == 0 or S == 0 or D == 0:
        return dst

    lens_i32 = lens.to(torch.int32).contiguous()

    BLOCK_S = min(triton.next_power_of_2(S), 1024)
    BLOCK_D = min(triton.next_power_of_2(D), 128)

    grid = (N, triton.cdiv(D, BLOCK_D))
    _left_pad_2d_kernel[grid](
        x, dst, lens_i32,
        S, D,
        x.stride(0), x.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )
    return dst


# ---------------------------------------------------------------------------
# Kernel 2: scatter_by_index
# Replaces restore_order() in dms_attention.py
# For each batch: dst[b, indices[b, j], :] = src[b, src_offset + j, :]
# ---------------------------------------------------------------------------

@triton.jit
def _scatter_by_index_kernel(
    src_ptr, dst_ptr, idx_ptr, idx_lens_ptr,
    S_src, S_dst, D,
    src_stride_b, src_stride_s,
    dst_stride_b, dst_stride_s,
    idx_stride_b, idx_stride_s,
    BLOCK_D: tl.constexpr,
):
    # Grid: (max_idx_len, B) — max_idx_len in X to avoid CUDA grid Y limit (65535)
    j = tl.program_id(0)
    b = tl.program_id(1)

    idx_len = tl.load(idx_lens_ptr + b)
    active = j < idx_len

    if active:
        # Source position: last idx_len elements of src[b]
        src_s = S_src - idx_len + j
        # Destination position from index array
        dst_s = tl.load(idx_ptr + b * idx_stride_b + j * idx_stride_s)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        src_off = b * src_stride_b + src_s * src_stride_s + d_offs
        vals = tl.load(src_ptr + src_off, mask=d_mask, other=0.0)

        dst_off = b * dst_stride_b + dst_s * dst_stride_s + d_offs
        tl.store(dst_ptr + dst_off, vals, mask=d_mask)


def scatter_by_index(
    src: torch.Tensor,
    indices: torch.Tensor,
    idx_lens: torch.Tensor,
    dst_seq_len: int,
) -> torch.Tensor:
    """Scatter last idx_lens[b] elements of src[b] to positions indices[b, :] in dst.

    src: [B, S_src, D]
    indices: [B, max_idx_len] (padded, int64 or int32)
    idx_lens: [B] number of valid indices per batch
    dst_seq_len: sequence length of output

    Returns: [B, dst_seq_len, D] with scattered values
    """
    B, S_src, D = src.shape
    dst = torch.zeros((B, dst_seq_len, D), device=src.device, dtype=src.dtype)
    if B == 0 or D == 0:
        return dst

    max_idx_len = indices.shape[1]
    idx_lens_i32 = idx_lens.to(torch.int32).contiguous()
    indices_i32 = indices.to(torch.int32).contiguous()

    BLOCK_D = min(triton.next_power_of_2(D), 128)

    grid = (max_idx_len, B)
    _scatter_by_index_kernel[grid](
        src, dst, indices_i32, idx_lens_i32,
        S_src, dst_seq_len, D,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        indices_i32.stride(0), indices_i32.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return dst


# ---------------------------------------------------------------------------
# Kernel 3: bool_gather_left_pad
# Replaces the inner loop of convert_to_left_padding() in dms_attention.py
# Given tensor[B, H, S, D] and bool mask[B, S], gather True positions and
# left-pad to original S length.
# ---------------------------------------------------------------------------

@triton.jit
def _bool_gather_left_pad_kernel(
    src_ptr, dst_ptr, gather_idx_ptr, counts_ptr,
    H, S, D, max_count,
    src_stride_b, src_stride_h, src_stride_s,
    dst_stride_b, dst_stride_h, dst_stride_s,
    gidx_stride_b, gidx_stride_s,
    BLOCK_D: tl.constexpr,
):
    # Grid: (max_count, B, H) — max_count in X to avoid CUDA grid Y/Z limit (65535)
    j = tl.program_id(0)  # index into gathered positions
    b = tl.program_id(1)
    h = tl.program_id(2)

    count = tl.load(counts_ptr + b)
    active = j < count

    if active:
        pad = S - count
        src_s = tl.load(gather_idx_ptr + b * gidx_stride_b + j * gidx_stride_s)
        dst_s = pad + j

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        src_off = b * src_stride_b + h * src_stride_h + src_s * src_stride_s + d_offs
        vals = tl.load(src_ptr + src_off, mask=d_mask, other=0.0)

        dst_off = b * dst_stride_b + h * dst_stride_h + dst_s * dst_stride_s + d_offs
        tl.store(dst_ptr + dst_off, vals, mask=d_mask)


def bool_gather_left_pad(
    src: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather positions where mask is True along dim=2, left-pad result.

    src: [B, H, S, D]
    mask: [B, S] (boolean)

    Returns:
        dst: [B, H, S, D] with gathered+left-padded data
        gather_indices: [B, max_count] indices of True positions (for restore_order)
        counts: [B] number of True positions per batch
    """
    B, H, S, D = src.shape
    assert mask.shape == (B, S)

    # Pre-compute gather indices using PyTorch (fast prefix-sum)
    counts = mask.sum(dim=-1)  # [B]
    max_count = int(counts.max().item())

    # Build padded gather index tensor
    # For each batch, collect positions where mask is True
    sorted_indices = torch.argsort(~mask, dim=-1, stable=True)  # True positions first
    gather_indices = sorted_indices[:, :max_count].contiguous()  # [B, max_count]

    dst = torch.zeros_like(src)
    if B == 0 or max_count == 0 or D == 0:
        return dst, gather_indices, counts

    counts_i32 = counts.to(torch.int32).contiguous()
    gather_i32 = gather_indices.to(torch.int32).contiguous()

    BLOCK_D = min(triton.next_power_of_2(D), 128)

    grid = (max_count, B, H)
    _bool_gather_left_pad_kernel[grid](
        src, dst, gather_i32, counts_i32,
        H, S, D, max_count,
        src.stride(0), src.stride(1), src.stride(2),
        dst.stride(0), dst.stride(1), dst.stride(2),
        gather_i32.stride(0), gather_i32.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return dst, gather_indices, counts


# ---------------------------------------------------------------------------
# Kernel 4: compact_by_bool
# Replaces the loop at get_contiguous_cache line 685:
#   for i in range(page_batch):
#       result[i, :num_true[i]] = src[i, mask[i]]
# Given src[N, S] and bool mask[N, S], compact True positions to front.
# ---------------------------------------------------------------------------

@triton.jit
def _compact_by_bool_kernel(
    src_ptr, dst_ptr, gather_idx_ptr, counts_ptr,
    N, S, max_count,
    src_stride_n, src_stride_s,
    dst_stride_n, dst_stride_s,
    gidx_stride_n, gidx_stride_s,
):
    # Grid: (max_count, N) — max_count in X to avoid CUDA grid Y limit (65535)
    j = tl.program_id(0)
    n = tl.program_id(1)

    count = tl.load(counts_ptr + n)
    active = j < count

    if active:
        src_s = tl.load(gather_idx_ptr + n * gidx_stride_n + j * gidx_stride_s)
        val = tl.load(src_ptr + n * src_stride_n + src_s * src_stride_s)
        tl.store(dst_ptr + n * dst_stride_n + j * dst_stride_s, val)


def compact_by_bool(
    src: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compact True positions of src[N, S] to front of output[N, S].

    src: [N, S]
    mask: [N, S] (boolean)

    Returns: [N, S] with True positions compacted to front, rest unchanged
    """
    N, S = src.shape
    assert mask.shape == (N, S)

    counts = mask.sum(dim=-1)  # [N]
    max_count = int(counts.max().item())

    sorted_indices = torch.argsort(~mask, dim=-1, stable=True)
    gather_indices = sorted_indices[:, :max_count].contiguous()

    dst = src.clone()
    if N == 0 or max_count == 0:
        return dst

    counts_i32 = counts.to(torch.int32).contiguous()
    gather_i32 = gather_indices.to(torch.int32).contiguous()

    grid = (max_count, N)
    _compact_by_bool_kernel[grid](
        src, dst, gather_i32, counts_i32,
        N, S, max_count,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        gather_i32.stride(0), gather_i32.stride(1),
    )
    return dst


# ---------------------------------------------------------------------------
# Kernel 5: DMS Fused Prefill Attention
# Replaces the chunked loop in dms_prefill_attention() in dms_attention.py
# Single-pass flash attention with DMS eviction masking built in.
# ---------------------------------------------------------------------------

@triton.jit
def _dms_flash_attn_fwd_kernel(
    Q, K, V, Evicted, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kn, stride_kk,
    stride_vz, stride_vn, stride_vk,
    stride_ez,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX,       # total KV sequence length
    Q_CTX,       # query sequence length
    Q_OFFSET,    # old_seq_len (positional offset for queries)
    q_per_kv,
    sm_scale,
    window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash attention kernel with DMS eviction masking.

    Grid: (page_batch * q_per_kv, cdiv(Q_CTX, BLOCK_M))

    Visibility rule for query q attending to key k:
        visible = (k <= q) AND (distance(q,k) <= window_size OR NOT evicted[k])
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    z = pid_bh // q_per_kv   # page_batch index (indexes K, V, Evicted)
    h = pid_bh % q_per_kv    # query head index (indexes Q)

    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m < Q_CTX
    q_abs = offs_m + Q_OFFSET   # absolute position in full sequence

    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block: [BLOCK_M, BLOCK_D]
    q_ptrs = (Q + z * stride_qz + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal: only process keys up to max query absolute position + 1
    causal_end = (pid_m + 1) * BLOCK_M + Q_OFFSET
    if causal_end > N_CTX:
        causal_end = N_CTX

    # Base pointers for K, V, Evicted (indexed by page_batch z)
    k_base = K + z * stride_kz
    v_base = V + z * stride_vz
    e_base = Evicted + z * stride_ez

    for start_n in range(0, causal_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_CTX

        # Load K transposed: [BLOCK_D, BLOCK_N] for Q @ K^T
        k_ptrs = k_base + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_block = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)

        # S = Q @ K^T: [BLOCK_M, BLOCK_N]
        s = tl.dot(q_block, k_block) * sm_scale

        # DMS visibility mask
        distance = q_abs[:, None] - offs_n[None, :]
        causal = distance >= 0
        in_window = distance <= window_size

        # Eviction state (int8: 0=visible, 1=evicted)
        evict_vals = tl.load(e_base + offs_n, mask=n_mask, other=1)
        not_evicted = evict_vals == 0

        visible = causal & (in_window | not_evicted[None, :]) & n_mask[None, :] & q_mask[:, None]
        s = tl.where(visible, s, float('-inf'))

        # Online softmax update (with safe -inf handling)
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # Avoid NaN from -inf - (-inf): when m_i is -inf, no previous data to rescale
        alpha = tl.where(m_i > float('-inf'), tl.exp(m_i - m_new), 0.0)
        # When m_new is -inf, all scores are -inf, so p should be 0
        p = tl.where(m_new[:, None] > float('-inf'), tl.exp(s - m_new[:, None]), 0.0)
        p = tl.where(visible, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V: [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_block = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc += tl.dot(p.to(v_block.dtype), v_block)
        m_i = m_new

    # Final normalization
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    # Store output
    out_ptrs = (Out + z * stride_oz + h * stride_oh
                + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


def dms_fused_prefill(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    old_k: torch.Tensor,
    old_v: torch.Tensor,
    old_seq_lengths: torch.Tensor,
    previous_evicted: torch.Tensor,
    new_decisions: torch.Tensor,
    window_size: int,
    attn_mask: torch.Tensor,
    attn_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused DMS prefill attention replacing the chunked loop.

    Pre-computes eviction state for all positions, then runs a single
    Triton flash attention kernel with DMS masking built in.

    Returns:
        attn_output: [page_batch, new_seq_len, q_per_kv, head_dim]
        new_seq_lengths: [batch, num_kv_heads]
    """
    batch, num_q_heads, new_seq_len, head_dim = q.shape
    _, num_kv_heads, old_seq_len, _ = old_k.shape
    q_per_kv = num_q_heads // num_kv_heads
    page_batch = batch * num_kv_heads

    # Reshape to page_batch layout
    q = q.reshape(page_batch, q_per_kv, new_seq_len, head_dim)
    new_k = new_k.reshape(page_batch, 1, new_seq_len, head_dim)
    new_v = new_v.reshape(page_batch, 1, new_seq_len, head_dim)
    new_decisions = new_decisions.reshape(page_batch, 1, new_seq_len)
    old_seq_lengths = old_seq_lengths.reshape(page_batch)

    # Concatenate old + new KV
    if old_seq_len > 0:
        old_k = old_k.reshape(page_batch, 1, old_seq_len, head_dim)
        old_v = old_v.reshape(page_batch, 1, old_seq_len, head_dim)
        previous_evicted = previous_evicted.reshape(page_batch, 1, old_seq_len)
        k = torch.cat([old_k, new_k], dim=2).squeeze(1)
        v = torch.cat([old_v, new_v], dim=2).squeeze(1)
    else:
        k = new_k.squeeze(1)
        v = new_v.squeeze(1)

    concat_seq_len = old_seq_len + new_seq_len

    # Build attention mask (same logic as original)
    new_attn_mask = torch.ones(
        (page_batch, 1, 1, concat_seq_len), dtype=torch.bool, device=q.device
    )
    if old_seq_len > 0:
        new_attn_mask[:, :, :, :old_seq_len] = (
            torch.arange(old_seq_len, device=q.device).flip(dims=(0,))[None, None, None, :]
            < old_seq_lengths[:, None, None, None]
        )
    if attn_mask is not None:
        assert attn_mask.ndim == 4
        am = attn_mask[:, :, -1, None, -new_seq_len:]
        am = am.broadcast_to((batch, num_kv_heads, 1, new_seq_len))
        am = am.reshape(page_batch, 1, 1, new_seq_len)
        new_attn_mask[:, :, :, -new_seq_len:] = am

    # Compute new_seq_lengths (needed for cache update)
    new_seq_lengths = (
        new_attn_mask[..., -new_seq_len:]
        .to(torch.int32).sum(dim=-1)[:, 0, 0]
        .reshape(page_batch)
    )

    # --- Pre-compute eviction state ---
    # Key insight: eviction is monotonic and each token's eviction is determined
    # independently by decisions[k+1]. No sequential chunk dependency.
    evicted = torch.zeros(page_batch, concat_seq_len, dtype=torch.int8, device=q.device)

    # Old tokens: inherit previous eviction state
    if old_seq_len > 0:
        evicted[:, :old_seq_len] = (previous_evicted[:, 0, :] > 0).to(torch.int8)

    # New tokens: decision at position k+1 evicts token k
    decisions_flat = new_decisions[:, 0, :]  # [page_batch, new_seq_len]
    if new_seq_len > 1:
        evicted[:, old_seq_len:old_seq_len + new_seq_len - 1] |= (
            (decisions_flat[:, 1:] > 0).to(torch.int8)
        )

    # Carry: decision[0] of new sequence evicts last old token
    if old_seq_len > 0:
        evicted[:, old_seq_len - 1] |= (decisions_flat[:, 0] > 0).to(torch.int8)

    # Apply attention mask: masked positions are evicted
    mask_flat = new_attn_mask[:, 0, 0, :]
    evicted |= (~mask_flat).to(torch.int8)

    evicted = evicted.contiguous()

    # --- Launch kernel ---
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)  # [page_batch, q_per_kv, new_seq_len, head_dim]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = head_dim

    grid = (page_batch * q_per_kv, triton.cdiv(new_seq_len, BLOCK_M))

    _dms_flash_attn_fwd_kernel[grid](
        q, k, v, evicted, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        evicted.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        concat_seq_len,
        new_seq_len,
        old_seq_len,   # Q_OFFSET
        q_per_kv,
        attn_scaling,
        window_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    # Transpose to match expected output: [page_batch, new_seq_len, q_per_kv, head_dim]
    attn_output = out.transpose(1, 2).contiguous()

    return attn_output, new_seq_lengths.reshape(batch, num_kv_heads)
