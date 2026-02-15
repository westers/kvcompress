"""Test that fused prefill produces same output as chunked prefill."""

import torch
from dms_local.dms_attention import dms_prefill_attention
from dms_local.triton_kernels import dms_fused_prefill


def test_fused_vs_chunked():
    """Compare fused Triton kernel output with original chunked SDPA."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch = 1
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    seq_len = 1024  # small enough to be fast, large enough to test chunking
    window_size = 512

    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    new_k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    new_v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    # No old cache (initial prefill)
    old_k = torch.empty(batch, num_kv_heads, 0, head_dim, device=device, dtype=dtype)
    old_v = torch.empty(batch, num_kv_heads, 0, head_dim, device=device, dtype=dtype)
    old_seq_lengths = torch.zeros(batch, num_kv_heads, device=device, dtype=dtype)

    # Random binary decisions (0 or 1)
    decisions = (torch.rand(batch, num_kv_heads, seq_len, device=device) > 0.5).to(dtype)

    # No previous eviction
    previous_evicted = torch.empty(batch, num_kv_heads, 0, device=device, dtype=dtype)

    attn_scaling = 1.0 / (head_dim ** 0.5)

    # Run original chunked version
    out_chunked, seq_lens_chunked = dms_prefill_attention(
        q=q.clone(), new_k=new_k.clone(), new_v=new_v.clone(),
        old_k=old_k.clone(), old_v=old_v.clone(),
        old_seq_lengths=old_seq_lengths.clone(),
        previous_evicted=previous_evicted.clone(),
        new_decisions=decisions.clone(),
        window_size=window_size,
        attn_mask=None,
        attn_scaling=attn_scaling,
    )

    # Run fused version
    out_fused, seq_lens_fused = dms_fused_prefill(
        q=q.clone(), new_k=new_k.clone(), new_v=new_v.clone(),
        old_k=old_k.clone(), old_v=old_v.clone(),
        old_seq_lengths=old_seq_lengths.clone(),
        previous_evicted=previous_evicted.clone(),
        new_decisions=decisions.clone(),
        window_size=window_size,
        attn_mask=None,
        attn_scaling=attn_scaling,
    )

    print(f"Chunked output shape: {out_chunked.shape}")
    print(f"Fused output shape:   {out_fused.shape}")
    print(f"Seq lengths match: {torch.equal(seq_lens_chunked, seq_lens_fused)}")

    # Check numerical closeness (bf16 has limited precision)
    max_diff = (out_chunked.float() - out_fused.float()).abs().max().item()
    mean_diff = (out_chunked.float() - out_fused.float()).abs().mean().item()
    print(f"Max abs diff:  {max_diff:.6f}")
    print(f"Mean abs diff: {mean_diff:.6f}")

    # bf16 has ~3 decimal digits of precision, so tolerance should be generous
    if max_diff < 0.05 and mean_diff < 0.005:
        print("PASS: outputs match within bf16 tolerance")
    else:
        print("FAIL: outputs differ too much")
        # Show where the biggest differences are
        diff = (out_chunked.float() - out_fused.float()).abs()
        flat_idx = diff.argmax()
        coords = []
        for dim_size in reversed(diff.shape):
            coords.append(flat_idx % dim_size)
            flat_idx = flat_idx // dim_size
        coords.reverse()
        print(f"  Max diff at position: {[c.item() for c in coords]}")
        print(f"  Chunked value: {out_chunked.flatten()[diff.argmax()].item():.6f}")
        print(f"  Fused value:   {out_fused.flatten()[diff.argmax()].item():.6f}")
        return False

    return True


if __name__ == "__main__":
    test_fused_vs_chunked()
