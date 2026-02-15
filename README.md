# KV Cache Compression Benchmark & DMS Triton Optimizations

Benchmarking KV cache compression methods for LLM inference, with Triton kernel
optimizations for NVIDIA's [Dynamic Memory Sparsification (DMS)](https://arxiv.org/abs/2506.05345).

## DMS Prefill Optimization Results

The DMS inference code shipped with [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x)
uses Python batch-loops and chunked SDPA calls during prefill. We replaced these
with Triton GPU kernels, including a fused flash attention kernel with DMS
eviction masking built in.

**Hardware:** NVIDIA RTX 4090 (24 GB), Qwen3-8B, batch size 1

### DMS-8x Prefill Speedup (Triton vs Original)

| Context | Original (Hub) | Triton Optimized | Speedup | VRAM Saved |
|---------|---------------|-----------------|---------|------------|
| 4K | 9.09s | 7.77s | 1.17x | 0.56 GB |
| 8K | 11.33s | 8.97s | 1.26x | 1.14 GB |
| 16K | 25.53s | 15.43s | **1.65x** | **2.25 GB** |
| 32K | OOM | 27.97s | **was OOM** | N/A |

Key improvements:
- **1.65x faster** at 16K context (the chunked loop was 32 iterations; now single kernel launch)
- **32K now fits in 24 GB** (19.42 GB vs OOM) due to eliminated per-chunk mask allocations
- **Up to 2.25 GB VRAM savings** from not materializing intermediate attention masks

### Full Benchmark Comparison

All methods tested at 4K-32K context with needle-in-a-haystack retrieval
(128 output tokens, bf16):

| Context | Method | Time (s) | Tok/s | VRAM (GB) |
|---------|--------|----------|-------|-----------|
| 4K | Vanilla Qwen3-8B | 3.61 | 35.5 | 16.25 |
| 4K | DMS-8x (Hub) | 9.09 | 14.1 | 16.55 |
| 4K | DMS-8x (Triton) | 7.77 | 16.5 | 15.99 |
| 4K | SnapKVPress(0.5) | 0.85 | 10.6 | 15.96 |
| 8K | Vanilla Qwen3-8B | 4.36 | 29.3 | 17.22 |
| 8K | DMS-8x (Hub) | 11.33 | 11.3 | 17.60 |
| 8K | DMS-8x (Triton) | 8.97 | 14.3 | 16.46 |
| 8K | SnapKVPress(0.5) | 1.38 | 6.5 | 16.64 |
| 16K | Vanilla Qwen3-8B | 5.96 | 21.5 | 19.16 |
| 16K | DMS-8x (Hub) | 25.53 | 5.0 | 19.73 |
| 16K | DMS-8x (Triton) | 15.43 | 8.3 | 17.48 |
| 16K | SnapKVPress(0.5) | 2.79 | 3.2 | 18.02 |
| 32K | DMS-8x (Triton) | 27.97 | 4.6 | 19.42 |
| 32K | SnapKVPress(0.5) | 6.22 | 1.4 | 20.78 |

## What Was Optimized

Five Triton kernels replace Python batch-loops in the DMS prefill path:

1. **`left_pad_2d`** - Parallel left-padding of `[N, S, D]` tensors (replaces
   `left_pad_one()` serial loop in `dms_cache.py`)

2. **`scatter_by_index`** - Parallel scatter of indexed elements (replaces
   `restore_order()` loop in `dms_attention.py`)

3. **`bool_gather_left_pad`** - Fused boolean gather + left-pad for `[B, H, S, D]`
   tensors (replaces `convert_to_left_padding()` inner loop)

4. **`compact_by_bool`** - Boolean-masked compaction (replaces
   `get_contiguous_cache()` compaction loop)

5. **`dms_fused_prefill`** - **Single-pass flash attention with DMS masking.**
   Replaces the entire chunked `dms_prefill_attention()` loop (32 iterations at
   16K context) with one Triton kernel launch. Pre-computes eviction state
   upfront (the sequential chunk dependency is actually redundant since eviction
   is monotonic and determined independently per token by `decisions[k+1]`).

## Setup

```bash
uv sync
```

## Usage

Run the benchmark:
```bash
uv run python benchmark.py
```

Run correctness tests:
```bash
uv run python test_fused_prefill.py
```

## License

- `dms_local/triton_kernels.py` and `benchmark.py`: Apache-2.0
- `dms_local/dms_attention.py`, `dms_local/dms_cache.py`: Apache-2.0 (modified
  from [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x),
  Copyright (c) 2025 NVIDIA CORPORATION)
- `dms_local/modeling_qwen3_dms.py`, `dms_local/configuration_qwen3_dms.py`:
  Apache-2.0 (unmodified from nvidia/Qwen3-8B-DMS-8x, Copyright (c) 2025 NVIDIA
  CORPORATION and The Qwen team, Alibaba Group)

## References

- [Inference-Time Hyper-Scaling with KV Cache Compression](https://arxiv.org/abs/2506.05345)
  (Lancucki, Staniszewski, Nawrot, Ponti)
- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) - KV cache compression library
- [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) - DMS model
