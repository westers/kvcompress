# KV Cache Compression Benchmark & DMS Triton Optimizations

Benchmarking KV cache compression methods for LLM inference, with Triton kernel
optimizations for NVIDIA's [Dynamic Memory Sparsification (DMS)](https://arxiv.org/abs/2506.05345).

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

## DMS Prefill Optimization Results

The DMS inference code shipped with [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x)
uses Python batch-loops and chunked SDPA calls during prefill. Two optimized
implementations replace these with GPU-native attention:

- **FlexAttn** — [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms) uses [`flex_attention`](https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html) with 4096-token chunked prefill
- **Triton** — Custom fused flash attention kernel with DMS eviction masking built in (this repo)

**Hardware:** NVIDIA RTX 4090 (24 GB), Qwen3-8B, batch size 1

### DMS-8x Prefill Speedup vs Original Hub Code

| Context | Original (Hub) | FlexAttn | Triton | FlexAttn Speedup | Triton Speedup |
|---------|---------------|----------|--------|-----------------|----------------|
| 4K | 3.457s | 1.260s | 3.083s | **2.74x** | 1.12x |
| 8K | 7.922s | 1.348s | 5.795s | **5.88x** | 1.37x |
| 16K | 22.211s | 2.503s | 12.699s | **8.88x** | 1.75x |
| 32K | OOM | 5.139s | 25.685s | **was OOM** | **was OOM** |

| Context | Hub VRAM | FlexAttn VRAM | Triton VRAM | FlexAttn Saved | Triton Saved |
|---------|----------|---------------|-------------|----------------|--------------|
| 4K | 16.89 GB | 16.84 GB | 15.99 GB | 0.05 GB | 0.90 GB |
| 8K | 17.93 GB | 16.97 GB | 16.47 GB | 0.96 GB | 1.46 GB |
| 16K | 20.07 GB | 17.21 GB | 17.48 GB | 2.86 GB | **2.59 GB** |
| 32K | OOM | 17.79 GB | 19.43 GB | — | — |

Key takeaways:
- **FlexAttn is the fastest prefill** — up to 8.88x faster than Hub at 16K, and scales better to 32K (5.1s vs 25.7s)
- **Triton uses less VRAM** at shorter contexts (15.99 GB vs 16.84 GB at 4K) due to eliminated per-chunk mask allocations
- **Both fit 32K** where the original Hub code OOMs
- **Decode speed is comparable** across all DMS variants (~20-42 tok/s depending on mode and context), so the prefill path is the key differentiator

### Benchmark Results

All methods tested with needle-in-a-haystack retrieval on an RTX 4090 (24 GB). Three iterations averaged per data point.

#### DMS-8x FlexAttn ([NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms))

| Context | no_think TTFT | no_think Tok/s | no_think Decode | think TTFT | think Tok/s | think Decode | VRAM (GB) | Correct |
|---------|---------------|----------------|-----------------|------------|-------------|--------------|-----------|---------|
| 4K | 1.260s | 6.2 | 27.4 | 0.654s | 28.6 | 32.3 | 16.84 | YES/YES |
| 8K | 1.348s | 5.4 | 20.3 | 1.255s | 27.3 | 33.8 | 16.97 | YES/YES |
| 16K | 2.503s | 3.5 | 27.3 | 2.517s | 24.9 | 33.1 | 17.21 | YES/YES |
| 32K | 5.139s | 1.8 | 26.1 | 5.118s | 17.8 | 32.4 | 17.79 | YES/YES |

> FlexAttn uses [torch.nn.attention.flex_attention](https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html) for the DMS attention prefill with 4096-token chunked prefill. This is the fastest DMS implementation, with prefill speeds approaching vanilla while maintaining 8x KV cache compression.

#### DMS-8x Hub (nvidia/Qwen3-8B-DMS-8x)

| Context | no_think TTFT | no_think Tok/s | no_think Decode | think TTFT | think Tok/s | think Decode | VRAM (GB) | Correct |
|---------|---------------|----------------|-----------------|------------|-------------|--------------|-----------|---------|
| 4K | 3.457s | 2.7 | 35.6 | 3.443s | 20.7 | 36.6 | 16.89 | YES/YES |
| 8K | 7.922s | 1.2 | 42.3 | 7.937s | 14.2 | 38.1 | 17.93 | YES/YES |
| 16K | 22.211s | 0.4 | 41.6 | 21.682s | 6.8 | 38.7 | 20.07 | YES/YES |
| 32K | OOM | — | — | OOM | — | — | — | — |

#### DMS-8x Triton (local optimized kernels)

| Context | no_think TTFT | no_think Tok/s | no_think Decode | think TTFT | think Tok/s | think Decode | VRAM (GB) | Correct |
|---------|---------------|----------------|-----------------|------------|-------------|--------------|-----------|---------|
| 4K | 3.083s | 3.0 | 33.7 | 2.951s | 24.0 | 36.2 | 15.99 | YES/YES |
| 8K | 5.795s | 1.7 | 38.7 | 5.813s | 16.7 | 36.3 | 16.47 | YES/YES |
| 16K | 12.699s | 0.8 | 37.6 | 12.464s | 12.9 | 36.5 | 17.48 | YES/YES |
| 32K | 25.685s | 0.4 | 39.1 | 25.191s | 6.6 | 36.8 | 19.43 | YES/YES |

#### Vanilla Qwen3-8B (no compression)

| Context | no_think TTFT | no_think Tok/s | no_think Decode | think TTFT | think Tok/s | think Decode | VRAM (GB) | Correct |
|---------|---------------|----------------|-----------------|------------|-------------|--------------|-----------|---------|
| 4K | 0.488s | 14.3 | 47.7 | 0.492s | 38.5 | 43.4 | 16.25 | YES/YES |
| 8K | 1.047s | 7.9 | 44.9 | 1.051s | 33.2 | 41.0 | 17.22 | YES/YES |
| 16K | 2.339s | 3.9 | 39.2 | 2.350s | 25.2 | 35.5 | 19.16 | YES/YES |
| 32K | OOM | — | — | OOM | — | — | — | — |

#### kvpress Methods (Qwen3-8B, compression_ratio=0.5)

| Context | KnormPress | SnapKV | ExpAttn | VRAM (GB) |
|---------|------------|--------|---------|-----------|
| 4K | 12.3 (no) | 12.1 | 11.1 | 15.96 |
| 8K | 6.8 (no) | 6.6 | 6.1 | 16.64 |
| 16K | 4.1 | 3.3 | 3.0 | 18.02 |
| 32K | 1.5 | 1.4 | 1.4 | 20.78 |

> Cell values are tok/s (no_think mode only). **(no)** = incorrect answer. TTFT and decode breakdown not available for kvpress pipeline.

## Methodology

**Hardware:** NVIDIA RTX 4090 (24 GB VRAM), AMD Ryzen, 64 GB RAM

**Task:** Needle-in-a-haystack retrieval. A fact ("The capital of Freedonia is Silverton") is embedded at the 50% position within filler text padded to the target context length. The model is asked to retrieve this fact.

**Metrics:**

| Metric | Definition | Notes |
|--------|-----------|-------|
| TTFT | Time to first token (prefill latency) | Not available for kvpress |
| Tok/s | Output tokens / total elapsed time | Includes prefill + decode |
| Decode tok/s | Output tokens / decode time only | Pure generation speed; not available for kvpress |
| VRAM | Peak GPU memory (`torch.cuda.max_memory_allocated`) | Reset before each run |
| Correct | Answer contains "Silverton" | YES/YES = both modes correct |

**Parameters:** `temperature=0.1`, `do_sample=True`, `max_new_tokens=128` (no_think) or `512` (think). Three iterations averaged per configuration. bf16 precision, `device_map="auto"`.

**Models tested:**

| Model | Weights | Attention | Description |
|-------|---------|-----------|-------------|
| DMS-8x FlexAttn | [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) | [`flex_attention`](https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html) | [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms) with 4096-token chunked prefill |
| DMS-8x Hub | [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) | Chunked SDPA | Original `trust_remote_code` implementation |
| DMS-8x Triton | [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) | Fused Triton kernel | This repo — single-pass flash attention with DMS masking |
| Vanilla | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Flash Attention 2 | No KV cache compression |
| kvpress | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Flash Attention 2 | Post-hoc compression via [kvpress](https://github.com/NVIDIA/kvpress) pipeline |

## License

- `dms_local/triton_kernels.py` and `benchmark.py`: Apache-2.0
- `dms_local/dms_attention.py`, `dms_local/dms_cache.py`: Apache-2.0 (modified
  from [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x),
  Copyright (c) 2025 NVIDIA CORPORATION)
- `dms_local/modeling_qwen3_dms.py`: Apache-2.0 (modified from
  nvidia/Qwen3-8B-DMS-8x, Copyright (c) 2025 NVIDIA CORPORATION and The Qwen
  team, Alibaba Group — added profiler hooks and cache offloading support)
- `dms_local/configuration_qwen3_dms.py`: Apache-2.0 (unmodified from
  nvidia/Qwen3-8B-DMS-8x, Copyright (c) 2025 NVIDIA CORPORATION and The Qwen
  team, Alibaba Group)

## References

- [Inference-Time Hyper-Scaling with KV Cache Compression](https://arxiv.org/abs/2506.05345)
  (Lancucki, Staniszewski, Nawrot, Ponti)
- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) - KV cache compression library
- [nvidia/Qwen3-8B-DMS-8x](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x) - DMS model
- [NVIDIA Model Optimizer — DMS](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms) - FlexAttention-based DMS inference
- [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/) - PyTorch blog
- [torch.nn.attention.flex_attention](https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html) - PyTorch API docs
