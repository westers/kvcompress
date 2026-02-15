# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kvcompress is a benchmarking suite for KV cache compression techniques in LLM inference. It compares three approaches:
- **DMS-8x**: Built-in compression via `Qwen3-8B-DMS-8x` model (requires `trust_remote_code=True`)
- **Vanilla**: Standard `Qwen/Qwen3-8B` without compression
- **kvpress**: Library-based compression applied to vanilla models via `kv-press-text-generation` pipeline

## Commands

```bash
# Install dependencies
uv sync

# Run benchmarks and tests
uv run python benchmark.py           # Full benchmark: DMS-8x vs vanilla vs kvpress
uv run python test_kvpress_api.py    # Explore kvpress compression methods and ratios
uv run python test_dms_inference.py  # DMS-8x vs vanilla side-by-side comparison
uv run python test_long_context.py   # Needle-in-haystack retrieval at 8K-32K contexts
```

## Architecture

All test files follow the same pattern:
1. Load models with `AutoModelForCausalLM` using `bfloat16` + `device_map="auto"`
2. Reset GPU memory stats before each test
3. Run generation with `torch.cuda.synchronize()` for accurate timing
4. Collect metrics: tokens/sec, peak VRAM, elapsed time, TTFT

**Key files:**
- `benchmark.py` — Main benchmark with `BenchmarkResult` dataclass, outputs formatted tables + JSON
- `test_kvpress_api.py` — Tests compression methods (KnormPress, ExpectedAttentionPress, SnapKVPress, StreamingLLMPress, DMSPress, DecodingPress) at varying ratios
- `test_long_context.py` — Embeds needle facts at 5%/50%/95% positions in padded contexts, tests retrieval accuracy

**kvpress integration pattern:**
```python
pipe = hf_pipeline("kv-press-text-generation", model=..., torch_dtype=torch.bfloat16, device_map="auto")
result = pipe(context=context, question=question, press=KnormPress(compression_ratio=0.5))
```

## Dependencies

- Python 3.12 (pinned in `.python-version`)
- PyTorch from CUDA 12.8 index (`https://download.pytorch.org/whl/cu128`)
- `transformers==4.57.3` (pinned), `kvpress>=0.4.3`, `flash-attn>=2.8.3`, `accelerate>=1.12.0`
- Managed with `uv` — do not use pip directly

## Conventions

- Generation params: `max_new_tokens=512, temperature=0.6, top_p=0.95, do_sample=True`
- All models use shared tokenizer: `Qwen/Qwen3-8B`
- GPU memory measured via `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`
