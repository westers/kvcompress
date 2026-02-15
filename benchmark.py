"""
Benchmark: DMS-8x vs Vanilla Qwen3-8B vs kvpress Compression Methods

Tests at long context lengths (4K-32K tokens) where KV cache compression
actually matters. Measures throughput, VRAM usage, and answer quality.
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)
from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    SnapKVPress,
)

# Local DMS with Triton-optimized kernels
from dms_local.configuration_qwen3_dms import Qwen3Config as DMSConfig
from dms_local.modeling_qwen3_dms import Qwen3ForCausalLM as DMSForCausalLM


@dataclass
class BenchmarkResult:
    model: str
    method: str
    context_tokens: int
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    peak_vram_gb: float
    answer_preview: str


QUESTION = "What is the capital of Freedonia?"
NEEDLE_FACT = "The capital of Freedonia is Silverton."
EXPECTED_ANSWER = "Silverton"

FILLER_PARAGRAPHS = [
    "The development of renewable energy sources has been a major focus of environmental policy in recent decades. Solar panel efficiency has improved dramatically, with modern panels converting over 22% of sunlight into electricity. Wind turbines have also grown larger and more efficient, with offshore installations generating significant power for coastal regions. The transition away from fossil fuels requires massive infrastructure investment and careful planning to ensure grid stability.",
    "Machine learning algorithms have transformed numerous industries. In healthcare, deep learning models can detect certain cancers from medical images with accuracy comparable to experienced radiologists. Natural language processing has enabled more natural human-computer interaction through chatbots and virtual assistants. Reinforcement learning has achieved superhuman performance in complex games like Go and StarCraft, demonstrating the potential of AI to master complex decision-making tasks.",
    "The history of space exploration spans several decades of remarkable achievement. The Apollo program successfully landed humans on the Moon six times between 1969 and 1972. The Space Shuttle program operated from 1981 to 2011, completing 135 missions. The International Space Station has been continuously occupied since November 2000, serving as a laboratory for scientific research in microgravity conditions.",
    "Oceanic ecosystems are among the most complex and vital environments on Earth. Coral reefs, often called the rainforests of the sea, support approximately 25% of all marine species despite covering less than 1% of the ocean floor. Deep sea hydrothermal vents host unique ecosystems that derive energy from chemical reactions rather than sunlight. The interconnected nature of ocean currents means that changes in one region can have far-reaching effects across the globe.",
    "Urban planning has evolved significantly to address modern challenges. Smart city initiatives leverage IoT sensors and data analytics to optimize traffic flow, energy usage, and public services. Green building standards have become increasingly stringent, requiring better insulation, efficient HVAC systems, and renewable energy integration. Mixed-use development aims to reduce commuting distances and create more walkable, livable communities.",
    "The field of materials science continues to produce innovations. Graphene, a single layer of carbon atoms arranged in a hexagonal lattice, possesses extraordinary strength and electrical conductivity. Shape-memory alloys can return to their original form after deformation when heated. Biodegradable plastics derived from plant materials offer a potential solution to plastic pollution in oceans and landfills.",
    "Modern agriculture faces the dual challenge of feeding a growing population while minimizing environmental impact. Precision agriculture uses GPS, sensors, and drones to optimize irrigation, fertilization, and pest management at the individual plant level. Vertical farming enables year-round crop production in controlled environments with minimal water usage. Gene editing technologies like CRISPR offer the potential to develop crops with improved yields, nutritional content, and disease resistance.",
    "The evolution of telecommunications has fundamentally changed human society. The transition from analog to digital networks enabled the internet revolution of the late 20th century. Mobile phones went from luxury items to essential tools carried by billions of people worldwide. The deployment of 5G networks promises to enable new applications requiring ultra-low latency, such as remote surgery and autonomous vehicles.",
]


def build_context(target_tokens, tokenizer, needle_position=0.5):
    """Build a context of target_tokens length with a fact needle embedded."""
    avg_tokens = sum(len(tokenizer.encode(p)) for p in FILLER_PARAGRAPHS) / len(FILLER_PARAGRAPHS)
    n_paragraphs = int(target_tokens / avg_tokens) + 5

    paragraphs = [FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)] for i in range(n_paragraphs)]

    # Insert needle fact at specified position
    insert_idx = max(1, int(needle_position * len(paragraphs)))
    paragraphs.insert(insert_idx, NEEDLE_FACT)

    context = "\n\n".join(paragraphs)
    tokens = tokenizer.encode(context)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    actual_tokens = len(tokenizer.encode(context))
    return context, actual_tokens


def run_direct_generation(model, tokenizer, context, model_name, method_name, ctx_label):
    """Generate directly via model.generate() and collect metrics."""
    messages = [
        {"role": "system", "content": "Answer the question based only on the provided context. Be brief and precise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    output_tokens = output_ids.shape[1] - input_len
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    decoded = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return BenchmarkResult(
        model=model_name,
        method=method_name,
        context_tokens=ctx_label,
        input_tokens=input_len,
        output_tokens=output_tokens,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
        peak_vram_gb=round(peak_mem_gb, 2),
        answer_preview=decoded[:200],
    )


def run_kvpress_generation(pipe, press, tokenizer, context, model_name, method_name, ctx_label):
    """Generate via kvpress pipeline and collect metrics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    result = pipe(
        context,
        question=f"\nAnswer based only on the context above. {QUESTION}",
        press=press,
        max_new_tokens=128,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    answer = result["answer"]
    output_tokens = len(tokenizer.encode(answer))
    input_tokens = len(tokenizer.encode(context))

    return BenchmarkResult(
        model=model_name,
        method=method_name,
        context_tokens=ctx_label,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
        peak_vram_gb=round(peak_mem_gb, 2),
        answer_preview=answer[:200],
    )


def print_results_table(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 130)
    print("BENCHMARK RESULTS")
    print("=" * 130)

    header = (
        f"{'Ctx Len':>8} {'Model':<20} {'Method':<30} "
        f"{'In Tok':>7} {'Out Tok':>8} {'Time(s)':>8} "
        f"{'Tok/s':>7} {'VRAM(GB)':>9} {'Correct':>8}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        correct = "YES" if EXPECTED_ANSWER.lower() in r.answer_preview.lower() else "no"
        print(
            f"{r.context_tokens:>8} {r.model:<20} {r.method:<30} "
            f"{r.input_tokens:>7} {r.output_tokens:>8} {r.elapsed_sec:>8.2f} "
            f"{r.tokens_per_sec:>7.1f} {r.peak_vram_gb:>9.2f} {correct:>8}"
        )

    # Aggregated summary by context length and method
    print("\n" + "=" * 100)
    print("AGGREGATED BY CONTEXT LENGTH & METHOD")
    print("=" * 100)

    header = f"{'Ctx Len':>8} {'Method':<40} {'Tok/s':>8} {'VRAM(GB)':>10} {'Time(s)':>10} {'Correct':>8}"
    print(header)
    print("-" * 100)

    # Group by context length
    ctx_lengths = sorted(set(r.context_tokens for r in results))
    for ctx_len in ctx_lengths:
        ctx_results = [r for r in results if r.context_tokens == ctx_len]
        for r in ctx_results:
            correct = "YES" if EXPECTED_ANSWER.lower() in r.answer_preview.lower() else "no"
            label = f"{r.model} / {r.method}"
            print(
                f"{ctx_len:>8} {label:<40} {r.tokens_per_sec:>8.1f} "
                f"{r.peak_vram_gb:>10.2f} {r.elapsed_sec:>10.2f} {correct:>8}"
            )
        print()


def save_results(results, path="benchmark_results.json"):
    """Save results to JSON for later analysis."""
    data = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")


def main():
    print("=" * 60)
    print("KV Cache Compression Benchmark (Long Context)")
    print("=" * 60)

    device = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device} ({total_mem:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    results = []

    context_lengths = [4096, 8192, 16384, 32768]

    # Pre-build contexts
    contexts = {}
    for ctx_len in context_lengths:
        context, actual = build_context(ctx_len, tokenizer)
        contexts[ctx_len] = context
        print(f"Built {ctx_len}-token context: {actual} actual tokens")

    # --- DMS-8x ---
    print("\n>>> Loading DMS-8x <<<")
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    for ctx_len in context_lengths:
        print(f"  DMS-8x @ {ctx_len} tokens...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B-DMS-8x", "DMS-8x (built-in)", ctx_len,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {ctx_len} tokens!")
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()

    # --- DMS-8x Local (Triton-optimized) ---
    print("\n>>> Loading DMS-8x Local (Triton kernels) <<<")
    config = DMSConfig.from_pretrained("nvidia/Qwen3-8B-DMS-8x")
    model = DMSForCausalLM.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for ctx_len in context_lengths:
        print(f"  DMS-8x-local @ {ctx_len} tokens...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "DMS-8x-local", "DMS-8x (Triton)", ctx_len,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {ctx_len} tokens!")
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()

    # --- Vanilla ---
    print("\n>>> Loading Vanilla Qwen3-8B <<<")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for ctx_len in context_lengths:
        print(f"  Vanilla @ {ctx_len} tokens...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B", "No compression", ctx_len,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {ctx_len} tokens!")
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()

    # --- kvpress methods ---
    print("\n>>> Loading kvpress pipeline <<<")
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model="Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    kvpress_methods = [
        ("KnormPress(0.5)", KnormPress(compression_ratio=0.5)),
        ("SnapKVPress(0.5)", SnapKVPress(compression_ratio=0.5)),
        ("ExpAttention(0.5)", ExpectedAttentionPress(compression_ratio=0.5)),
    ]

    for method_name, press in kvpress_methods:
        for ctx_len in context_lengths:
            print(f"  {method_name} @ {ctx_len} tokens...")
            try:
                r = run_kvpress_generation(
                    pipe, press, tokenizer, contexts[ctx_len],
                    "Qwen3-8B", method_name, ctx_len,
                )
                results.append(r)
            except (torch.cuda.OutOfMemoryError, Exception) as e:
                if "OutOfMemory" in type(e).__name__:
                    print(f"    OOM at {ctx_len} tokens!")
                    torch.cuda.empty_cache()
                    break
                else:
                    print(f"    Error: {e}")
                    torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
