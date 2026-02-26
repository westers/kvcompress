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
torch._dynamo.config.suppress_errors = True

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

# Model Optimizer DMS with FlexAttention prefill
sys.path.insert(0, "vendor/Model-Optimizer/experimental/dms")
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS as ModelOptDMS
from models.qwen3.configuration_qwen3_dms import Qwen3ConfigDMS as ModelOptConfig


@dataclass
class BenchmarkResult:
    model: str
    method: str
    mode: str                              # "think" or "no_think"
    context_tokens: int
    input_tokens: int
    output_tokens: float
    elapsed_sec: float
    tokens_per_sec: float
    peak_vram_gb: float
    answer_preview: str
    ttft_sec: float | None = None
    prefill_tok_per_sec: float | None = None
    decode_time_sec: float | None = None
    decode_tok_per_sec: float | None = None
    num_iterations: int = 1
    warmup_sec: float | None = None


def _is_oom(e):
    """Check if an exception is a CUDA out-of-memory error (any variant)."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


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


def run_warmup(model, tokenizer):
    """Short generation to trigger torch.compile caching."""
    context, _ = build_context(512, tokenizer)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What is this about?"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, temperature=0.1, do_sample=True)
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - t_start
    torch.cuda.empty_cache()
    return warmup_sec


def run_kvpress_warmup(pipe, tokenizer):
    """Short pipeline call to warm up kvpress."""
    context, _ = build_context(512, tokenizer)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    pipe(context, question="What is this about?", press=KnormPress(compression_ratio=0.5), max_new_tokens=32)
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - t_start
    torch.cuda.empty_cache()
    return warmup_sec


def run_direct_generation(
    model, tokenizer, context, model_name, method_name, ctx_label,
    enable_thinking=False, num_iterations=3, warmup_sec=None
):
    """Generate directly via model.generate() and collect metrics."""
    messages = [
        {"role": "system", "content": "Answer the question based only on the provided context. Be brief and precise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    max_new_tokens = 512 if enable_thinking else 128
    mode = "think" if enable_thinking else "no_think"

    elapsed_list = []
    ttft_list = []
    output_tokens_list = []
    peak_vram_list = []
    best_answer = ""

    for i in range(num_iterations):
        ttft_holder = {}

        class TTFTProcessor:
            def __call__(self, input_ids, scores):
                if "ttft" not in ttft_holder:
                    torch.cuda.synchronize()
                    ttft_holder["ttft"] = time.perf_counter() - t_start
                return scores

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                logits_processor=[TTFTProcessor()],
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        output_tokens = output_ids.shape[1] - input_len
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        decoded = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        elapsed_list.append(elapsed)
        if "ttft" in ttft_holder:
            ttft_list.append(ttft_holder["ttft"])
        output_tokens_list.append(output_tokens)
        peak_vram_list.append(peak_mem_gb)

        # Extract answer for think mode
        if enable_thinking and "</think>" in decoded:
            answer_part = decoded.split("</think>")[-1].strip()
        else:
            answer_part = decoded

        if EXPECTED_ANSWER.lower() in answer_part.lower() or not best_answer:
            best_answer = answer_part

    # Compute averages
    mean_elapsed = sum(elapsed_list) / len(elapsed_list)
    mean_output = sum(output_tokens_list) / len(output_tokens_list)
    mean_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else None
    max_vram = max(peak_vram_list)

    # Derived metrics
    prefill_tok_per_sec = input_len / mean_ttft if mean_ttft and mean_ttft > 0 else None
    decode_time_sec = mean_elapsed - mean_ttft if mean_ttft is not None else None
    decode_tok_per_sec = mean_output / decode_time_sec if decode_time_sec and decode_time_sec > 0 else None

    return BenchmarkResult(
        model=model_name,
        method=method_name,
        mode=mode,
        context_tokens=ctx_label,
        input_tokens=input_len,
        output_tokens=round(mean_output, 1),
        elapsed_sec=round(mean_elapsed, 2),
        tokens_per_sec=round(mean_output / mean_elapsed, 1) if mean_elapsed > 0 else 0,
        peak_vram_gb=round(max_vram, 2),
        answer_preview=best_answer[:200],
        ttft_sec=round(mean_ttft, 3) if mean_ttft is not None else None,
        prefill_tok_per_sec=round(prefill_tok_per_sec, 1) if prefill_tok_per_sec is not None else None,
        decode_time_sec=round(decode_time_sec, 3) if decode_time_sec is not None else None,
        decode_tok_per_sec=round(decode_tok_per_sec, 1) if decode_tok_per_sec is not None else None,
        num_iterations=num_iterations,
        warmup_sec=round(warmup_sec, 2) if warmup_sec is not None else None,
    )


def run_kvpress_generation(
    pipe, press, tokenizer, context, model_name, method_name, ctx_label,
    num_iterations=3, warmup_sec=None
):
    """Generate via kvpress pipeline and collect metrics."""
    elapsed_list = []
    output_tokens_list = []
    peak_vram_list = []
    best_answer = ""

    for i in range(num_iterations):
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

        elapsed_list.append(elapsed)
        output_tokens_list.append(output_tokens)
        peak_vram_list.append(peak_mem_gb)

        if EXPECTED_ANSWER.lower() in answer.lower() or not best_answer:
            best_answer = answer

    mean_elapsed = sum(elapsed_list) / len(elapsed_list)
    mean_output = sum(output_tokens_list) / len(output_tokens_list)
    max_vram = max(peak_vram_list)
    input_tokens = len(tokenizer.encode(context))

    return BenchmarkResult(
        model=model_name,
        method=method_name,
        mode="no_think",
        context_tokens=ctx_label,
        input_tokens=input_tokens,
        output_tokens=round(mean_output, 1),
        elapsed_sec=round(mean_elapsed, 2),
        tokens_per_sec=round(mean_output / mean_elapsed, 1) if mean_elapsed > 0 else 0,
        peak_vram_gb=round(max_vram, 2),
        answer_preview=best_answer[:200],
        ttft_sec=None,
        prefill_tok_per_sec=None,
        decode_time_sec=None,
        decode_tok_per_sec=None,
        num_iterations=num_iterations,
        warmup_sec=round(warmup_sec, 2) if warmup_sec is not None else None,
    )


def print_results_table(results, warmup_times=None):
    """Print formatted comparison table."""
    # Main detailed table
    col_width = 145
    print("\n" + "=" * col_width)
    print("BENCHMARK RESULTS")
    print("=" * col_width)

    header = (
        f"{'Ctx Len':>8} {'Mode':<10} {'Model':<20} {'Method':<26} "
        f"{'In Tok':>7} {'Out Tok':>8} {'Time(s)':>8} {'TTFT(s)':>8} "
        f"{'Prefill':>8} {'Decode':>8} {'Tok/s':>7} {'VRAM(GB)':>9} {'Correct':>8}"
    )
    print(header)
    print("-" * col_width)

    for r in results:
        correct = "YES" if EXPECTED_ANSWER.lower() in r.answer_preview.lower() else "no"
        ttft_str = f"{r.ttft_sec:.3f}" if r.ttft_sec is not None else "    N/A"
        prefill_str = f"{r.prefill_tok_per_sec:.1f}" if r.prefill_tok_per_sec is not None else "    N/A"
        decode_str = f"{r.decode_tok_per_sec:.1f}" if r.decode_tok_per_sec is not None else "    N/A"
        print(
            f"{r.context_tokens:>8} {r.mode:<10} {r.model:<20} {r.method:<26} "
            f"{r.input_tokens:>7} {r.output_tokens:>8.1f} {r.elapsed_sec:>8.2f} {ttft_str:>8} "
            f"{prefill_str:>8} {decode_str:>8} {r.tokens_per_sec:>7.1f} {r.peak_vram_gb:>9.2f} {correct:>8}"
        )

    # Warmup summary
    if warmup_times:
        print("\n" + "=" * 60)
        print("WARMUP TIMES")
        print("=" * 60)
        print(f"{'Model':<30} {'Warmup (s)':>12}")
        print("-" * 60)
        for model_name, wt in warmup_times.items():
            print(f"{model_name:<30} {wt:>12.2f}")

    # Per-model breakdown tables
    # Build unique model keys from (model, method) pairs, preserving insertion order
    model_keys = list(dict.fromkeys((r.model, r.method) for r in results))

    # Global union of all context lengths
    all_ctx_lengths = sorted(set(r.context_tokens for r in results))

    for model_name, method_name in model_keys:
        model_results = [r for r in results if r.model == model_name and r.method == method_name]
        modes_present = list(dict.fromkeys(r.mode for r in model_results))
        dual_mode = len(modes_present) == 2 and "no_think" in modes_present and "think" in modes_present

        # Index results by (mode, context_length) for fast lookup
        result_map = {}
        for r in model_results:
            result_map[(r.mode, r.context_tokens)] = r

        # Print model header
        print("\n" + "=" * 60)
        print(f"{method_name} [{model_name}]")
        print("=" * 60)

        if dual_mode:
            # Dual-mode header
            print(f"{'':>8}  {'---- no_think ----':^28}  {'------- think -------':^28}")
            print(
                f"{'Ctx Len':>8}  {'TTFT(s)':>7} {'Tok/s':>5} {'Decode':>6} {'VRAM':>6}"
                f"  {'TTFT(s)':>7} {'Tok/s':>5} {'Decode':>6} {'VRAM':>6}  {'Correct':>8}"
            )
            print("-" * 80)

            for ctx_len in all_ctx_lengths:
                r_nt = result_map.get(("no_think", ctx_len))
                r_th = result_map.get(("think", ctx_len))

                if r_nt is None and r_th is None:
                    # Both missing = OOM for this context length
                    print(f"{ctx_len:>8}    {'--- OOM ---':^24}    {'--- OOM ---':^24}")
                    continue

                # Format no_think columns
                if r_nt is not None:
                    nt_ttft = f"{r_nt.ttft_sec:.2f}" if r_nt.ttft_sec is not None else "  N/A"
                    nt_tps = f"{r_nt.tokens_per_sec:.1f}"
                    nt_dec = f"{r_nt.decode_tok_per_sec:.1f}" if r_nt.decode_tok_per_sec is not None else "  N/A"
                    nt_vram = f"{r_nt.peak_vram_gb:.2f}"
                    nt_correct = "YES" if EXPECTED_ANSWER.lower() in r_nt.answer_preview.lower() else "no"
                    nt_str = f"{nt_ttft:>7} {nt_tps:>5} {nt_dec:>6} {nt_vram:>6}"
                else:
                    nt_correct = "---"
                    nt_str = f"{'--- OOM ---':^28}"

                # Format think columns
                if r_th is not None:
                    th_ttft = f"{r_th.ttft_sec:.2f}" if r_th.ttft_sec is not None else "  N/A"
                    th_tps = f"{r_th.tokens_per_sec:.1f}"
                    th_dec = f"{r_th.decode_tok_per_sec:.1f}" if r_th.decode_tok_per_sec is not None else "  N/A"
                    th_vram = f"{r_th.peak_vram_gb:.2f}"
                    th_correct = "YES" if EXPECTED_ANSWER.lower() in r_th.answer_preview.lower() else "no"
                    th_str = f"{th_ttft:>7} {th_tps:>5} {th_dec:>6} {th_vram:>6}"
                else:
                    th_correct = "---"
                    th_str = f"{'--- OOM ---':^28}"

                correct_str = f"{nt_correct}/{th_correct}"
                print(f"{ctx_len:>8}  {nt_str}  {th_str}  {correct_str:>8}")
        else:
            # Single-mode header
            mode = modes_present[0]
            mode_label = f"---- {mode} ----"
            print(f"{'':>8}  {mode_label:^28}")
            print(
                f"{'Ctx Len':>8}  {'TTFT(s)':>7} {'Tok/s':>5} {'Decode':>6} {'VRAM':>6}  {'Correct':>8}"
            )
            print("-" * 52)

            for ctx_len in all_ctx_lengths:
                r = result_map.get((mode, ctx_len))

                if r is None:
                    print(f"{ctx_len:>8}    {'--- OOM ---':^24}")
                    continue

                ttft = f"{r.ttft_sec:.2f}" if r.ttft_sec is not None else "  N/A"
                tps = f"{r.tokens_per_sec:.1f}"
                dec = f"{r.decode_tok_per_sec:.1f}" if r.decode_tok_per_sec is not None else "  N/A"
                vram = f"{r.peak_vram_gb:.2f}"
                correct = "YES" if EXPECTED_ANSWER.lower() in r.answer_preview.lower() else "no"

                print(f"{ctx_len:>8}  {ttft:>7} {tps:>5} {dec:>6} {vram:>6}  {correct:>8}")


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
    warmup_times = {}

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
    print("  Running warmup...")
    wt = run_warmup(model, tokenizer)
    warmup_times["Qwen3-8B-DMS-8x"] = wt
    print(f"  Warmup: {wt:.2f}s")

    # No-think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x @ {ctx_len} tokens [no_think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B-DMS-8x", "DMS-8x (built-in)", ctx_len,
                enable_thinking=False, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    # Think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x @ {ctx_len} tokens [think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B-DMS-8x", "DMS-8x (built-in)", ctx_len,
                enable_thinking=True, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

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
    print("  Running warmup...")
    wt = run_warmup(model, tokenizer)
    warmup_times["DMS-8x-local"] = wt
    print(f"  Warmup: {wt:.2f}s")

    # No-think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x-local @ {ctx_len} tokens [no_think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "DMS-8x-local", "DMS-8x (Triton)", ctx_len,
                enable_thinking=False, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    # Think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x-local @ {ctx_len} tokens [think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "DMS-8x-local", "DMS-8x (Triton)", ctx_len,
                enable_thinking=True, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    del model
    torch.cuda.empty_cache()

    # --- DMS-8x Model Optimizer (FlexAttention) ---
    print("\n>>> Loading DMS-8x Model Optimizer (FlexAttention) <<<")
    config = ModelOptConfig.from_pretrained("nvidia/Qwen3-8B-DMS-8x")
    config.dms_chunked_prefill = 4096
    model = ModelOptDMS.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("  Running warmup...")
    wt = run_warmup(model, tokenizer)
    warmup_times["DMS-8x-ModelOpt"] = wt
    print(f"  Warmup: {wt:.2f}s")

    # No-think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x-ModelOpt @ {ctx_len} tokens [no_think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "DMS-8x-ModelOpt", "DMS-8x (FlexAttn)", ctx_len,
                enable_thinking=False, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    # Think pass
    for ctx_len in context_lengths:
        print(f"  DMS-8x-ModelOpt @ {ctx_len} tokens [think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "DMS-8x-ModelOpt", "DMS-8x (FlexAttn)", ctx_len,
                enable_thinking=True, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    del model
    torch.cuda.empty_cache()

    # --- Vanilla ---
    print("\n>>> Loading Vanilla Qwen3-8B <<<")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("  Running warmup...")
    wt = run_warmup(model, tokenizer)
    warmup_times["Qwen3-8B"] = wt
    print(f"  Warmup: {wt:.2f}s")

    # No-think pass
    for ctx_len in context_lengths:
        print(f"  Vanilla @ {ctx_len} tokens [no_think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B", "No compression", ctx_len,
                enable_thinking=False, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

    # Think pass
    for ctx_len in context_lengths:
        print(f"  Vanilla @ {ctx_len} tokens [think]...")
        try:
            r = run_direct_generation(
                model, tokenizer, contexts[ctx_len],
                "Qwen3-8B", "No compression", ctx_len,
                enable_thinking=True, num_iterations=3, warmup_sec=wt,
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _is_oom(e):
                print(f"    OOM at {ctx_len} tokens!")
                torch.cuda.empty_cache()
                break
            raise

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
    print("  Running warmup...")
    wt = run_kvpress_warmup(pipe, tokenizer)
    warmup_times["Qwen3-8B (kvpress)"] = wt
    print(f"  Warmup: {wt:.2f}s")

    kvpress_methods = [
        ("KnormPress(0.5)", KnormPress(compression_ratio=0.5)),
        ("SnapKVPress(0.5)", SnapKVPress(compression_ratio=0.5)),
        ("ExpAttention(0.5)", ExpectedAttentionPress(compression_ratio=0.5)),
    ]

    for method_name, press in kvpress_methods:
        for ctx_len in context_lengths:
            print(f"  {method_name} @ {ctx_len} tokens [no_think]...")
            try:
                r = run_kvpress_generation(
                    pipe, press, tokenizer, contexts[ctx_len],
                    "Qwen3-8B", method_name, ctx_len,
                    num_iterations=3, warmup_sec=wt,
                )
                results.append(r)
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                if _is_oom(e):
                    print(f"    OOM at {ctx_len} tokens!")
                    torch.cuda.empty_cache()
                    break
                else:
                    print(f"    Error: {e}")
                    torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    print_results_table(results, warmup_times=warmup_times)
    save_results(results)


if __name__ == "__main__":
    main()
