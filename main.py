"""
DMS Inference CLI with optional attention profiling.

Usage:
    uv run python main.py                          # Basic DMS inference
    uv run python main.py --profile                # With attention profiling
    uv run python main.py --profile --context-len 8192 --max-tokens 128
    uv run python main.py --context-len 16384 --needle-positions 0.05 0.5 0.95
"""

import argparse
import random
import time

import torch
from transformers import AutoTokenizer

from dms_local.configuration_qwen3_dms import Qwen3Config
from dms_local.modeling_qwen3_dms import Qwen3ForCausalLM


NEEDLE_FACTS = [
    {"fact": "The capital of Freedonia is Silverton.", "question": "What is the capital of Freedonia?", "answer": "Silverton"},
    {"fact": "Project Falcon was launched on March 15, 2024.", "question": "When was Project Falcon launched?", "answer": "March 15, 2024"},
    {"fact": "The maximum speed of the XR-7 prototype is 340 km/h.", "question": "What is the maximum speed of the XR-7 prototype?", "answer": "340 km/h"},
]

FILLER_PARAGRAPHS = [
    "The development of renewable energy sources has been a major focus of environmental policy in recent decades. Solar panel efficiency has improved dramatically, with modern panels converting over 22% of sunlight into electricity. Wind turbines have also grown larger and more efficient, with offshore installations generating significant power for coastal regions.",
    "Machine learning algorithms have transformed numerous industries. In healthcare, deep learning models can detect certain cancers from medical images with accuracy comparable to experienced radiologists. Natural language processing has enabled more natural human-computer interaction through chatbots and virtual assistants.",
    "The history of space exploration spans several decades of remarkable achievement. The Apollo program successfully landed humans on the Moon six times between 1969 and 1972. The International Space Station has been continuously occupied since November 2000, serving as a laboratory for scientific research in microgravity.",
    "Oceanic ecosystems are among the most complex and vital environments on Earth. Coral reefs support approximately 25% of all marine species despite covering less than 1% of the ocean floor. Deep sea hydrothermal vents host unique ecosystems that derive energy from chemical reactions rather than sunlight.",
    "Urban planning has evolved significantly to address modern challenges. Smart city initiatives leverage IoT sensors and data analytics to optimize traffic flow, energy usage, and public services. Green building standards have become increasingly stringent, requiring better insulation and renewable energy integration.",
    "The field of materials science continues to produce innovations. Graphene possesses extraordinary strength and electrical conductivity. Shape-memory alloys can return to their original form after deformation when heated. Biodegradable plastics offer a potential solution to pollution.",
    "Modern agriculture faces the dual challenge of feeding a growing population while minimizing environmental impact. Precision agriculture uses GPS, sensors, and drones to optimize irrigation and pest management. Vertical farming enables year-round crop production in controlled environments.",
    "The evolution of telecommunications has fundamentally changed human society. Mobile phones went from luxury items to essential tools carried by billions. The deployment of 5G networks promises to enable new applications requiring ultra-low latency.",
]


def build_context(target_tokens, needle_positions, tokenizer):
    """Build a padded context with needle facts at specified positions."""
    avg_tokens = sum(len(tokenizer.encode(p)) for p in FILLER_PARAGRAPHS) / len(FILLER_PARAGRAPHS)
    n_paragraphs = int(target_tokens / avg_tokens) + 5

    paragraphs = [FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)] for i in range(n_paragraphs)]

    needles_used = []
    for i, pos in enumerate(needle_positions):
        if i >= len(NEEDLE_FACTS):
            break
        idx = max(1, min(int(pos * len(paragraphs)), len(paragraphs) - 1))
        paragraphs.insert(idx, NEEDLE_FACTS[i]["fact"])
        needles_used.append(NEEDLE_FACTS[i])

    context = "\n\n".join(paragraphs)
    tokens = tokenizer.encode(context)
    if len(tokens) > target_tokens:
        context = tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)

    actual = len(tokenizer.encode(context))
    return context, needles_used, actual


def run_inference(model, tokenizer, context, question, max_new_tokens, prefill_chunk_size=None):
    """Run a single inference and return response + metrics."""
    messages = [
        {"role": "system", "content": "Answer the question based only on the provided context. Be brief and precise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
    }
    if prefill_chunk_size is not None:
        generate_kwargs["prefill_chunk_size"] = prefill_chunk_size

    # TTFT hook: capture time when first decode logits are computed
    ttft_holder = {}
    class TTFTProcessor:
        def __call__(self, input_ids, scores):
            if "ttft" not in ttft_holder:
                torch.cuda.synchronize()
                ttft_holder["ttft"] = time.perf_counter() - t_start
            return scores
    generate_kwargs["logits_processor"] = [TTFTProcessor()]

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    input_len = inputs["input_ids"].shape[1]
    output_len = output_ids.shape[1] - input_len
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    ttft = ttft_holder.get("ttft")
    decode_time = elapsed - ttft if ttft else elapsed

    return {
        "response": response,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "elapsed_sec": round(elapsed, 2),
        "ttft_sec": round(ttft, 2) if ttft else None,
        "decode_tok_per_sec": round(output_len / decode_time, 1) if decode_time > 0.5 else None,
        "tokens_per_sec": round(output_len / elapsed, 1),
        "peak_vram_gb": round(peak_vram, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="DMS inference with optional attention profiling")
    parser.add_argument("--profile", action="store_true", help="Enable attention sparsity profiling")
    parser.add_argument("--context-len", type=int, default=4096, help="Target context length in tokens")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--needle-positions", type=float, nargs="+", default=[0.05, 0.5, 0.95],
                        help="Relative positions for needle facts (0.0-1.0)")
    parser.add_argument("--model", type=str, default="nvidia/Qwen3-8B-DMS-8x", help="Model name/path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for profile results")
    parser.add_argument("--offload", action="store_true", help="Enable CPU cache offloading for long contexts (128K+)")
    parser.add_argument("--chunk-prefill", type=int, default=None, metavar="SIZE",
                        help="Chunked prefill size (e.g. 16384). Limits activation memory without cache offloading. "
                             "Implied by --offload if not set.")
    args = parser.parse_args()

    random.seed(42)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    config = Qwen3Config.from_pretrained(args.model)

    # Increase max_position_embeddings for long contexts
    needs_extended_pos = args.offload or (args.context_len > config.max_position_embeddings)
    if needs_extended_pos:
        original_max_pos = config.max_position_embeddings
        config.max_position_embeddings = max(config.max_position_embeddings, args.context_len + 1024)  # +1024 for chat template
        print(f"Extended max_position_embeddings {original_max_pos} → {config.max_position_embeddings}")

    model = Qwen3ForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable cache offloading if requested
    if args.offload:
        model.model.offload_cache = True
        print(f"CPU offloading enabled for contexts up to {config.max_position_embeddings} tokens")

    # Set up profiler if requested
    profiler = None
    if args.profile:
        from profile_attention import AttentionProfiler, SparsityAnalyzer

        num_layers = model.config.num_hidden_layers
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads

        profiler = AttentionProfiler(
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )

        # Attach profiler to every attention layer
        for layer in model.model.layers:
            layer.self_attn.profiler = profiler

        print(f"Profiler enabled: {num_layers} layers, {num_q_heads} Q heads, {num_kv_heads} KV heads")

    # Build context with needles
    print(f"\nBuilding context (~{args.context_len} tokens)...")
    context, needles, actual_tokens = build_context(args.context_len, args.needle_positions, tokenizer)
    print(f"Context: {actual_tokens} tokens, {len(needles)} needles")

    # Run inference for each needle question
    for i, needle in enumerate(needles):
        pos_label = ["beginning", "middle", "end"][i] if i < 3 else f"pos_{args.needle_positions[i]}"
        print(f"\n--- Question {i+1} ({pos_label}) ---")
        print(f"Q: {needle['question']}")

        chunk_size = args.chunk_prefill or (16384 if args.offload else None)
        result = run_inference(model, tokenizer, context, needle["question"], args.max_tokens, prefill_chunk_size=chunk_size)

        correct = needle["answer"].lower() in result["response"].lower()
        status = "CORRECT" if correct else "WRONG"
        print(f"A: {result['response'][:200]}")
        print(f"[{status}] Expected: {needle['answer']}")
        ttft_str = f"TTFT {result['ttft_sec']}s | " if result.get('ttft_sec') else ""
        decode_str = f"decode {result['decode_tok_per_sec']} tok/s | " if result.get('decode_tok_per_sec') else ""
        print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out | "
              f"{ttft_str}{decode_str}"
              f"{result['tokens_per_sec']} tok/s overall | {result['elapsed_sec']}s | "
              f"{result['peak_vram_gb']} GB VRAM")

    # Run profiler analysis
    if profiler is not None:
        print(f"\n{'='*60}")
        print("ATTENTION SPARSITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Captured {len(profiler.snapshots)} snapshots across {profiler.decode_step} decode steps")

        analyzer = SparsityAnalyzer(profiler)
        analysis = analyzer.analyze()
        analyzer.print_report(analysis)

        output_path = args.output or f"attention_profile_{args.context_len}.json"
        analyzer.save_report(output_path, analysis)
        print(f"\nProfile saved to {output_path}")


if __name__ == "__main__":
    main()
