"""
Long-Context Retrieval Test

Tests the key DMS claim: DMS enables more generation within fixed compute
by compressing the KV cache while maintaining retrieval accuracy.

1. Constructs long contexts (8K-32K tokens) with embedded facts at various positions
2. Asks questions requiring retrieval from beginning, middle, and end
3. Compares retrieval accuracy: DMS-8x vs vanilla Qwen3-8B vs kvpress compression
4. Measures memory savings at different context lengths
"""

import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from kvpress import KnormPress, ExpectedAttentionPress


# Seed for reproducibility
random.seed(42)

# Facts to embed at specific positions in the context
NEEDLE_FACTS = [
    {"fact": "The capital of Freedonia is Silverton.", "question": "What is the capital of Freedonia?", "answer": "Silverton"},
    {"fact": "Project Falcon was launched on March 15, 2024.", "question": "When was Project Falcon launched?", "answer": "March 15, 2024"},
    {"fact": "The maximum speed of the XR-7 prototype is 340 km/h.", "question": "What is the maximum speed of the XR-7 prototype?", "answer": "340 km/h"},
    {"fact": "Dr. Elena Vasquez invented the quantum resonance detector.", "question": "Who invented the quantum resonance detector?", "answer": "Dr. Elena Vasquez"},
    {"fact": "The Treaty of Eastbrook was signed by 17 nations.", "question": "How many nations signed the Treaty of Eastbrook?", "answer": "17"},
]

# Filler text paragraphs (diverse topics to pad context length)
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


def build_context(target_tokens, needle_positions, tokenizer):
    """
    Build a context string of approximately target_tokens length with facts
    embedded at specified positions.

    needle_positions: list of floats in [0, 1] indicating relative position
                      for each NEEDLE_FACT
    """
    # Estimate tokens per paragraph
    avg_tokens_per_para = sum(len(tokenizer.encode(p)) for p in FILLER_PARAGRAPHS) / len(FILLER_PARAGRAPHS)
    n_paragraphs = int(target_tokens / avg_tokens_per_para) + 5  # overshoot slightly

    # Build paragraph list with filler
    paragraphs = []
    for i in range(n_paragraphs):
        paragraphs.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)])

    # Insert needles at specified positions
    needles_used = []
    for i, pos in enumerate(needle_positions):
        if i >= len(NEEDLE_FACTS):
            break
        insert_idx = max(1, int(pos * len(paragraphs)))
        insert_idx = min(insert_idx, len(paragraphs) - 1)
        needle = NEEDLE_FACTS[i]
        paragraphs.insert(insert_idx, needle["fact"])
        needles_used.append(needle)

    # Join and truncate to target token count
    context = "\n\n".join(paragraphs)
    tokens = tokenizer.encode(context)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    actual_tokens = len(tokenizer.encode(context))
    return context, needles_used, actual_tokens


def check_answer(response, expected):
    """Check if expected answer appears in model response."""
    return expected.lower() in response.lower()


def test_retrieval_direct(model, tokenizer, context, needles, model_name):
    """Test retrieval using direct model.generate()."""
    results = []

    for needle in needles:
        messages = [
            {"role": "system", "content": "Answer the question based only on the provided context. Be brief and precise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {needle['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

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
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        correct = check_answer(response, needle["answer"])

        results.append({
            "model": model_name,
            "question": needle["question"],
            "expected": needle["answer"],
            "response": response[:200],
            "correct": correct,
            "elapsed_sec": round(elapsed, 2),
            "peak_vram_gb": round(peak_mem_gb, 2),
            "input_tokens": input_len,
        })

    return results


def test_retrieval_kvpress(pipe, press, tokenizer, context, needles, method_name):
    """Test retrieval using kvpress pipeline."""
    results = []

    for needle in needles:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        result = pipe(
            context,
            question=f"\nAnswer based only on the context above. {needle['question']}",
            press=press,
            max_new_tokens=128,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        response = result["answer"]
        correct = check_answer(response, needle["answer"])
        input_tokens = len(tokenizer.encode(context))

        results.append({
            "model": "Qwen3-8B + kvpress",
            "method": method_name,
            "question": needle["question"],
            "expected": needle["answer"],
            "response": response[:200],
            "correct": correct,
            "elapsed_sec": round(elapsed, 2),
            "peak_vram_gb": round(peak_mem_gb, 2),
            "input_tokens": input_tokens,
        })

    return results


def run_context_length_test(context_tokens, tokenizer_id="Qwen/Qwen3-8B"):
    """Run retrieval test at a specific context length."""
    print(f"\n{'='*80}")
    print(f"CONTEXT LENGTH: ~{context_tokens} tokens")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # Place needles at beginning (0.05), middle (0.5), and end (0.95)
    needle_positions = [0.05, 0.5, 0.95]
    context, needles, actual_tokens = build_context(context_tokens, needle_positions, tokenizer)
    print(f"Built context: {actual_tokens} tokens, {len(needles)} needles embedded")
    for i, n in enumerate(needles):
        pos_label = ["beginning", "middle", "end"][i]
        print(f"  Needle ({pos_label}): {n['fact'][:60]}...")

    all_results = []

    # Test DMS-8x
    print(f"\n--- DMS-8x Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    dms_results = test_retrieval_direct(model, tokenizer, context, needles, "DMS-8x")
    all_results.extend(dms_results)
    for r in dms_results:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"  [{status}] {r['question']} -> {r['response'][:100]}")
    del model
    torch.cuda.empty_cache()

    # Test Vanilla Qwen3-8B
    print(f"\n--- Vanilla Qwen3-8B ---")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    vanilla_results = test_retrieval_direct(model, tokenizer, context, needles, "Vanilla")
    all_results.extend(vanilla_results)
    for r in vanilla_results:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"  [{status}] {r['question']} -> {r['response'][:100]}")
    del model
    torch.cuda.empty_cache()

    # Test kvpress compression on vanilla model
    print(f"\n--- kvpress (KnormPress, cr=0.5) ---")
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model="Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    press = KnormPress(compression_ratio=0.5)
    kvpress_results = test_retrieval_kvpress(pipe, press, tokenizer, context, needles, "KnormPress(0.5)")
    all_results.extend(kvpress_results)
    for r in kvpress_results:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"  [{status}] {r['question']} -> {r['response'][:100]}")

    # Also test higher compression
    print(f"\n--- kvpress (ExpectedAttentionPress, cr=0.75) ---")
    press75 = ExpectedAttentionPress(compression_ratio=0.75)
    kvpress_results_75 = test_retrieval_kvpress(pipe, press75, tokenizer, context, needles, "ExpAttention(0.75)")
    all_results.extend(kvpress_results_75)
    for r in kvpress_results_75:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"  [{status}] {r['question']} -> {r['response'][:100]}")

    del pipe
    torch.cuda.empty_cache()

    return all_results


def print_summary(all_results_by_length):
    """Print summary across all context lengths."""
    print("\n" + "=" * 100)
    print("LONG-CONTEXT RETRIEVAL SUMMARY")
    print("=" * 100)

    header = f"{'Context Len':>12} {'Model/Method':<35} {'Accuracy':>10} {'Avg VRAM':>10} {'Avg Time':>10}"
    print(header)
    print("-" * 100)

    for ctx_len, results in sorted(all_results_by_length.items()):
        # Group by model
        groups = {}
        for r in results:
            key = r.get("method", r["model"])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        for method, runs in groups.items():
            correct = sum(1 for r in runs if r["correct"])
            total = len(runs)
            accuracy = f"{correct}/{total}"
            avg_vram = sum(r["peak_vram_gb"] for r in runs) / len(runs)
            avg_time = sum(r["elapsed_sec"] for r in runs) / len(runs)
            print(f"{ctx_len:>12} {method:<35} {accuracy:>10} {avg_vram:>10.2f} {avg_time:>10.2f}")
        print()


def main():
    print("=" * 60)
    print("Long-Context Retrieval Test")
    print("=" * 60)

    device = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device} ({total_mem:.1f} GB)")

    all_results = {}

    # Test at different context lengths
    # Start small and increase (bail early if OOM)
    for ctx_len in [8192, 16384, 32768]:
        try:
            results = run_context_length_test(ctx_len)
            all_results[ctx_len] = results
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at context length {ctx_len}! Stopping here.")
            torch.cuda.empty_cache()
            break

    print_summary(all_results)


if __name__ == "__main__":
    main()
