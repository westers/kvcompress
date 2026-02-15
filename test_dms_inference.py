"""
DMS Inference Test: nvidia/Qwen3-8B-DMS-8x vs vanilla Qwen3-8B

Tests basic inference, measures tokens/sec, peak GPU memory, and generation time.
Compares output quality between DMS-compressed and vanilla models.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPTS = [
    {
        "name": "math_reasoning",
        "messages": [{"role": "user", "content": "Solve step by step: What is the sum of all prime numbers less than 50?"}],
        "enable_thinking": True,
    },
    {
        "name": "code_generation",
        "messages": [{"role": "user", "content": "Write a Python function to find the longest palindromic substring in a string. Explain your approach."}],
        "enable_thinking": False,
    },
    {
        "name": "factual_qa",
        "messages": [{"role": "user", "content": "Explain the key differences between TCP and UDP protocols. When would you choose one over the other?"}],
        "enable_thinking": False,
    },
]

GENERATION_KWARGS = dict(
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)


def measure_generation(model, tokenizer, messages, enable_thinking=False):
    """Run generation and return metrics."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_KWARGS)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_tokens = output_ids.shape[1] - input_len
    elapsed = t_end - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0

    decoded = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return {
        "output": decoded,
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_gb": round(peak_mem_gb, 2),
    }


def load_model(model_id, trust_remote_code=False):
    """Load model and tokenizer."""
    # DMS model requires the base Qwen3-8B tokenizer
    tokenizer_id = "Qwen/Qwen3-8B"
    print(f"\nLoading tokenizer from {tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(f"Loading model from {model_id} (trust_remote_code={trust_remote_code})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def run_model_tests(model_id, trust_remote_code=False):
    """Run all test prompts on a model and return results."""
    model, tokenizer = load_model(model_id, trust_remote_code)
    results = {}

    for prompt_info in PROMPTS:
        name = prompt_info["name"]
        print(f"\n--- {name} ---")
        metrics = measure_generation(
            model, tokenizer,
            prompt_info["messages"],
            enable_thinking=prompt_info["enable_thinking"],
        )
        results[name] = metrics
        print(f"  Input: {metrics['input_tokens']} tokens")
        print(f"  Output: {metrics['output_tokens']} tokens in {metrics['elapsed_sec']}s")
        print(f"  Speed: {metrics['tokens_per_sec']} tok/s")
        print(f"  Peak VRAM: {metrics['peak_vram_gb']} GB")
        print(f"  Response preview: {metrics['output'][:200]}...")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()
    return results


def print_comparison(dms_results, vanilla_results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON: DMS-8x vs Vanilla Qwen3-8B")
    print("=" * 90)
    header = f"{'Prompt':<20} {'Metric':<15} {'DMS-8x':>12} {'Vanilla':>12} {'Delta':>10}"
    print(header)
    print("-" * 90)

    for name in PROMPTS:
        pname = name["name"]
        dms = dms_results[pname]
        van = vanilla_results[pname]

        for metric, label in [
            ("tokens_per_sec", "tok/s"),
            ("peak_vram_gb", "VRAM (GB)"),
            ("elapsed_sec", "time (s)"),
            ("output_tokens", "out tokens"),
        ]:
            d_val = dms[metric]
            v_val = van[metric]
            if isinstance(d_val, float):
                delta = f"{d_val - v_val:+.1f}"
                print(f"{pname:<20} {label:<15} {d_val:>12.1f} {v_val:>12.1f} {delta:>10}")
            else:
                delta = f"{d_val - v_val:+d}"
                print(f"{pname:<20} {label:<15} {d_val:>12d} {v_val:>12d} {delta:>10}")
        print()


def main():
    print("=" * 60)
    print("DMS KV Cache Compression - Inference Test")
    print("=" * 60)

    device = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device} ({total_mem:.1f} GB)")

    # Test DMS-8x model
    print("\n>>> Testing DMS-8x Model <<<")
    dms_results = run_model_tests(
        "nvidia/Qwen3-8B-DMS-8x",
        trust_remote_code=True,
    )

    # Test vanilla Qwen3-8B for comparison
    print("\n>>> Testing Vanilla Qwen3-8B <<<")
    vanilla_results = run_model_tests("Qwen/Qwen3-8B")

    # Print comparison
    print_comparison(dms_results, vanilla_results)


if __name__ == "__main__":
    main()
