"""
kvpress Library Exploration

Tests multiple compression strategies via the kvpress pipeline:
- ExpectedAttentionPress, KnormPress, SnapKVPress, StreamingLLMPress
- DMSPress (threshold-based eviction)
- DecodingPress (decoding-phase compression)
- Varying compression ratios (0.25, 0.5, 0.75)
"""

import time
import torch
from transformers import pipeline as hf_pipeline
from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    SnapKVPress,
    StreamingLLMPress,
    DMSPress,
    DecodingPress,
)

MODEL_ID = "Qwen/Qwen3-8B"

# A long-ish context for meaningful compression testing
CONTEXT = """
Quantum computing represents a fundamentally different approach to computation. Unlike classical
computers that use bits representing 0 or 1, quantum computers use quantum bits (qubits) that can
exist in superposition — simultaneously representing both 0 and 1. This property, combined with
quantum entanglement and interference, allows quantum computers to solve certain problems
exponentially faster than classical machines.

The development of quantum computing has progressed through several key milestones. In 1994, Peter
Shor developed an algorithm that could factor large numbers exponentially faster than any known
classical algorithm, threatening modern cryptography. In 1996, Lov Grover created an algorithm
providing quadratic speedup for unstructured search problems. These theoretical advances drove
massive investment in building physical quantum computers.

Modern quantum hardware comes in several forms. Superconducting qubits, used by IBM and Google,
operate at temperatures near absolute zero and manipulate microwave photons. Trapped ion systems,
developed by IonQ and Quantinuum, use electromagnetic fields to suspend individual atoms and
manipulate them with lasers. Photonic quantum computers, like those from Xanadu, encode information
in light particles. Each approach has tradeoffs in coherence time, gate fidelity, and scalability.

In 2019, Google claimed "quantum supremacy" with their 53-qubit Sycamore processor, completing a
specific calculation in 200 seconds that they estimated would take a classical supercomputer 10,000
years. This claim was contested by IBM, who argued classical simulation could be done in 2.5 days
with sufficient resources. Regardless of the exact comparison, the experiment demonstrated that
quantum processors can outperform classical machines on specifically designed tasks.

Error correction remains the central challenge in quantum computing. Qubits are extremely fragile —
environmental noise causes decoherence, destroying the quantum information. Quantum error correction
codes protect information by encoding one logical qubit across many physical qubits. Current estimates
suggest thousands of physical qubits may be needed for each logical qubit, meaning millions of
physical qubits may be required for practical fault-tolerant quantum computing.

The applications of quantum computing span multiple domains. In chemistry, quantum simulation could
revolutionize drug discovery by accurately modeling molecular interactions. In optimization, quantum
algorithms could improve logistics, financial portfolio management, and machine learning. In
cryptography, quantum computers threaten current encryption standards but also enable quantum key
distribution for theoretically unbreakable communication.

The quantum computing industry has attracted billions in investment. Major technology companies
including IBM, Google, Microsoft, and Amazon have dedicated quantum computing divisions. Startups
like Rigetti, IonQ, and PsiQuantum have raised hundreds of millions in funding. Governments
worldwide have launched national quantum initiatives, with China, the US, and EU each committing
billions to quantum research.

Despite the hype, practical quantum advantage for real-world problems remains elusive. Current
quantum computers are in the "noisy intermediate-scale quantum" (NISQ) era, with too few qubits
and too much noise for most practical applications. The timeline for fault-tolerant quantum
computing ranges from optimistic estimates of 5-10 years to more conservative predictions of
20-30 years. The field continues to advance rapidly, but significant engineering challenges remain
before quantum computers can fulfill their transformative potential.
""".strip()

QUESTION = "\nWhat are the main approaches to building quantum computer hardware, and what are their tradeoffs?"


def create_pipeline():
    """Create the kvpress text generation pipeline."""
    print(f"Loading model {MODEL_ID}...")
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Pipeline ready.")
    return pipe


def test_press(pipe, press, name, max_new_tokens=256):
    """Test a single press configuration and return metrics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    result = pipe(
        CONTEXT,
        question=QUESTION,
        press=press,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    answer = result["answer"]

    return {
        "name": name,
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": round(peak_mem_gb, 2),
        "answer_len": len(answer),
        "answer_preview": answer[:300],
    }


def test_compression_ratios(pipe):
    """Test different compression methods at various ratios."""
    print("\n" + "=" * 70)
    print("PART 1: Compression Methods at Different Ratios")
    print("=" * 70)

    press_classes = [
        ("KnormPress", KnormPress),
        ("ExpectedAttentionPress", ExpectedAttentionPress),
        ("SnapKVPress", SnapKVPress),
        ("StreamingLLMPress", StreamingLLMPress),
    ]

    ratios = [0.25, 0.5, 0.75]
    results = []

    for cls_name, cls in press_classes:
        for ratio in ratios:
            name = f"{cls_name}(cr={ratio})"
            print(f"\n--- {name} ---")
            press = cls(compression_ratio=ratio)
            metrics = test_press(pipe, press, name)
            results.append(metrics)
            print(f"  Time: {metrics['elapsed_sec']}s | VRAM: {metrics['peak_vram_gb']} GB")
            print(f"  Answer ({metrics['answer_len']} chars): {metrics['answer_preview'][:150]}...")

    return results


def test_dms_press(pipe):
    """Test DMSPress with threshold-based eviction."""
    print("\n" + "=" * 70)
    print("PART 2: DMSPress (Threshold-Based Eviction)")
    print("=" * 70)

    results = []
    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        press = DMSPress(
            press=KnormPress(),
            threshold=threshold,
            sliding_window_size=128,
        )
        name = f"DMSPress(threshold={threshold})"
        print(f"\n--- {name} ---")
        metrics = test_press(pipe, press, name)
        results.append(metrics)
        print(f"  Time: {metrics['elapsed_sec']}s | VRAM: {metrics['peak_vram_gb']} GB")
        print(f"  Answer ({metrics['answer_len']} chars): {metrics['answer_preview'][:150]}...")

    return results


def test_decoding_press(pipe):
    """Test DecodingPress for decoding-phase compression."""
    print("\n" + "=" * 70)
    print("PART 3: DecodingPress (Decoding-Phase Compression)")
    print("=" * 70)

    results = []
    configs = [
        {"target_size": 512, "compression_interval": 256},
        {"target_size": 1024, "compression_interval": 512},
        {"target_size": 2048, "compression_interval": 512},
    ]

    for cfg in configs:
        press = DecodingPress(
            base_press=KnormPress(),
            target_size=cfg["target_size"],
            compression_interval=cfg["compression_interval"],
        )
        name = f"DecodingPress(target={cfg['target_size']}, interval={cfg['compression_interval']})"
        print(f"\n--- {name} ---")
        metrics = test_press(pipe, press, name, max_new_tokens=512)
        results.append(metrics)
        print(f"  Time: {metrics['elapsed_sec']}s | VRAM: {metrics['peak_vram_gb']} GB")
        print(f"  Answer ({metrics['answer_len']} chars): {metrics['answer_preview'][:150]}...")

    return results


def print_summary(all_results):
    """Print summary table of all results."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = f"{'Method':<55} {'Time(s)':>8} {'VRAM(GB)':>9} {'Ans Len':>8}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(f"{r['name']:<55} {r['elapsed_sec']:>8.2f} {r['peak_vram_gb']:>9.2f} {r['answer_len']:>8}")


def main():
    print("=" * 60)
    print("kvpress Library API Exploration")
    print("=" * 60)

    device = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device} ({total_mem:.1f} GB)")

    pipe = create_pipeline()

    # Run a baseline (no compression) for reference
    print("\n--- Baseline (no compression) ---")
    baseline_press = KnormPress(compression_ratio=0.0)
    baseline = test_press(pipe, baseline_press, "Baseline (no compression)")
    print(f"  Time: {baseline['elapsed_sec']}s | VRAM: {baseline['peak_vram_gb']} GB")
    print(f"  Answer: {baseline['answer_preview'][:200]}...")

    all_results = [baseline]
    all_results.extend(test_compression_ratios(pipe))
    all_results.extend(test_dms_press(pipe))
    all_results.extend(test_decoding_press(pipe))

    print_summary(all_results)


if __name__ == "__main__":
    main()
