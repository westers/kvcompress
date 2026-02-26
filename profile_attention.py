"""Attention sparsity profiling module for DMS KV cache compression on Qwen3-8B.

This module provides two classes:
1. AttentionProfiler: Captures attention weight statistics during DMS decode steps
2. SparsityAnalyzer: Post-generation analysis of captured attention patterns
"""

import torch
import math
import json
from collections import defaultdict


class AttentionProfiler:
    """Captures attention weight statistics during DMS decode steps."""

    def __init__(self, num_layers=36, num_q_heads=32, num_kv_heads=4, top_k=32):
        """Initialize the profiler.

        Args:
            num_layers: Number of transformer layers (default: 36 for Qwen3-8B)
            num_q_heads: Number of query heads (default: 32)
            num_kv_heads: Number of KV heads for GQA (default: 4)
            top_k: Number of top attention positions to track (default: 32)
        """
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.top_k = top_k
        self.snapshots = []
        self.decode_step = 0

    @property
    def num_decode_steps(self):
        """Return the current number of completed decode steps."""
        return self.decode_step

    def capture(self, new_q, k_cache, cache_seq_lengths, layer_idx, decisions=None):
        """Compute attention weights manually and store lightweight statistics.

        All heavy computation stays on GPU; only scalar stats and top-k
        indices/values (O(top_k) per head) are moved to CPU.

        Args:
            new_q: Query states, shape [batch, 32, 1, 128] (bf16, CUDA)
            k_cache: Key cache, shape [batch, 4, cache_len, 128] (bf16, CUDA)
            cache_seq_lengths: Valid lengths per KV head, shape [batch, 4] (int32)
            layer_idx: Current layer index
            decisions: DMS eviction decisions, shape [batch, 4, 1] (bf16), optional
        """
        with torch.no_grad():
            # Stay on GPU for heavy computation, convert to float32
            q = new_q.float()  # [batch, 32, 1, 128]
            k = k_cache.float()  # [batch, 4, cache_len, 128]
            seq_lens = cache_seq_lengths  # [batch, 4], keep on GPU

            # GQA expansion - expand K from 4 to 32 heads
            q_per_kv = q.shape[1] // k.shape[1]  # 8
            k_expanded = k.repeat_interleave(q_per_kv, dim=1)  # [batch, 32, cache_len, 128]

            # Compute attention scores
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            # scores: [batch, 32, 1, cache_len]

            # Create validity mask from cache_seq_lengths (expand for GQA)
            seq_lens_expanded = seq_lens.repeat_interleave(q_per_kv, dim=1)  # [batch, 32]
            pos = torch.arange(k.shape[2], device=k.device)[None, None, None, :]
            mask = pos < seq_lens_expanded[:, :, None, None]  # [batch, 32, 1, cache_len]
            scores = scores.masked_fill(~mask, float('-inf'))

            # Softmax to get weights
            weights = torch.softmax(scores, dim=-1)  # [batch, 32, 1, cache_len]
            weights = weights.squeeze(2)  # [batch, 32, cache_len]

            # For batch=0 only (single-batch profiling), compute per-head stats
            w = weights[0]  # [32, cache_len]

            per_head_stats = []
            for h in range(self.num_q_heads):
                # Use per-KV-head valid length (DMS eviction is per-head)
                kv_head = h // q_per_kv
                valid_len = int(seq_lens[0, kv_head].item())
                head_weights = w[h, :valid_len]

                # Handle edge case: empty cache
                if valid_len == 0:
                    per_head_stats.append({
                        'head_idx': h,
                        'top_k_positions': [],
                        'top_k_values': [],
                        'entropy': 0.0,
                        'sparsity_at_1pct': 0,
                        'sink_attention': 0.0,
                        'locality_ratio': 0.0,
                    })
                    continue

                # Top-k positions and values
                k_actual = min(self.top_k, valid_len)
                topk_values, topk_indices = torch.topk(head_weights, k_actual)

                # Entropy: -sum(w * log(w + eps))
                entropy = -torch.sum(head_weights * torch.log(head_weights + 1e-10)).item()

                # Sparsity at 1%: number of tokens needed for 99% of attention mass
                sorted_weights, _ = torch.sort(head_weights, descending=True)
                cumsum = torch.cumsum(sorted_weights, dim=0)
                threshold_indices = (cumsum >= 0.99).nonzero(as_tuple=True)[0]
                if len(threshold_indices) > 0:
                    sparsity_at_1pct = int(threshold_indices[0].item()) + 1
                else:
                    sparsity_at_1pct = valid_len

                # Sink attention: sum of attention on first 4 positions
                sink_attention = head_weights[:min(4, valid_len)].sum().item()

                # Locality ratio: fraction of attention within last 128 positions
                local_window = min(128, valid_len)
                locality_ratio = head_weights[-local_window:].sum().item()

                per_head_stats.append({
                    'head_idx': h,
                    'top_k_positions': topk_indices.cpu().tolist(),
                    'top_k_values': topk_values.cpu().tolist(),
                    'entropy': entropy,
                    'sparsity_at_1pct': sparsity_at_1pct,
                    'sink_attention': sink_attention,
                    'locality_ratio': locality_ratio,
                })

            # Store snapshot — use max across KV heads for num_cached_tokens
            snapshot = {
                'decode_step': self.decode_step,
                'layer_idx': layer_idx,
                'num_cached_tokens': int(seq_lens[0].max().item()),
                'per_head_cache_len': seq_lens[0].cpu().tolist(),
                'heads': per_head_stats,
                'decisions': decisions[0, :, 0].cpu().tolist() if decisions is not None else None,
            }
            self.snapshots.append(snapshot)

            # If this is the last layer, increment decode_step
            if layer_idx == self.num_layers - 1:
                self.decode_step += 1


class SparsityAnalyzer:
    """Post-generation analysis of captured attention patterns."""

    def __init__(self, profiler: AttentionProfiler):
        """Initialize the analyzer with a profiler.

        Args:
            profiler: AttentionProfiler instance with captured snapshots
        """
        self.profiler = profiler

        # Organize snapshots by (decode_step, layer_idx) for easy lookup
        self.snapshot_map = {}
        for snapshot in profiler.snapshots:
            key = (snapshot['decode_step'], snapshot['layer_idx'])
            self.snapshot_map[key] = snapshot

    def analyze(self) -> dict:
        """Compute comprehensive sparsity report.

        Returns:
            Dictionary containing:
            - per_layer_summary: Aggregated stats per layer
            - per_head_summary: Detailed stats per (layer, head) pair
            - heavy_hitter_stability: Jaccard similarity of top-k across steps
            - cross_layer_similarity: Cosine similarity between adjacent layers
            - dms_correlation: Correlation between DMS decisions and attention
            - summary_stats: Overall statistics
        """
        analysis = {
            'per_layer_summary': self._compute_per_layer_summary(),
            'per_head_summary': self._compute_per_head_summary(),
            'heavy_hitter_stability': self._compute_heavy_hitter_stability(),
            'cross_layer_similarity': self._compute_cross_layer_similarity(),
            'dms_correlation': self._compute_dms_correlation(),
            'summary_stats': {},
        }

        # Compute summary stats from per_layer_summary
        analysis['summary_stats'] = self._compute_summary_stats(analysis)

        return analysis

    def _compute_per_layer_summary(self) -> dict:
        """Compute average statistics per layer across all heads and decode steps."""
        per_layer = {}

        for layer_idx in range(self.profiler.num_layers):
            entropies = []
            sparsities = []
            sinks = []
            localities = []

            for decode_step in range(self.profiler.num_decode_steps):
                key = (decode_step, layer_idx)
                if key not in self.snapshot_map:
                    continue

                snapshot = self.snapshot_map[key]
                for head_stats in snapshot['heads']:
                    entropies.append(head_stats['entropy'])
                    sparsities.append(head_stats['sparsity_at_1pct'])
                    sinks.append(head_stats['sink_attention'])
                    localities.append(head_stats['locality_ratio'])

            if entropies:
                per_layer[layer_idx] = {
                    'avg_entropy': sum(entropies) / len(entropies),
                    'avg_sparsity_at_1pct': sum(sparsities) / len(sparsities),
                    'avg_sink_attention': sum(sinks) / len(sinks),
                    'avg_locality_ratio': sum(localities) / len(localities),
                }
            else:
                per_layer[layer_idx] = {
                    'avg_entropy': 0.0,
                    'avg_sparsity_at_1pct': 0.0,
                    'avg_sink_attention': 0.0,
                    'avg_locality_ratio': 0.0,
                }

        return per_layer

    def _compute_per_head_summary(self) -> dict:
        """Compute statistics per (layer, head) pair and classify head types."""
        per_head = {}

        for layer_idx in range(self.profiler.num_layers):
            for head_idx in range(self.profiler.num_q_heads):
                entropies = []
                sparsities = []
                sinks = []
                localities = []
                num_cached_tokens_list = []

                for decode_step in range(self.profiler.num_decode_steps):
                    key = (decode_step, layer_idx)
                    if key not in self.snapshot_map:
                        continue

                    snapshot = self.snapshot_map[key]
                    head_stats = snapshot['heads'][head_idx]

                    entropies.append(head_stats['entropy'])
                    sparsities.append(head_stats['sparsity_at_1pct'])
                    sinks.append(head_stats['sink_attention'])
                    localities.append(head_stats['locality_ratio'])
                    num_cached_tokens_list.append(snapshot['num_cached_tokens'])

                if not entropies:
                    continue

                avg_entropy = sum(entropies) / len(entropies)
                avg_sparsity = sum(sparsities) / len(sparsities)
                avg_sink = sum(sinks) / len(sinks)
                avg_locality = sum(localities) / len(localities)
                avg_num_cached = sum(num_cached_tokens_list) / len(num_cached_tokens_list)

                # Classify head type
                if avg_sink > 0.5:
                    head_type = "sink-dominated"
                elif avg_locality > 0.7:
                    head_type = "local"
                elif avg_num_cached > 0 and avg_entropy > math.log(avg_num_cached) * 0.9:
                    head_type = "diffuse"
                else:
                    head_type = "retrieval"

                per_head[(layer_idx, head_idx)] = {
                    'avg_entropy': avg_entropy,
                    'avg_sparsity_at_1pct': avg_sparsity,
                    'avg_sink_attention': avg_sink,
                    'avg_locality_ratio': avg_locality,
                    'head_type': head_type,
                }

        return per_head

    def _compute_heavy_hitter_stability(self) -> dict:
        """Compute Jaccard similarity of top-k token sets across consecutive decode steps."""
        stability = {}

        for layer_idx in range(self.profiler.num_layers):
            jaccard_scores = []

            for decode_step in range(self.profiler.num_decode_steps - 1):
                key1 = (decode_step, layer_idx)
                key2 = (decode_step + 1, layer_idx)

                if key1 not in self.snapshot_map or key2 not in self.snapshot_map:
                    continue

                snapshot1 = self.snapshot_map[key1]
                snapshot2 = self.snapshot_map[key2]

                # Average Jaccard across all heads
                head_jaccards = []
                for h in range(self.profiler.num_q_heads):
                    topk1 = set(snapshot1['heads'][h]['top_k_positions'])
                    topk2 = set(snapshot2['heads'][h]['top_k_positions'])

                    if len(topk1) == 0 and len(topk2) == 0:
                        jaccard = 1.0
                    elif len(topk1) == 0 or len(topk2) == 0:
                        jaccard = 0.0
                    else:
                        intersection = len(topk1 & topk2)
                        union = len(topk1 | topk2)
                        jaccard = intersection / union if union > 0 else 0.0

                    head_jaccards.append(jaccard)

                if head_jaccards:
                    jaccard_scores.append(sum(head_jaccards) / len(head_jaccards))

            stability[layer_idx] = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

        return stability

    def _compute_cross_layer_similarity(self) -> dict:
        """Compute cosine similarity of average attention vectors between adjacent layers."""
        similarity = {}

        for layer_idx in range(self.profiler.num_layers - 1):
            cosine_scores = []

            for decode_step in range(self.profiler.num_decode_steps):
                key1 = (decode_step, layer_idx)
                key2 = (decode_step, layer_idx + 1)

                if key1 not in self.snapshot_map or key2 not in self.snapshot_map:
                    continue

                snapshot1 = self.snapshot_map[key1]
                snapshot2 = self.snapshot_map[key2]

                # Get cache length (should be same for both)
                cache_len = snapshot1['num_cached_tokens']

                if cache_len == 0:
                    continue

                # Compute average attention vector across heads for each layer
                # Reconstruct full attention vector from top-k (approximate)
                avg_attn1 = torch.zeros(cache_len)
                avg_attn2 = torch.zeros(cache_len)

                for h in range(self.profiler.num_q_heads):
                    head1 = snapshot1['heads'][h]
                    head2 = snapshot2['heads'][h]

                    # Approximate full attention from top-k
                    for pos, val in zip(head1['top_k_positions'], head1['top_k_values']):
                        if pos < cache_len:
                            avg_attn1[pos] += val

                    for pos, val in zip(head2['top_k_positions'], head2['top_k_values']):
                        if pos < cache_len:
                            avg_attn2[pos] += val

                # Normalize
                avg_attn1 = avg_attn1 / self.profiler.num_q_heads
                avg_attn2 = avg_attn2 / self.profiler.num_q_heads

                # Cosine similarity
                norm1 = torch.norm(avg_attn1)
                norm2 = torch.norm(avg_attn2)

                if norm1 > 0 and norm2 > 0:
                    cosine = torch.dot(avg_attn1, avg_attn2) / (norm1 * norm2)
                    cosine_scores.append(cosine.item())

            similarity[(layer_idx, layer_idx + 1)] = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0

        return similarity

    def _compute_dms_correlation(self) -> dict:
        """Analyze DMS eviction decisions and attention patterns.

        The DMS model was jointly trained with eviction — the model learned to
        produce useful outputs even after tokens are evicted. So attention patterns
        already reflect the model's adaptation to eviction. We measure:

        1. Eviction rate per layer (fraction of tokens marked for eviction)
        2. Decision-attention separation: how attention stats differ between
           evict and keep decisions (reflects the model's learned strategy)
        3. Offload opportunity: for CPU offloading, we care about whether
           evicted tokens are EVER important in future steps. If evicted tokens
           rarely appear in top-k across subsequent steps, they're safe to
           discard (or offload to CPU as cold storage).

        Returns per-layer dict with eviction stats and offload analysis.
        """
        correlation = {}
        q_per_kv = self.profiler.num_q_heads // self.profiler.num_kv_heads

        for layer_idx in range(self.profiler.num_layers):
            evict_locality = []
            keep_locality = []
            evict_entropy = []
            keep_entropy = []
            evict_sink = []
            keep_sink = []

            # Track positions marked for eviction and check future top-k presence
            # key: kv_head, value: set of cache positions marked for eviction
            evicted_positions = defaultdict(set)
            # Count how often evicted positions appear in future top-k
            future_topk_hits = 0
            future_topk_checks = 0

            for decode_step in range(self.profiler.num_decode_steps):
                key = (decode_step, layer_idx)
                if key not in self.snapshot_map:
                    continue

                snapshot = self.snapshot_map[key]

                # Check if any previously-evicted positions appear in current top-k
                for kv_head_idx in range(self.profiler.num_kv_heads):
                    if evicted_positions[kv_head_idx]:
                        q_head_start = kv_head_idx * q_per_kv
                        for q_offset in range(q_per_kv):
                            q_head_idx = q_head_start + q_offset
                            head_stats = snapshot['heads'][q_head_idx]
                            topk_set = set(head_stats['top_k_positions'])
                            hits = len(evicted_positions[kv_head_idx] & topk_set)
                            future_topk_hits += hits
                            future_topk_checks += len(evicted_positions[kv_head_idx])

                if snapshot['decisions'] is None:
                    continue

                for kv_head_idx in range(self.profiler.num_kv_heads):
                    decision = snapshot['decisions'][kv_head_idx]
                    q_head_start = kv_head_idx * q_per_kv

                    # Track evicted position: use per-head cache len - 1 as
                    # the approximate position of the decided token
                    if decision == 1 and 'per_head_cache_len' in snapshot:
                        head_len = int(snapshot['per_head_cache_len'][kv_head_idx])
                        if head_len > 0:
                            evicted_positions[kv_head_idx].add(head_len - 1)

                    # Aggregate attention stats across Q heads in this KV group
                    group_locality = []
                    group_entropy = []
                    group_sink = []

                    for q_offset in range(q_per_kv):
                        q_head_idx = q_head_start + q_offset
                        head_stats = snapshot['heads'][q_head_idx]
                        group_locality.append(head_stats['locality_ratio'])
                        group_entropy.append(head_stats['entropy'])
                        group_sink.append(head_stats['sink_attention'])

                    mean_loc = sum(group_locality) / len(group_locality)
                    mean_ent = sum(group_entropy) / len(group_entropy)
                    mean_sink = sum(group_sink) / len(group_sink)

                    if decision == 1:
                        evict_locality.append(mean_loc)
                        evict_entropy.append(mean_ent)
                        evict_sink.append(mean_sink)
                    else:
                        keep_locality.append(mean_loc)
                        keep_entropy.append(mean_ent)
                        keep_sink.append(mean_sink)

            total = len(evict_locality) + len(keep_locality)
            if total == 0:
                correlation[layer_idx] = None
                continue

            eviction_rate = len(evict_locality) / total

            mean_evict_loc = sum(evict_locality) / len(evict_locality) if evict_locality else 0.0
            mean_keep_loc = sum(keep_locality) / len(keep_locality) if keep_locality else 0.0
            mean_evict_ent = sum(evict_entropy) / len(evict_entropy) if evict_entropy else 0.0
            mean_keep_ent = sum(keep_entropy) / len(keep_entropy) if keep_entropy else 0.0
            mean_evict_sink = sum(evict_sink) / len(evict_sink) if evict_sink else 0.0
            mean_keep_sink = sum(keep_sink) / len(keep_sink) if keep_sink else 0.0

            # Separations: show how the model's attention differs between
            # evict and keep decisions (reflects learned strategy, not "quality")
            loc_sep = mean_keep_loc - mean_evict_loc
            ent_sep = mean_evict_ent - mean_keep_ent
            sink_sep = mean_evict_sink - mean_keep_sink

            # Offload safety: how often do evicted positions show up in future top-k?
            # Low rate = evicted tokens are truly unneeded = safe to offload/discard
            future_hit_rate = future_topk_hits / future_topk_checks if future_topk_checks > 0 else 0.0

            correlation[layer_idx] = {
                'eviction_rate': round(eviction_rate, 4),
                'keep_locality': round(mean_keep_loc, 4),
                'evict_locality': round(mean_evict_loc, 4),
                'locality_separation': round(loc_sep, 4),
                'keep_entropy': round(mean_keep_ent, 4),
                'evict_entropy': round(mean_evict_ent, 4),
                'entropy_separation': round(ent_sep, 4),
                'keep_sink': round(mean_keep_sink, 4),
                'evict_sink': round(mean_evict_sink, 4),
                'sink_separation': round(sink_sep, 4),
                'future_hit_rate': round(future_hit_rate, 6),
                'evicted_positions_tracked': sum(len(s) for s in evicted_positions.values()),
            }

        return correlation

    def _compute_summary_stats(self, analysis: dict) -> dict:
        """Compute overall summary statistics from the analysis."""
        per_layer = analysis['per_layer_summary']
        per_head = analysis['per_head_summary']

        if not per_layer:
            return {
                'mean_sparsity_at_1pct': 0.0,
                'mean_entropy': 0.0,
                'pct_sink_dominated_heads': 0.0,
                'pct_local_heads': 0.0,
                'pct_retrieval_heads': 0.0,
                'pct_diffuse_heads': 0.0,
            }

        # Mean across layers
        mean_sparsity = sum(layer['avg_sparsity_at_1pct'] for layer in per_layer.values()) / len(per_layer)
        mean_entropy = sum(layer['avg_entropy'] for layer in per_layer.values()) / len(per_layer)

        # Head type distribution
        head_types = [head['head_type'] for head in per_head.values()]
        total_heads = len(head_types)

        pct_sink = sum(1 for ht in head_types if ht == 'sink-dominated') / total_heads * 100 if total_heads > 0 else 0.0
        pct_local = sum(1 for ht in head_types if ht == 'local') / total_heads * 100 if total_heads > 0 else 0.0
        pct_retrieval = sum(1 for ht in head_types if ht == 'retrieval') / total_heads * 100 if total_heads > 0 else 0.0
        pct_diffuse = sum(1 for ht in head_types if ht == 'diffuse') / total_heads * 100 if total_heads > 0 else 0.0

        return {
            'mean_sparsity_at_1pct': mean_sparsity,
            'mean_entropy': mean_entropy,
            'pct_sink_dominated_heads': pct_sink,
            'pct_local_heads': pct_local,
            'pct_retrieval_heads': pct_retrieval,
            'pct_diffuse_heads': pct_diffuse,
        }

    def print_report(self, analysis: dict = None):
        """Print a human-readable text report of the analysis.

        Args:
            analysis: Analysis dictionary from analyze(). If None, will run analyze().
        """
        if analysis is None:
            analysis = self.analyze()

        print("=" * 80)
        print("ATTENTION SPARSITY ANALYSIS REPORT")
        print("=" * 80)
        print()

        # Per-layer summary
        print("PER-LAYER SUMMARY")
        print("-" * 80)
        print(f"{'Layer':<8} {'Entropy':<12} {'Sparsity@1%':<14} {'Sink':<12} {'Locality':<12}")
        print("-" * 80)

        for layer_idx in sorted(analysis['per_layer_summary'].keys()):
            layer_data = analysis['per_layer_summary'][layer_idx]
            print(f"{layer_idx:<8} "
                  f"{layer_data['avg_entropy']:<12.3f} "
                  f"{layer_data['avg_sparsity_at_1pct']:<14.2f} "
                  f"{layer_data['avg_sink_attention']:<12.3f} "
                  f"{layer_data['avg_locality_ratio']:<12.3f}")
        print()

        # Head type distribution
        print("HEAD TYPE DISTRIBUTION")
        print("-" * 80)
        summary = analysis['summary_stats']
        print(f"Sink-dominated: {summary['pct_sink_dominated_heads']:.1f}%")
        print(f"Local:          {summary['pct_local_heads']:.1f}%")
        print(f"Retrieval:      {summary['pct_retrieval_heads']:.1f}%")
        print(f"Diffuse:        {summary['pct_diffuse_heads']:.1f}%")
        print()

        # Heavy-hitter stability
        print("HEAVY-HITTER STABILITY (Jaccard Similarity)")
        print("-" * 80)
        stability = analysis['heavy_hitter_stability']
        if stability:
            avg_stability = sum(stability.values()) / len(stability)
            print(f"Average stability across layers: {avg_stability:.3f}")
            print(f"Min stability: {min(stability.values()):.3f} (Layer {min(stability, key=stability.get)})")
            print(f"Max stability: {max(stability.values()):.3f} (Layer {max(stability, key=stability.get)})")
        else:
            print("No stability data available")
        print()

        # Cross-layer similarity
        print("CROSS-LAYER SIMILARITY (Cosine Similarity)")
        print("-" * 80)
        cross_layer = analysis['cross_layer_similarity']
        if cross_layer:
            avg_similarity = sum(cross_layer.values()) / len(cross_layer)
            print(f"Average similarity between adjacent layers: {avg_similarity:.3f}")
        else:
            print("No cross-layer data available")
        print()

        # DMS correlation
        print("DMS EVICTION ANALYSIS")
        print("-" * 80)
        dms_corr = analysis['dms_correlation']
        valid_corr = {k: v for k, v in dms_corr.items() if v is not None}
        if valid_corr:
            avg_eviction_rate = sum(v['eviction_rate'] for v in valid_corr.values()) / len(valid_corr)
            avg_loc_sep = sum(v['locality_separation'] for v in valid_corr.values()) / len(valid_corr)
            avg_ent_sep = sum(v['entropy_separation'] for v in valid_corr.values()) / len(valid_corr)
            avg_sink_sep = sum(v['sink_separation'] for v in valid_corr.values()) / len(valid_corr)
            avg_future_hit = sum(v['future_hit_rate'] for v in valid_corr.values()) / len(valid_corr)

            print(f"Eviction rate:      {avg_eviction_rate:.1%}")
            print(f"Layers with data:   {len(valid_corr)}")
            print()
            print("Decision-Attention Separation (model's learned strategy):")
            print(f"  Locality (keep - evict): {avg_loc_sep:+.4f}")
            print(f"  Entropy  (evict - keep): {avg_ent_sep:+.4f}")
            print(f"  Sink     (evict - keep): {avg_sink_sep:+.4f}")
            print()
            print("CPU Offload Safety:")
            print(f"  Future top-k hit rate: {avg_future_hit:.4f}")
            print(f"  (Low = evicted tokens rarely needed later = safe to offload)")

            # Per-layer future hit rates
            hit_rates = [(k, v['future_hit_rate']) for k, v in valid_corr.items()]
            hit_rates.sort(key=lambda x: x[1], reverse=True)
            print(f"\n  Highest regret: Layer {hit_rates[0][0]} (hit_rate={hit_rates[0][1]:.4f})")
            print(f"  Lowest regret:  Layer {hit_rates[-1][0]} (hit_rate={hit_rates[-1][1]:.4f})")
        else:
            print("No DMS decision data available")
        print()

        # Summary stats
        print("OVERALL SUMMARY")
        print("-" * 80)
        print(f"Mean sparsity at 1%: {summary['mean_sparsity_at_1pct']:.2f} tokens")
        print(f"Mean entropy:        {summary['mean_entropy']:.3f}")
        print(f"Total snapshots:     {len(self.profiler.snapshots)}")
        print(f"Decode steps:        {self.profiler.num_decode_steps}")
        print("=" * 80)

    def save_report(self, filepath: str, analysis: dict = None):
        """Save the full analysis dict as JSON.

        Args:
            filepath: Path to save JSON report
            analysis: Analysis dictionary from analyze(). If None, will run analyze().
        """
        if analysis is None:
            analysis = self.analyze()

        # Convert all numeric types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(v) for v in obj)
            elif isinstance(obj, (torch.Tensor, torch.FloatTensor, torch.IntTensor)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj

        # Convert analysis to native types
        native_analysis = convert_to_native(analysis)

        # Convert tuple keys to strings for JSON serialization
        if 'per_head_summary' in native_analysis:
            native_analysis['per_head_summary'] = {
                f"{k[0]}_{k[1]}": v for k, v in native_analysis['per_head_summary'].items()
            }

        if 'cross_layer_similarity' in native_analysis:
            native_analysis['cross_layer_similarity'] = {
                f"{k[0]}_{k[1]}": v for k, v in native_analysis['cross_layer_similarity'].items()
            }

        with open(filepath, 'w') as f:
            json.dump(native_analysis, f, indent=2)
