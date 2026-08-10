# Phase 6 · Lesson 4 — Transformer Internals

> Prerequisite: Tokenization, Embeddings, Positional Encoding (Lessons 1–3), Phase 5 Lesson 7 (Transformers)

---

## 1. Introduction

### What does this lesson cover?
A deeper look at the specific architectural refinements that distinguish modern (2023-2026) production LLMs from the original 2017 Transformer: efficient attention variants (Multi-Query and Grouped-Query Attention), modern normalization choices (RMSNorm), modern activation functions (SwiGLU), and the KV-cache — the single most important inference-time optimization for autoregressive generation. Where Phase 5 Lesson 7 built the *conceptual* Transformer block, this lesson covers the *engineering* refinements that make it efficient at real production scale.

### Why does it exist?
The original Transformer architecture works, but running inference on a 70-billion-parameter model, one token at a time, for a user waiting on a response, exposes efficiency bottlenecks invisible at small scale — redundant computation, excessive memory bandwidth use, and unnecessarily large KV storage. The specific refinements in this lesson are direct, measured engineering responses to those bottlenecks, each addressing a concrete cost or latency problem at production LLM scale.

### Historical background
Multi-Query Attention (Shazeer, 2019) and its generalization Grouped-Query Attention (Ainslie et al., 2023) emerged specifically to address KV-cache memory bandwidth as a serving bottleneck. RMSNorm (Zhang & Sennrich, 2019) simplified LayerNorm for comparable performance at lower computational cost. SwiGLU (Shazeer, 2020) improved on the standard ReLU/GELU feedforward activation. These refinements collectively define the architecture used by Llama, Mistral, and most other modern open-weight models — genuinely distinct from, if closely related to, the original 2017 design.

### Real-world motivation
If you ever fine-tune, quantize (Lesson 8), or deploy an open-weight LLM, you will encounter these exact architectural terms in model configs (`num_key_value_heads`, `rms_norm_eps`) — this lesson makes them concrete rather than opaque configuration values.

---

## 2. Theory

### The KV-cache — the central inference-time optimization
During autoregressive generation (Phase 5 Lesson 7's `generate` loop), each new token's query attends to *all* previous tokens' keys and values. Without caching, generating token $t$ would require recomputing keys/values for tokens $1, \dots, t-1$ from scratch — wastefully redundant, since those keys/values never change once computed. The **KV-cache** stores previously computed keys/values, so generating each new token only requires computing *that token's* new key/value pair and reusing the cached rest — converting an $O(n^2)$-per-token-generated cost pattern into $O(n)$ per new token (still $O(n^2)$ total across a full generation, but avoiding *repeated* redundant computation).

### Multi-Head, Multi-Query, and Grouped-Query Attention
- **Multi-Head Attention (MHA)**, Phase 5 Lesson 6: separate $K, V$ projections *per head* — most expressive, but the KV-cache scales linearly with the number of heads, becoming a serious memory bandwidth bottleneck at large scale (loading a large KV-cache from memory for every generated token is often the actual inference speed bottleneck, more than raw compute).
- **Multi-Query Attention (MQA)**: all heads *share* a single $K, V$ projection (only $Q$ remains per-head) — dramatically shrinks the KV-cache (by a factor of the number of heads), at some cost to model quality/expressiveness.
- **Grouped-Query Attention (GQA)**: a middle ground — heads are divided into groups, each group sharing one $K,V$ projection — tunable tradeoff between MHA's quality and MQA's efficiency, and the choice used by most modern (2024-2026) production LLMs (Llama 2 70B onward, Mistral, and others).

### RMSNorm — a simplified, cheaper alternative to LayerNorm
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$
Unlike LayerNorm (Phase 5 Lesson 7), RMSNorm **omits the mean-centering step** — it only rescales by the root-mean-square, without subtracting the mean first. Empirically, this simplification loses essentially no performance while being computationally cheaper (one fewer reduction operation per normalization call) — a genuine "simpler is just as good, and faster" architectural finding, now standard in Llama, Mistral, and most modern LLMs.

### SwiGLU — a gated, smoother feedforward activation
$$
\text{SwiGLU}(x) = (xW_1 \odot \text{Swish}(xW_2))W_3, \qquad \text{Swish}(x) = x \cdot \sigma(x)
$$
A **gated linear unit** variant: one linear projection is passed through a Swish activation, then used to *gate* (elementwise multiply) a second linear projection — empirically outperforming the simpler ReLU/GELU feedforward (Phase 5 Lesson 7) used in the original Transformer, at the cost of a third weight matrix (hence needing correspondingly narrower hidden dimensions to keep total parameter count comparable).

---

## 3. Mathematical Foundations

### KV-cache memory footprint, quantified
For a model with $L$ layers, $h$ heads, head dimension $d_h$, and sequence length $n$ (in `float16`, 2 bytes/value):
$$
\text{KV-cache size} = 2 \, (\text{K and V}) \times L \times h \times d_h \times n \times 2 \text{ bytes}
$$
For a 70B-parameter-scale model ($L=80$, $h=64$, $d_h=128$) at a 4096-token context, this reaches **tens of gigabytes per sequence** — directly explaining why serving many concurrent long-context requests is a genuine memory (not just compute) engineering challenge, and precisely why GQA's KV-cache reduction (shrinking the *effective* $h$ for K/V storage by the grouping factor) is such a consequential optimization at scale.

### GQA's memory savings, formalized
With $h$ query heads grouped into $g$ groups (each group sharing one K/V head), the KV-cache shrinks by a factor of $h/g$ compared to standard MHA — e.g., 64 query heads grouped into 8 KV groups yields an 8× KV-cache reduction, a substantial, direct, quantifiable memory/bandwidth saving with empirically modest quality cost (assuming $g$ is chosen sensibly, not too aggressively small).

### Why RMSNorm's simplification doesn't hurt performance (an empirical/intuitive argument)
LayerNorm's mean-centering step primarily helps when a layer's outputs have a genuinely meaningful, informative *shift* (non-zero mean) that needs to be corrected for stable training. Empirically, in Transformer architectures, this component appears to contribute little beyond what the rescaling (RMS-normalization) step alone captures — an empirical finding (not a fully first-principles theoretical guarantee) validated across numerous large-scale training runs, which is *why* it's now trusted as a standard default rather than a purely theoretical curiosity.

### FLOPs and memory bandwidth — the two distinct inference bottlenecks
Generating one token at a time (autoregressive decoding) is fundamentally **memory-bandwidth-bound**, not compute-bound: the model must load its full parameter set (and the KV-cache) from memory for each single-token forward pass, while doing comparatively little actual arithmetic per token — the exact opposite bottleneck profile of *training* (which processes many tokens in parallel per forward pass, making it more compute-bound). This distinction directly explains why techniques targeting memory footprint (GQA, quantization — Lesson 8) matter enormously for *inference* serving cost/latency, sometimes more than raw FLOP-reduction techniques would.

---

## 4. Algorithm — Autoregressive Generation WITH KV-Caching (fully specified)

```
GIVEN a prompt of length p, and a trained model:
1. PROCESS the full prompt in one forward pass (parallelizable, as in training)
   -> CACHE every layer's K, V tensors for all p prompt positions
2. FOR each new token to generate:
     a. Take ONLY the most recently generated token as input (not the whole sequence again!)
     b. Compute its Q, K, V
     c. APPEND this new K, V to the cache (cache now covers p + (tokens generated so far))
     d. Compute attention using this token's Q against the ENTIRE cached K, V (old + new)
     e. Produce the next token's logits, sample/argmax the next token
3. REPEAT step 2 until a stop condition (end-of-sequence token, max length) is reached
```
Without KV-caching, step 2 would need to reprocess the *entire* sequence from scratch at every single new token — an $O(n)$-times-more-expensive generation loop; with caching, each new token step only computes that one token's new K/V and reuses everything else, a dramatic and essential practical speedup for any real-time LLM serving system.

---

## 5. Python Implementation

```python
"""transformer_internals_core.py — RMSNorm, SwiGLU, GQA, and a from-scratch KV-cache"""
import numpy as np


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def swish(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def swiglu_ffn(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    return (x @ W1) * swish(x @ W2) @ W3   # NOTE: careful operator precedence -- (x@W1 * swish(x@W2)) @ W3


class KVCache:
    """A from-scratch KV-cache -- the essential inference optimization (Section 2/4)."""
    def __init__(self):
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def append(self, new_k: np.ndarray, new_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.keys.append(new_k)
        self.values.append(new_v)
        return np.vstack(self.keys), np.vstack(self.values)   # returns the FULL cached K, V so far

    def __len__(self) -> int:
        return len(self.keys)


def grouped_query_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_query_heads: int, n_kv_groups: int):
    """Q: (n_query_heads, seq_len, d_head). K, V: (n_kv_groups, seq_len, d_head).
    Each query head is mapped to ONE kv group (heads_per_group = n_query_heads / n_kv_groups)."""
    heads_per_group = n_query_heads // n_kv_groups
    outputs = []
    for h in range(n_query_heads):
        group_idx = h // heads_per_group           # which KV group this query head uses
        d_k = Q.shape[-1]
        scores = Q[h] @ K[group_idx].T / np.sqrt(d_k)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        outputs.append(weights @ V[group_idx])
    return np.stack(outputs)   # (n_query_heads, seq_len, d_head)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16))
    gamma = np.ones(16)
    print("RMSNorm output shape:", rms_norm(x, gamma).shape)

    # Simulate KV-cache growth across 5 generation steps
    cache = KVCache()
    for t in range(5):
        new_k = rng.normal(size=(1, 8))
        new_v = rng.normal(size=(1, 8))
        full_k, full_v = cache.append(new_k, new_v)
        print(f"Step {t}: cache now covers {len(cache)} positions, K shape {full_k.shape}")

    # GQA example: 8 query heads, 2 KV groups (4 query heads share each KV group)
    Q = rng.normal(size=(8, 6, 4))
    K = rng.normal(size=(2, 6, 4))
    V = rng.normal(size=(2, 6, 4))
    gqa_out = grouped_query_attention(Q, K, V, n_query_heads=8, n_kv_groups=2)
    print("GQA output shape:", gqa_out.shape)
```

---

## 6. Build From Scratch

**Quantifying KV-cache memory savings from GQA (making Section 3's formula concrete):**
```python
def kv_cache_memory_mb(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int, bytes_per_val: int = 2) -> float:
    total_values = 2 * n_layers * n_kv_heads * d_head * seq_len   # 2 for K AND V
    return total_values * bytes_per_val / (1024 ** 2)

# A 70B-scale model: 80 layers, 64 attention heads, d_head=128, comparing MHA vs GQA (8 KV groups)
mha_memory = kv_cache_memory_mb(n_layers=80, n_kv_heads=64, d_head=128, seq_len=4096)
gqa_memory = kv_cache_memory_mb(n_layers=80, n_kv_heads=8, d_head=128, seq_len=4096)
print(f"MHA KV-cache: {mha_memory:.1f} MB per sequence")
print(f"GQA KV-cache: {gqa_memory:.1f} MB per sequence ({mha_memory/gqa_memory:.1f}x smaller)")
```
Running this concretely shows an 8× reduction in KV-cache size — directly translating to either 8× more concurrent sequences servable in the same GPU memory, or the ability to serve much longer contexts within a fixed memory budget, a genuinely material production engineering consideration.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `KVCache` (Python lists) | Hugging Face Transformers' built-in `past_key_values` caching, or specialized inference servers (vLLM's PagedAttention) using highly optimized, memory-efficient cache management |
| `grouped_query_attention` | Native GQA support in Llama/Mistral model implementations, fused into optimized attention kernels (FlashAttention-2 with GQA support) |
| `rms_norm`/`swiglu_ffn` | `torch.nn.RMSNorm` (native support added in recent PyTorch versions) and standard SwiGLU implementations in Hugging Face model code |

---

## 8. Visual Explanations

**MHA vs MQA vs GQA (K/V head sharing patterns):**
```
MHA (Multi-Head):        MQA (Multi-Query):        GQA (Grouped-Query):
Q1-K1,V1                  Q1─┐                       Q1-Q2 ──▶ K1,V1 (group 1)
Q2-K2,V2                  Q2─┼──▶ K,V (SHARED)        Q3-Q4 ──▶ K2,V2 (group 2)
Q3-K3,V3                  Q3─┤     by ALL heads       Q5-Q6 ──▶ K3,V3 (group 3)
Q4-K4,V4                  Q4─┘                        Q7-Q8 ──▶ K4,V4 (group 4)
(full KV cache size)      (KV cache / n_heads)        (KV cache / group_size -- tunable middle ground)
```

**KV-cache growing across generation steps (memory bandwidth, not compute, is the bottleneck):**
```
Step 1: cache = [K1,V1]                       -- process 1 new token, load model weights + tiny cache
Step 2: cache = [K1,V1, K2,V2]                 -- process 1 new token, load model weights + slightly bigger cache
Step 3: cache = [K1,V1, K2,V2, K3,V3]          -- ... and so on, cache GROWS every step
   (model weights must be reloaded from memory EVERY step regardless -- this dominates latency)
```

---

## 9. Practical Examples

**Simple:** implement RMSNorm and verify its output has unit root-mean-square across the feature dimension.
**Medium:** implement grouped-query attention (Section 5) and verify it reduces to standard multi-head attention when `n_kv_groups == n_query_heads`, and to multi-query attention when `n_kv_groups == 1`.
**Real-world:** compute the KV-cache memory footprint (Section 6) for a specific open-weight model's published architecture config (e.g., look up a Mistral or Llama model's `num_attention_heads`/`num_key_value_heads`/`hidden_size` from its Hugging Face config file) and quantify the actual GQA memory savings for that real model.

---

## 10. Real Industry Use Cases

- **Llama 2 (70B) and Llama 3, Mistral, and most modern open-weight LLMs**: use GQA specifically to make large-context, high-throughput serving economically feasible.
- **vLLM and other production inference servers** (Phase 8-adjacent): implement highly optimized KV-cache management (PagedAttention specifically manages KV-cache memory similarly to how an OS manages virtual memory pages, Phase 1 Lesson 6) to serve many concurrent requests efficiently.
- **FlashAttention and its successors**: fuse the attention computation (including causal masking, Phase 5 Lesson 6) into a single, highly memory-efficient GPU kernel, directly addressing the memory-bandwidth-bound nature of attention computation described in Section 3.
- **RMSNorm and SwiGLU**: standard components in essentially every major 2024-2026 open-weight LLM's published architecture.

---

## 11. Common Mistakes

- Forgetting to implement/use KV-caching when building a custom autoregressive generation loop — leads to needlessly, severely slow (quadratically-scaling-per-token) generation.
- Confusing FLOPs-bound and memory-bandwidth-bound bottlenecks — assuming a faster GPU (more FLOPs) will proportionally speed up single-sequence autoregressive generation, when the actual bottleneck is often memory bandwidth (loading weights/KV-cache), not raw compute throughput.
- Choosing an overly aggressive GQA grouping (very few KV groups) without empirically validating the quality tradeoff — the "sweet spot" is architecture/task-dependent, not a universal constant.
- Assuming RMSNorm and LayerNorm are interchangeable in an already-pretrained model — they have different learned parameters and slightly different behavior; you cannot simply swap one for the other in an existing checkpoint without retraining/fine-tuning.

---

## 12. Best Practices (2026)

- Always use KV-caching for any production autoregressive generation implementation — a foundational, non-optional optimization at this point, either via framework support or a properly implemented custom cache.
- Prefer GQA over full MHA for any new large-scale LLM architecture, choosing the grouping factor based on empirical quality/memory tradeoff experiments at your target scale.
- Use RMSNorm and SwiGLU as sensible modern defaults for new Transformer architectures, given their now well-established track record.
- When evaluating LLM serving infrastructure (Phase 8), understand whether the deployment is compute-bound or memory-bandwidth-bound for your specific workload (long-context single requests vs. many concurrent short requests behave very differently) before choosing optimization strategies.

---

## 13. Exercises

**Easy:** Implement RMSNorm and compare its output numerically against a standard LayerNorm implementation on the same input, noting the difference (RMSNorm doesn't subtract the mean).
**Medium:** Implement the from-scratch `KVCache` (Section 5) and integrate it into the Phase 5 Lesson 7 `MiniTransformer`'s generation loop, verifying identical output to the uncached version but with less redundant computation.
**Hard:** Implement Grouped-Query Attention (Section 5) with a configurable number of groups and empirically measure the KV-cache memory savings (Section 6) across several different grouping factors for a fixed model size.
**Mathematical:** Derive the KV-cache memory footprint formula (Section 3) from first principles, given a model's layer count, head count, head dimension, sequence length, and numerical precision.
**Coding:** Implement SwiGLU and compare its parameter count against a standard ReLU-based feedforward layer of comparable total parameter budget (accounting for SwiGLU's extra weight matrix by appropriately narrowing its hidden dimension).

---

## 14. Mini Project

**Upgrade the Phase 5 Lesson 7 `MiniTransformer` to a modern (2026-style) architecture**: replace LayerNorm with RMSNorm, replace the GELU feedforward with SwiGLU, replace standard multi-head attention with Grouped-Query Attention, add a proper KV-cache to the generation loop, and retrain/regenerate text with this upgraded architecture — quantifying (via the Section 6 memory-footprint calculation) the KV-cache savings versus the original architecture, and qualitatively comparing generation speed and output quality before and after these modernizations.

---

## 15. Interview Preparation

- Explain the KV-cache and why it's essential for efficient autoregressive generation.
- What is Grouped-Query Attention, and what tradeoff does it navigate between Multi-Head and Multi-Query Attention?
- Why is autoregressive LLM inference typically memory-bandwidth-bound rather than compute-bound?
- What are RMSNorm and SwiGLU, and why have they become standard in modern LLM architectures over the original Transformer's LayerNorm and ReLU/GELU feedforward?

---

## 16. Summary

This lesson's architectural refinements — KV-caching, Grouped-Query Attention, RMSNorm, and SwiGLU — are direct, measured engineering responses to the specific memory-bandwidth and compute bottlenecks exposed when running the original 2017 Transformer architecture at real production LLM scale. KV-caching eliminates redundant recomputation during generation; GQA shrinks the resulting cache's memory footprint by a quantifiable, tunable factor; RMSNorm and SwiGLU offer modest efficiency gains with no quality cost. Together, these refinements define the architecture used by essentially every major open-weight LLM (Llama, Mistral, and others) in 2026 — direct, practical extensions of Phase 5's foundational Transformer that every subsequent lesson in this phase (pretraining, fine-tuning, quantization) will assume as the baseline architecture.

---

## 17. References

- Shazeer, N. — "Fast Transformer Decoding: One Write-Head is All You Need" (2019, Multi-Query Attention)
- Ainslie et al. — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- Zhang, B. & Sennrich, R. — "Root Mean Square Layer Normalization" (2019, RMSNorm)
- Shazeer, N. — "GLU Variants Improve Transformer" (2020, SwiGLU)
- Kwon et al. — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023, the vLLM paper)
