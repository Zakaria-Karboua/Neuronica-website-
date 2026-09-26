# Phase 6 · Lesson 3 — Positional Encoding

> Prerequisite: Embeddings (Lesson 2), Phase 5 Lessons 6–7 (Attention, Transformers)

---

## 1. Introduction

### What is positional encoding?
A mechanism for injecting information about token *order* into a Transformer, since self-attention (Phase 5 Lesson 6) is fundamentally **permutation-invariant** — it computes relevance based purely on content (query-key similarity), with no inherent notion of which token came first. Without positional encoding, "the cat sat on the mat" and "mat the on sat cat the" would produce identical attention outputs — a real, foundational gap that must be explicitly addressed.

### Why does it exist?
RNNs (Phase 5 Lesson 4) have order built into their architecture by construction — they process tokens sequentially, one at a time. Transformers deliberately discard this sequential processing for parallelizability (Phase 5 Lesson 7), which means order must be reintroduced through some *other* mechanism — positional encoding is exactly that mechanism, added to (or otherwise combined with) token embeddings before the first Transformer layer.

### Historical background
The original Transformer (Vaswani et al., 2017) used fixed sinusoidal positional encodings. BERT (2018) and early GPT models used simple learned positional embeddings instead. Both approaches struggled to generalize to sequence lengths longer than seen during training — a genuinely important practical limitation directly motivating the 2020s development of Rotary Positional Embeddings (RoPE, Su et al., 2021) and ALiBi (Press et al., 2021), which better generalize to longer contexts and are now the standard in most 2026 frontier LLMs.

### Real-world motivation
"Context length" — a headline feature of every LLM release — is directly, mechanically determined by positional encoding design choices. Understanding why some models handle 128K+ token contexts gracefully while older architectures degrade sharply beyond their training length requires understanding exactly this lesson's content.

---

## 2. Theory

### Absolute vs. relative positional encoding
- **Absolute**: each position $t$ gets its own fixed encoding, added directly to the token embedding at that position — simple, but the model must learn relationships between positions from scratch for every absolute position pair.
- **Relative**: encodes the *relative distance* between two positions directly into the attention computation itself — often generalizes better, since "how far apart are these two tokens" is arguably the more fundamentally useful and directly reusable signal (a relationship the model has seen at *many* different absolute position pairs during training, unlike a specific absolute position it may have rarely or never seen at inference time if the sequence is longer than training data).

### Sinusoidal positional encoding (the original Transformer's approach)
Fixed (non-learned) functions of position, using sine/cosine at varying frequencies — chosen specifically so that the encoding for position $t+k$ can be expressed as a *linear function* of the encoding for position $t$ (a deliberate mathematical property intended to help the model learn relative-position relationships even from an absolute-position encoding scheme).

### Rotary Positional Embeddings (RoPE) — the 2026 standard
Rather than *adding* a positional vector to the embedding, RoPE **rotates** the query and key vectors (before the attention dot product) by an angle proportional to their position — the key insight being that the dot product of two *rotated* vectors depends only on their *relative* rotation (i.e., their relative position), not their absolute positions individually, elegantly building relative-position-awareness directly into the attention mechanism's mathematics rather than bolting it on as an additive embedding.

### ALiBi (Attention with Linear Biases) — an even simpler alternative
Instead of modifying the query/key vectors at all, ALiBi simply **subtracts a penalty proportional to distance** directly from the raw attention scores before softmax — closer tokens get less penalty (more attention), farther tokens get more penalty (less attention) — a strikingly simple, parameter-free mechanism that has demonstrated excellent length-generalization properties (performing well on sequences much longer than seen during training).

---

## 3. Mathematical Foundations

### Sinusoidal encoding, the exact formula

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Different dimensions $i$ use different frequencies (from very fast-varying to very slow-varying as $i$ increases) — this multi-frequency design is directly analogous to binary representation (different "digits" capturing position information at different granularities), giving the model a rich, multi-scale signal about position.

### Why this encoding supports relative-position linearity
Using the angle-addition trigonometric identities:

$$
\sin(a+b) = \sin a\cos b + \cos a\sin b, \qquad \cos(a+b) = \cos a\cos b - \sin a\sin b
$$

$PE_{pos+k}$ can be written as a linear combination of $PE_{pos}$'s sine/cosine components (with coefficients depending only on $k$, not on $pos$ itself) — meaning, in principle, a linear transformation exists that maps any position's encoding to any other position offset by a fixed $k$, giving the network a mathematically clean pathway to learn relative-position-sensitive behavior even from these absolute encodings.

### RoPE, formalized
For a 2D subspace of the embedding (RoPE is applied pairwise across the embedding dimension), rotate a query/key vector's pair of coordinates $(x_1, x_2)$ at position $pos$ by angle $\theta_{pos} = pos \cdot \omega$ (for some frequency $\omega$, varying across dimension pairs analogous to the sinusoidal encoding's multi-frequency design):

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos\theta_{pos} & -\sin\theta_{pos} \\ \sin\theta_{pos} & \cos\theta_{pos} \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

The key mathematical property: for a query at position $m$ and a key at position $n$, their dot product after rotation depends only on $\theta_m - \theta_n$ — i.e., only on the *relative* position $m-n$, not on $m$ and $n$ individually. This is a direct, provable relative-position-encoding property, in contrast to sinusoidal encoding's weaker (linear-transformation-*exists* but isn't explicitly built into the attention computation itself) relative-position support.

### ALiBi's attention score modification

$$
\text{attention\_score}(i,j) = q_i \cdot k_j - m\cdot|i-j|
$$

where $m$ is a fixed, head-specific slope (different attention heads use different penalty steepnesses, letting some heads focus more locally and others more globally) — remarkably simple, requires no additional learned parameters for the positional mechanism itself, and has empirically demonstrated strong extrapolation to sequence lengths well beyond training data.

---

## 4. Algorithm — Applying RoPE to Query/Key Vectors (conceptual)

```
GIVEN query/key vectors of dimension d (d assumed even, processed in pairs), position pos:
FOR each pair of dimensions (2i, 2i+1) in the vector:
    theta_i = pos / (10000 ^ (2i/d))                 # position- AND dimension-pair-dependent angle
    x_2i_new     = x_2i * cos(theta_i) - x_2i+1 * sin(theta_i)
    x_2i+1_new   = x_2i * sin(theta_i) + x_2i+1 * cos(theta_i)
RETURN the rotated vector, used in place of the original query/key in the attention dot product
```
Applied to both queries and keys before computing $QK^T$ (Phase 5 Lesson 6), this single rotation operation is all that's needed to inject relative positional awareness directly into the attention scores — no separate positional embedding vector needs to be added to the token embeddings at all.

---

## 5. Python Implementation

```python
"""positional_encoding_core.py — sinusoidal, RoPE, and ALiBi implementations"""
import numpy as np


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(max_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** (2 * (dims // 2) / d_model))
    angles = positions * angle_rates
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotates pairs of dimensions -- the core RoPE operation."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """x: (seq_len, d). Applies rotary positional embedding in-place, conceptually."""
    d = x.shape[-1]
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
    angles = positions[:, None] * freqs[None, :]            # (seq_len, d/2)
    cos = np.repeat(np.cos(angles), 2, axis=-1)
    sin = np.repeat(np.sin(angles), 2, axis=-1)
    return x * cos + rotate_half(x) * sin


def alibi_bias(seq_len: int, n_heads: int) -> np.ndarray:
    """Returns (n_heads, seq_len, seq_len) bias matrix to ADD to raw attention scores before softmax."""
    slopes = np.array([2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)])  # geometric slope sequence
    positions = np.arange(seq_len)
    distance = np.abs(positions[:, None] - positions[None, :])                  # (seq_len, seq_len)
    return -slopes[:, None, None] * distance[None, :, :]                        # penalty grows with distance


if __name__ == "__main__":
    pe = sinusoidal_positional_encoding(max_len=10, d_model=8)
    print("Sinusoidal PE shape:", pe.shape)

    rng = np.random.default_rng(0)
    q = rng.normal(size=(6, 8))
    positions = np.arange(6)
    q_rotated = apply_rope(q, positions)
    print("RoPE-applied query shape:", q_rotated.shape)

    bias = alibi_bias(seq_len=6, n_heads=4)
    print("ALiBi bias shape:", bias.shape, "-- example row (head 0, query pos 5):", bias[0, 5].round(2))
```

---

## 6. Build From Scratch

**Verifying RoPE's key mathematical property (relative-position-only dependence) empirically:**
```python
import numpy as np

def verify_rope_relative_property(d: int = 8):
    """Confirms: dot product of rotated (q at pos m, k at pos n) depends ONLY on (m-n), not on m,n individually."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=d)
    k = rng.normal(size=d)

    results = {}
    for m, n in [(5, 3), (10, 8), (100, 98), (2, 0)]:   # all these pairs have the SAME relative distance: 2
        q_rot = apply_rope(q[None, :], np.array([m]))[0]
        k_rot = apply_rope(k[None, :], np.array([n]))[0]
        results[(m, n)] = q_rot @ k_rot

    print("Dot products for pairs with relative distance = 2:", results)
    # All four values should be (numerically) IDENTICAL, confirming RoPE's relative-position property
```
Running this reveals all four dot products are equal (up to floating-point precision) — a directly verifiable confirmation of Section 3's mathematical claim, not just an assertion to take on faith.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `sinusoidal_positional_encoding` | Original Transformer implementations; largely superseded in modern LLMs |
| `apply_rope` | Hugging Face Transformers' built-in RoPE implementations (used in Llama, Mistral, and most 2024-2026 open-weight models) — highly optimized, integrated with efficient attention kernels |
| `alibi_bias` | Used in models like BLOOM and MPT; increasingly seen alongside or compared against RoPE for long-context extrapolation |

---

## 8. Visual Explanations

**Why permutation invariance is a problem without positional encoding:**
```
"the cat sat"  -- self-attention computes relevance based on CONTENT only
"sat cat the"  -- IDENTICAL attention pattern without positional info (attention doesn't "see" order)
   (positional encoding breaks this symmetry, letting the model distinguish word ORDER)
```

**RoPE's rotation (relative position encoded via rotation angle difference):**
```
Query at position m:  rotated by angle m*ω
Key at position n:    rotated by angle n*ω
Dot product after rotation depends on (m*ω - n*ω) = (m-n)*ω  -- ONLY the relative offset survives
```

**ALiBi's distance penalty (closer = less penalty = more attention):**
```
Query position 10, attending to keys at positions 0..10:
Position:   0    2    4    6    8   10
Penalty:  -10   -8   -6   -4   -2    0   (linearly increasing penalty with distance, SUBTRACTED before softmax)
             (far positions get heavily penalized -> naturally attend LESS to distant tokens by default)
```

---

## 9. Practical Examples

**Simple:** compute and plot sinusoidal positional encodings for a short sequence, visually confirming different dimensions vary at different frequencies.
**Medium:** apply RoPE to a query and key vector and empirically verify (Section 6) that only relative position affects their dot product.
**Real-world:** investigate a specific open-weight LLM's positional encoding choice (e.g., check a Llama model's config for RoPE parameters) and relate its stated maximum context length to its positional encoding design — directly connecting this lesson's theory to a real model's documented specifications.

---

## 10. Real Industry Use Cases

- **Llama, Mistral, and most modern open-weight LLMs** (2024-2026): use RoPE as their standard positional encoding.
- **Long-context model research**: "context length extension" techniques (e.g., NTK-aware scaling, YaRN) are specifically modifications to RoPE's frequency parameters, allowing models trained on shorter sequences to be adapted for much longer inference-time contexts — directly relevant to understanding how a "128K context" model announcement is technically achieved.
- **BLOOM, MPT, and some other models**: use ALiBi, valued for its simplicity and strong length-generalization properties.
- **Original BERT/GPT-2**: used simpler learned or sinusoidal absolute positional embeddings — now understood to generalize less gracefully to longer contexts, a key motivation for RoPE/ALiBi's subsequent adoption.

---

## 11. Common Mistakes

- Assuming any positional encoding scheme extrapolates gracefully to sequence lengths far beyond training data without any special handling — even RoPE, while better than naive learned absolute embeddings, benefits from specific scaling techniques (NTK-aware scaling, position interpolation) for genuinely long-context extrapolation.
- Forgetting that positional information must be present at *every* layer's attention computation in RoPE-style schemes (since it's baked into the rotation applied at each layer) versus additive schemes where it's typically added once, at the input embedding layer only.
- Confusing "context length" (how many tokens a model was trained/configured to handle) with "attention span" (how effectively a model actually uses distant context, which can degrade well before the nominal maximum length is reached) — a genuine, empirically documented distinction.
- Implementing RoPE incorrectly (e.g., rotating the wrong dimension pairs, or using an inconsistent frequency base) — silently breaks the elegant relative-position property without necessarily causing an obvious error, only degraded downstream performance.

---

## 12. Best Practices (2026)

- Default to RoPE for any new Transformer architecture — the dominant, well-validated standard for modern LLMs.
- When working with existing pretrained models, use their exact positional encoding configuration (don't casually swap encoding schemes) — architectural mismatches here silently and severely degrade performance.
- For long-context applications, understand whether the target model uses any specific context-extension technique (position interpolation, NTK-aware scaling) and be aware that effective performance may still degrade gracefully-but-really at the very edges of the claimed maximum context length.
- Use the verification technique from Section 6 (checking relative-position-only dependence) as a genuine sanity check if implementing or modifying any custom positional encoding scheme.

---

## 13. Exercises

**Easy:** Compute sinusoidal positional encodings for a 20-position sequence and plot several dimensions to visually confirm the multi-frequency design.
**Medium:** Implement RoPE (Section 5) and verify the relative-position property (Section 6) holds for several different position pairs.
**Hard:** Implement ALiBi's bias matrix and integrate it into the Phase 5 Lesson 6 scaled dot-product attention implementation, verifying the resulting attention weights favor nearby positions more than an equivalent unbiased attention computation.
**Mathematical:** Using the angle-addition trigonometric identities, derive explicitly why sinusoidal positional encoding supports a linear relationship between $PE_{pos}$ and $PE_{pos+k}$.
**Coding:** Implement a simple position-interpolation technique (compressing position indices to fit within a model's original training range when processing longer sequences) and empirically compare its effect on RoPE's relative-position dot products versus using raw, uncompressed positions beyond the training range.

---

## 14. Mini Project

**Extend the Phase 5 Lesson 7 `MiniTransformer` with RoPE**: replace its simple learned positional embeddings with the RoPE implementation from this lesson (Section 5), verify the relative-position property holds within your model's actual query/key computations (not just in isolation), retrain the tiny character-level language model, and qualitatively/quantitatively compare its behavior (and if feasible, its ability to generalize to sequences longer than seen during training) against the original learned-positional-embedding version.

---

## 15. Interview Preparation

- Why is positional encoding necessary for Transformers but not for RNNs?
- Explain the key mathematical property that makes RoPE encode *relative* rather than absolute position.
- What is ALiBi, and how does its approach to positional information differ fundamentally from RoPE's?
- Why do models trained with certain positional encoding schemes struggle to generalize to sequences longer than their training length?

---

## 16. Summary

Positional encoding solves self-attention's fundamental permutation-invariance limitation by injecting order information — sinusoidal encoding (the original, additive approach) offers a mathematically elegant but only loosely-enforced relative-position property, while RoPE (the 2026 standard) builds relative-position-dependence directly and provably into the attention dot product via rotation, and ALiBi achieves strong length-generalization through a strikingly simple distance-based penalty on raw attention scores. Every modern LLM's advertised "context length" is directly, mechanically shaped by exactly these architectural choices — understanding them demystifies both why context windows have specific limits and how techniques like position interpolation extend them.

---

## 17. References

- Vaswani et al. — "Attention Is All You Need" (2017, original sinusoidal positional encoding)
- Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021, the original RoPE paper)
- Press, Smith, Lewis — "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (2021, the original ALiBi paper)
- Peng et al. — "YaRN: Efficient Context Window Extension of Large Language Models" (2023, a widely-used RoPE-scaling technique for long-context extension)
