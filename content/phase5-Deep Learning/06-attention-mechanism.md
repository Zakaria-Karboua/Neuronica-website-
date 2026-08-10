# Phase 5 · Lesson 6 — Attention Mechanism

> Prerequisite: RNNs, LSTMs & GRUs (Lessons 4–5) — this lesson is the direct bridge to Transformers (Lesson 7)

---

## 1. Introduction

### What is attention?
A mechanism allowing a model to dynamically compute, for each element of its output, a **weighted combination of all elements of the input**, with the weights themselves learned and *data-dependent* (different for every input) rather than fixed or purely positional. It directly solves RNN/LSTM's fundamental limitation: even with gating, information from distant time steps must still pass sequentially through every intermediate hidden state; attention gives *direct*, unmediated access to every input position, regardless of distance.

### Why does it exist?
Sequence-to-sequence RNN/LSTM models (Lesson 4) compress an entire input sequence into a single fixed-size final hidden state (the "encoder bottleneck") before the decoder even begins generating output — for long sequences, this bottleneck loses information regardless of how well LSTMs mitigate vanishing gradients (Lesson 5). Attention (Bahdanau et al., 2014) was introduced specifically to let the decoder look back at *all* encoder hidden states directly, weighted by relevance to what it's currently generating, eliminating the fixed-bottleneck problem entirely.

### Historical background
Bahdanau attention (2014) was introduced as an add-on to RNN-based machine translation. Luong et al. (2015) simplified and generalized the formulation. The pivotal moment was Vaswani et al.'s "Attention Is All You Need" (2017), which showed attention alone — with **no recurrence at all** — could outperform RNN/LSTM-based architectures while being dramatically more parallelizable, directly launching the Transformer architecture (Lesson 7) that underlies virtually every modern LLM (Phase 6).

### Real-world motivation
Every LLM you'll study in Phase 6 — every GPT, Claude, Llama-family model — is built from stacked attention layers. This lesson derives the exact mathematics (queries, keys, values, scaled dot-product attention) that Phase 6 will assume as known background.

---

## 2. Theory

### The query-key-value (QKV) framework
Attention reframes "which parts of the input are relevant to what I'm computing right now" using an information-retrieval analogy:
- **Query** ($Q$): "what am I looking for?" — derived from the current decoding position/element.
- **Key** ($K$): "what do I contain?" — derived from each input position, used to compute relevance.
- **Value** ($V$): "what do I actually contribute?" — derived from each input position, this is what actually gets combined once relevance is determined.

Relevance is computed as a similarity between the query and each key (commonly a dot product); these similarities are converted to weights via softmax (Phase 5 Lesson 1), and the output is the weighted sum of values using those weights.

### Self-attention vs. cross-attention
- **Cross-attention**: queries come from one sequence (e.g., the decoder), keys/values from another (e.g., the encoder) — the original Bahdanau/Luong translation use case.
- **Self-attention**: queries, keys, AND values all come from the *same* sequence — each element attends to every other element (including itself) within one sequence, the mechanism that makes Transformers (Lesson 7) work without any recurrence at all.

### Multi-head attention
Rather than computing attention once, split $Q, K, V$ into multiple lower-dimensional "heads," compute attention independently in each, and concatenate the results — allowing different heads to specialize in capturing different types of relationships (e.g., one head might learn syntactic dependencies, another semantic similarity) simultaneously.

---

## 3. Mathematical Foundations

### Scaled dot-product attention, the exact formula
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$ ($n$ = number of query positions, $m$ = number of key/value positions, $d_k$ = key/query dimension). $QK^T$ computes all pairwise dot-product similarities at once — pure linear algebra (Phase 3 Lesson 1), producing an $n\times m$ matrix of raw relevance scores.

### Why the $\sqrt{d_k}$ scaling factor (a subtle but essential detail)
For query/key vectors with independent, zero-mean, unit-variance components, the dot product $q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ has variance $d_k$ (sum of $d_k$ independent unit-variance terms). As $d_k$ grows, the dot products' magnitude grows with it — pushed into softmax's extreme, near-saturated regions, where gradients vanish (an issue directly connecting back to Phase 5 Lesson 1's saturating-activation gradient concerns). Dividing by $\sqrt{d_k}$ exactly renormalizes the variance back to 1 regardless of dimensionality, keeping softmax's input in a well-behaved, gradient-friendly range — a precise, derivable fix, not an arbitrary hyperparameter.

### Multi-head attention, formalized
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O, \qquad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
Each head projects $Q,K,V$ into a lower-dimensional subspace ($d_{model}/h$ dimensions per head, for $h$ heads and total model dimension $d_{model}$) — the total computational cost is roughly the same as one full-dimensional attention computation, but the model gains the ability to attend to different "types" of relationships in each subspace simultaneously, an empirically crucial architectural choice.

### Computational complexity — attention's central tradeoff
Computing $QK^T$ for a sequence of length $n$ costs $O(n^2 \cdot d_k)$ — **quadratic in sequence length**. This is attention's fundamental scaling limitation (directly relevant to Phase 6's discussion of long-context LLMs and the various efficient-attention approximations developed to address it), in exchange for: full parallelizability across sequence positions (unlike RNN's inherently sequential $O(n)$ *serial* dependency, Lesson 4) and direct, unmediated access between any two positions regardless of distance (unlike RNN/LSTM's information having to pass through every intermediate hidden state).

---

## 4. Algorithm — Scaled Dot-Product Attention (fully specified)

```
GIVEN Q (n x d_k), K (m x d_k), V (m x d_v):
1. COMPUTE raw similarity scores: S = Q @ K.T                    # (n x m), O(n*m*d_k)
2. SCALE: S_scaled = S / sqrt(d_k)                                # variance-normalization (Section 3)
3. (OPTIONAL, for causal/decoder self-attention) MASK: 
     set S_scaled[i,j] = -infinity for all j > i                 # prevents "looking into the future"
4. NORMALIZE: A = softmax(S_scaled, axis=-1)                       # each row sums to 1 -- a valid weighting
5. COMPUTE output: output = A @ V                                  # (n x d_v), weighted combination of values
RETURN output, and OPTIONALLY the attention weights A (for interpretability/visualization)
```
The causal mask (step 3) is specifically what makes autoregressive language model training possible: when predicting token $t$, the model must not be allowed to attend to tokens $t+1, t+2, \dots$ (which wouldn't exist yet at real generation time) — setting those positions' scores to $-\infty$ before softmax makes their attention weight exactly zero.

---

## 5. Python Implementation

```python
"""attention_core.py — scaled dot-product and multi-head attention, from scratch"""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal_mask: bool = False):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                          # Section 3's scaling

    if causal_mask:
        n, m = scores.shape
        mask = np.triu(np.ones((n, m)), k=1).astype(bool)     # upper triangle = "future" positions
        scores = np.where(mask, -np.inf, scores)

    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V
    return output, attn_weights


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W_Q = rng.normal(0, scale, (d_model, d_model))
        self.W_K = rng.normal(0, scale, (d_model, d_model))
        self.W_V = rng.normal(0, scale, (d_model, d_model))
        self.W_O = rng.normal(0, scale, (d_model, d_model))

    def forward(self, X: np.ndarray, causal_mask: bool = False) -> np.ndarray:
        n, d_model = X.shape
        Q, K, V = X @ self.W_Q, X @ self.W_K, X @ self.W_V

        # Split into heads: (n, d_model) -> (n_heads, n, d_head)
        def split_heads(M):
            return M.reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        head_outputs = []
        for h in range(self.n_heads):
            out, _ = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], causal_mask=causal_mask)
            head_outputs.append(out)

        concatenated = np.concatenate(head_outputs, axis=-1)   # (n, d_model)
        return concatenated @ self.W_O


# Example: self-attention over a sequence of 6 tokens, d_model=8, 2 heads
X = np.random.default_rng(0).normal(size=(6, 8))
mha = MultiHeadAttention(d_model=8, n_heads=2)
output = mha.forward(X, causal_mask=True)   # causal: appropriate for a decoder/autoregressive LM
print("Output shape:", output.shape)   # (6, 8) -- same shape as input, ready to feed the next layer
```

---

## 6. Build From Scratch

**Visualizing attention weights (to build direct intuition for what attention actually learns to do):**
```python
import numpy as np

def visualize_attention(attn_weights: np.ndarray, tokens: list[str]) -> None:
    """Prints a simple text-based heatmap of attention weights -- which tokens attend to which."""
    print("      " + " ".join(f"{t:>6}" for t in tokens))
    for i, row in enumerate(attn_weights):
        weights_str = " ".join(f"{w:6.2f}" for w in row)
        print(f"{tokens[i]:>6}{weights_str}")

# A toy example: "the cat sat on the mat" -- self-attention weights (illustrative, not learned)
tokens = ["the", "cat", "sat", "on", "the", "mat"]
Q = K = V = np.random.default_rng(1).normal(size=(6, 4))
output, attn_weights = scaled_dot_product_attention(Q, K, V)
visualize_attention(attn_weights, tokens)
```
Running this on a real *trained* attention layer (rather than random weights, as shown here for mechanical illustration) typically reveals interpretable patterns — e.g., a pronoun attending strongly to the noun it refers to, or adjacent words attending to each other for local syntax — a well-documented phenomenon that makes attention weights a (partial, imperfect) window into model behavior, occasionally used for interpretability research.

---

## 7. Library/Tool Comparison

| From scratch | PyTorch |
|---|---|
| `scaled_dot_product_attention` | `torch.nn.functional.scaled_dot_product_attention` — highly optimized (FlashAttention-style fused kernels on GPU, dramatically reducing memory usage for long sequences) |
| `MultiHeadAttention` | `torch.nn.MultiheadAttention` — production-grade, handles batching, padding masks, and both self/cross-attention configurations |
| Manual causal masking | Built-in `is_causal=True` flag in PyTorch's attention functions, or explicit mask tensors |

---

## 8. Visual Explanations

**QKV attention as a "soft dictionary lookup":**
```
Query: "what am I looking for?"
   │
   ▼ (compare against every Key)
Keys:   [k1] [k2] [k3] [k4]   -- similarity scores via dot product
   │
   ▼ softmax -> weights [0.1, 0.6, 0.2, 0.1]  (these sum to 1)
Values: [v1] [v2] [v3] [v4]
   │
   ▼ weighted sum: 0.1*v1 + 0.6*v2 + 0.2*v3 + 0.1*v4  = OUTPUT
   (unlike a HARD dictionary lookup, this is a SOFT, differentiable blend of ALL values,
    weighted by learned relevance -- fully compatible with backpropagation)
```

**Causal masking (preventing a decoder from "seeing the future"):**
```
Attention scores BEFORE masking:      AFTER causal masking (upper triangle = -infinity):
     tok1 tok2 tok3 tok4                  tok1  tok2  tok3  tok4
tok1  0.5  0.3  0.1  0.4              tok1  0.5  -inf  -inf  -inf
tok2  0.2  0.6  0.3  0.2       -->    tok2  0.2   0.6  -inf  -inf
tok3  0.1  0.4  0.7  0.3              tok3  0.1   0.4   0.7  -inf
tok4  0.3  0.2  0.5  0.6              tok4  0.3   0.2   0.5   0.6
  (after softmax, -infinity entries become EXACTLY 0 -- token t cannot attend to any t' > t)
```

---

## 9. Practical Examples

**Simple:** compute scaled dot-product attention by hand for a tiny 3-token sequence and verify the output shape and that attention weights sum to 1 per row.
**Medium:** implement multi-head attention (Section 5) and verify that splitting into more heads (while keeping `d_model` fixed) changes the per-head dimensionality but not the total parameter count of the QKV projection matrices.
**Real-world:** implement causal self-attention (Section 5, `causal_mask=True`) and use it as the core building block for a tiny autoregressive character-level language model, generating text one character at a time — direct hands-on preparation for Phase 6's full LLM pretraining content.

---

## 10. Real Industry Use Cases

- **Every modern LLM** (GPT-family, Claude, Llama, Gemini): built from stacked multi-head self-attention layers exactly as derived in this lesson — Phase 6 will show the full Transformer block assembling this with feedforward layers and normalization.
- **Machine translation** (Google Translate's post-2016 architecture, and essentially all modern translation systems): cross-attention between encoder and decoder remains conceptually central even in fully-transformer-based systems.
- **Vision Transformers (ViT)**: apply the exact same self-attention mechanism to image patches instead of text tokens, demonstrating attention's generality beyond sequential/text data.
- **Interpretability research**: attention weight visualization (Section 6) is a standard, if imperfect and sometimes misleading, tool for probing what a trained model is "focusing on."

---

## 11. Common Mistakes

- Forgetting the $\sqrt{d_k}$ scaling factor — causes softmax saturation and vanishing gradients in high-dimensional attention, especially noticeable as model size (and thus $d_k$) grows.
- Implementing causal masking incorrectly (off-by-one errors in the triangular mask) — silently allows a decoder to "cheat" by attending to future tokens during training, producing a model that performs deceptively well in training/validation but fails at real autoregressive generation time.
- Treating attention weights as a definitive, complete explanation of model reasoning — research has shown attention weights don't always faithfully reflect what information the model actually uses (a genuine, ongoing interpretability research debate, not settled fact).
- Not accounting for attention's $O(n^2)$ sequence-length scaling when designing systems for very long contexts — a real, practical engineering constraint addressed by various efficient-attention techniques (sparse attention, linear attention approximations) in modern large-context LLMs.

---

## 12. Best Practices (2026)

- Use framework-provided fused attention implementations (`torch.nn.functional.scaled_dot_product_attention`, which automatically dispatches to FlashAttention-style kernels when available) rather than a naive from-scratch implementation for any real training/inference workload — dramatically better memory efficiency and speed.
- Always apply causal masking correctly for autoregressive (decoder-only) language modeling, and verify it via a targeted unit test (e.g., confirming a change to a "future" token doesn't affect an earlier token's output).
- Be appropriately skeptical of attention-weight-based interpretability claims — treat them as suggestive, not definitive, evidence of model behavior.
- Understand the $O(n^2)$ complexity tradeoff when working with long-context applications (Phase 7's RAG systems, in particular) — this directly motivates chunking/retrieval strategies rather than naively feeding arbitrarily long contexts.

---

## 13. Exercises

**Easy:** Compute scaled dot-product attention by hand (on paper or in code) for a 2-token sequence with small, simple numbers, and verify against the Section 5 implementation.
**Medium:** Implement causal masking and write a unit test confirming that changing a future token's input value doesn't change any earlier token's attention output.
**Hard:** Implement multi-head attention from scratch (Section 5) and empirically verify that increasing the number of heads (while holding `d_model` fixed) changes the learned attention patterns' diversity — visualize a few heads' attention weight matrices side by side after training a small model.
**Mathematical:** Derive the variance of $q\cdot k$ for $d_k$-dimensional vectors with i.i.d. zero-mean, unit-variance components, confirming it equals $d_k$ and justifying the $\sqrt{d_k}$ scaling factor.
**Coding:** Build a tiny character-level autoregressive language model using only the from-scratch causal self-attention (Section 5) plus a simple feedforward output layer, and generate text by sampling one character at a time.

---

## 14. Mini Project

Build a **tiny GPT-style character-level language model from scratch**: implement causal multi-head self-attention (Section 5), stack 2-3 such layers with residual connections and layer normalization (a preview of Lesson 7's full Transformer block), train it on a small text corpus to predict the next character, and generate sample text via autoregressive sampling — this is, in miniature, exactly the architecture Phase 6 will scale up into a full LLM, making this project the direct hands-on bridge between this lesson and that phase.

---

## 15. Interview Preparation

- Derive the scaled dot-product attention formula and explain why the $\sqrt{d_k}$ scaling factor is necessary.
- Explain the difference between self-attention and cross-attention.
- Why is attention more parallelizable than RNN-based sequence processing?
- What is causal masking, and why is it essential for training autoregressive language models?

---

## 16. Summary

Attention replaces RNN/LSTM's sequential, bottlenecked information flow with a direct, learned, data-dependent weighting over all input positions simultaneously — the query-key-value framework computes relevance via scaled dot products (with the $\sqrt{d_k}$ scaling factor precisely counteracting a specific variance-growth problem), multi-head attention lets the model attend to multiple types of relationships in parallel subspaces, and causal masking makes autoregressive language modeling possible by preventing any position from attending to future positions. This lesson's scaled dot-product attention formula, implemented fully from scratch here, is the single most important mathematical object underlying every LLM in Phase 6 — the Transformer (Lesson 7) is, at its core, exactly this mechanism stacked with feedforward layers and normalization.

---

## 17. References

- Bahdanau, Cho, Bengio — "Neural Machine Translation by Jointly Learning to Align and Translate" (2014, original attention paper)
- Vaswani et al. — "Attention Is All You Need" (2017, the paper that launched Transformers)
- Alammar, J. — "The Illustrated Transformer" (jalammar.github.io, an exceptionally clear visual explanation)
- Dao et al. — "FlashAttention" (2022, the modern, memory-efficient production attention implementation referenced in Section 12)
