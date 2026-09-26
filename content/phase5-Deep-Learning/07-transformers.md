# Phase 5 · Lesson 7 — Transformers

> Prerequisite: Attention Mechanism (Lesson 6), Neural Networks (Lesson 1). This lesson is the direct architectural foundation for all of Phase 6.

---

## 1. Introduction

### What is a Transformer?
A neural network architecture built entirely from stacked self-attention layers (Lesson 6) and position-wise feedforward layers, with residual connections and layer normalization holding everything together — and crucially, **no recurrence and no convolution at all**. Introduced in "Attention Is All You Need" (Vaswani et al., 2017), it is, without qualification, the architecture underlying every major LLM in 2026 (Phase 6) and increasingly dominant in vision (ViT) and other modalities as well.

### Why does it exist?
RNNs/LSTMs (Lessons 4-5) process sequences strictly sequentially — token $t$'s computation must wait for token $t-1$'s hidden state, fundamentally preventing parallelization across the sequence dimension during training, a severe practical bottleneck at the scale (billions of tokens) needed for modern language modeling. Transformers replace recurrence with self-attention (Lesson 6), which computes relationships between all sequence positions **simultaneously** — fully parallelizable, at the cost of attention's $O(n^2)$ sequence-length complexity, a tradeoff that has proven overwhelmingly favorable given modern parallel hardware (GPUs/TPUs).

### Historical background
2017's original Transformer was an encoder-decoder architecture for machine translation. Its components were rapidly specialized: BERT (2018) used only the encoder (bidirectional, for understanding tasks); GPT (2018 onward) used only the decoder (causal/autoregressive, for generation) — the decoder-only architecture that every major LLM (GPT-family, Claude, Llama, Gemini) is now built from, a direct consequence of decoder-only models' natural fit for the "predict the next token" pretraining objective (Phase 6).

### Real-world motivation
Everything in Phase 6 — pretraining, fine-tuning, LoRA, quantization — operates on exactly this architecture. This lesson assembles Lesson 6's attention mechanism into the complete Transformer block, the single most important architectural template in contemporary AI.

---

## 2. Theory

### The Transformer block (decoder-only, the GPT/Claude/Llama-family template)
Each block consists of:
1. **Multi-head causal self-attention** (Lesson 6) — with a residual connection and layer normalization.
2. **Position-wise feedforward network** — a simple 2-layer MLP (Phase 5 Lesson 1) applied identically and independently to each position — with another residual connection and layer normalization.

Stacking many such blocks (e.g., dozens for a large model) builds up increasingly abstract representations, directly analogous to CNN's hierarchical feature learning (Lesson 3), but via attention-based context-mixing instead of spatial convolution.

### Residual connections — why they're essential at Transformer depth

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

The `+ x` (a **residual/skip connection**) provides a direct gradient path around each sublayer — if the sublayer's gradient contribution is small or noisy, the identity path still lets gradients flow, directly mitigating the vanishing gradient problem (Phase 5 Lesson 1's Section 3) that would otherwise cripple training in networks dozens of layers deep — the same idea underlying ResNet's success in very deep CNNs (Lesson 3's reference to He et al., 2015).

### Layer normalization
Unlike batch normalization (normalizing across the batch dimension), layer normalization normalizes across the *feature* dimension, independently for each sequence position — a choice specifically well-suited to variable-length sequences and small/variable batch sizes common in language model training and inference.

### Positional information
Since self-attention itself has no inherent notion of sequence order (attention treats the input as an unordered *set* of positions, computing relevance based purely on content) — **positional encodings** (Phase 6 Lesson 3 covers this in full depth) must be added to give the model any sense of token order at all, a genuinely necessary architectural addition, not an optional refinement.

---

## 3. Mathematical Foundations

### The full Transformer block, equation by equation

$$
X' = \text{LayerNorm}(X + \text{MultiHeadAttention}(X,X,X))
$$


$$
\text{FFN}(x) = W_2 \cdot \text{activation}(W_1x + b_1) + b_2
$$


$$
X'' = \text{LayerNorm}(X' + \text{FFN}(X'))
$$

The feedforward network is applied **position-wise** — the exact same $W_1, W_2$ weights are used at every sequence position independently (no mixing across positions happens in the FFN; that's entirely attention's job) — a clean separation of concerns: attention mixes information *across* positions, the FFN transforms information *within* each position.

### Why "Pre-LN" vs "Post-LN" matters (a real, non-trivial architectural detail)
The original 2017 paper applied LayerNorm *after* the residual addition ("Post-LN," as shown above). Later work found applying LayerNorm *before* each sublayer ("Pre-LN": $X + \text{Sublayer}(\text{LayerNorm}(X))$) produces measurably more stable gradients in very deep Transformers, becoming the standard choice in most modern large-scale LLMs — a concrete example of how seemingly small architectural details (the exact placement of a normalization operation) have real, measurable consequences at scale.

### Computational complexity of a full Transformer block
Self-attention: $O(n^2 \cdot d)$ (Lesson 6); position-wise FFN: $O(n \cdot d^2)$ (since FFN's hidden dimension is typically $4d$, this is $O(n \cdot d^2)$ per block). For typical LLM configurations, $d$ (model dimension, often thousands) tends to dominate over $n$ (sequence length, often thousands too, but growing rapidly in modern "long-context" models) — meaning at extreme context lengths, attention's quadratic term eventually dominates, directly motivating the efficient-attention research area referenced in Lesson 6.

### Parameter count of a full Transformer model
For $L$ layers, model dimension $d$, feedforward dimension $4d$ (standard convention): each block has roughly $4d^2$ (QKVO projections) $+ 8d^2$ (FFN's two $d \times 4d$ matrices) $\approx 12d^2$ parameters, giving a total model parameter count of roughly $12Ld^2$ — the formula practitioners use to reason about the relationship between a model's depth/width and its total parameter count (directly relevant to understanding why "70B parameter model" implies particular $(L,d)$ architectural choices).

---

## 4. Algorithm — Full Decoder-Only Transformer Forward Pass

```
GIVEN token embeddings X (n x d) with positional encodings ALREADY added:
FOR each of L transformer blocks:
    # Sub-layer 1: causal self-attention with residual + norm
    attn_out = MultiHeadCausalSelfAttention(X)
    X = LayerNorm(X + attn_out)          # (or Pre-LN variant: X = X + MultiHeadAttn(LayerNorm(X)))

    # Sub-layer 2: position-wise feedforward with residual + norm
    ffn_out = FeedForward(X)              # applied INDEPENDENTLY to each of the n positions
    X = LayerNorm(X + ffn_out)

FINAL: apply a linear "language modeling head" projecting X (n x d) to (n x vocab_size),
       then softmax to get a probability distribution over the vocabulary at EACH position
       (during training: compare against the true next token via cross-entropy, Phase 3 Lesson 6;
        during generation: sample/argmax from the FINAL position's distribution, Phase 6)
```

---

## 5. Python Implementation

```python
"""transformer_core.py — a complete, small decoder-only Transformer block, from scratch"""
import numpy as np
from attention_core import MultiHeadAttention, softmax   # reusing Lesson 6's implementation


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """Smooth, modern alternative to ReLU -- standard in transformer feedforward layers."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.attn = MultiHeadAttention(d_model, n_heads, seed=seed)
        scale = 0.1
        self.W1 = rng.normal(0, scale, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        self.gamma1 = np.ones(d_model); self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model); self.beta2 = np.zeros(d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Sub-layer 1: causal self-attention + residual + norm (Post-LN, matching the original paper)
        attn_out = self.attn.forward(X, causal_mask=True)
        X = layer_norm(X + attn_out, self.gamma1, self.beta1)

        # Sub-layer 2: position-wise feedforward + residual + norm
        ffn_out = gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2
        X = layer_norm(X + ffn_out, self.gamma2, self.beta2)
        return X


class MiniTransformer:
    """A tiny GPT-style decoder-only model, stacking several TransformerBlocks."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int, seed=0):
        rng = np.random.default_rng(seed)
        self.token_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.pos_embedding = rng.normal(0, 0.02, (max_len, d_model))    # learned positional embeddings (simple choice)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, seed=seed + i) for i in range(n_layers)]
        self.lm_head = rng.normal(0, 0.02, (d_model, vocab_size))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        n = len(token_ids)
        X = self.token_embedding[token_ids] + self.pos_embedding[:n]
        for block in self.blocks:
            X = block.forward(X)
        logits = X @ self.lm_head            # (n, vocab_size) -- a distribution over next-token predictions
        return logits


# Example: tiny character-level model, vocab of 27 (a-z + space)
model = MiniTransformer(vocab_size=27, d_model=32, n_heads=4, d_ff=128, n_layers=2, max_len=50)
token_ids = np.array([0, 4, 12, 8, 14])   # e.g., encoded characters
logits = model.forward(token_ids)
print("Logits shape:", logits.shape)   # (5, 27) -- next-token prediction distribution at EACH position
```

---

## 6. Build From Scratch

Section 5's `MiniTransformer` is already a genuinely complete, from-scratch decoder-only Transformer (built entirely on Lesson 6's from-scratch attention). The natural additional exercise is implementing **greedy/sampled text generation** using this architecture:

```python
import numpy as np

def generate(model: "MiniTransformer", prompt_ids: list[int], max_new_tokens: int = 20,
              temperature: float = 1.0) -> list[int]:
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model.forward(np.array(ids))
        next_token_logits = logits[-1] / temperature      # only the LAST position's prediction matters
        probs = softmax(next_token_logits.reshape(1, -1))[0]
        next_id = np.random.default_rng().choice(len(probs), p=probs)   # SAMPLE (not argmax) for diversity
        ids.append(next_id)
    return ids

generated = generate(model, prompt_ids=[0, 4, 12], max_new_tokens=10, temperature=0.8)
print("Generated token IDs:", generated)
```
This is precisely the autoregressive generation loop every LLM (Phase 6) uses at inference time: feed the sequence so far, take the *last* position's predicted distribution, sample (or greedily argmax) a next token, append it, and repeat — `temperature` controls how "sharp" (low temperature, closer to argmax) or "flat" (high temperature, closer to uniform random) the sampling distribution is, a direct, practically important generation hyperparameter you'll encounter constantly in Phase 6.

---

## 7. Library/Tool Comparison

| From scratch | PyTorch / Hugging Face |
|---|---|
| `TransformerBlock`/`MiniTransformer` | `torch.nn.TransformerDecoderLayer`, or (far more commonly in practice) Hugging Face's pre-built model classes (`GPT2Model`, `LlamaModel`, etc.) — production-grade, pretrained, GPU-optimized |
| Manual `generate` loop | Hugging Face's `.generate()` method — supports beam search, nucleus/top-p sampling, repetition penalties, and many other generation strategies far beyond simple temperature sampling |
| Manual learned positional embeddings | Modern LLMs increasingly use Rotary Positional Embeddings (RoPE) or ALiBi instead of simple learned embeddings — covered fully in Phase 6 Lesson 3 |

---

## 8. Visual Explanations

**Full decoder-only Transformer block (data flow):**
```
Input X ──┬─────────────────────────────────────┐
          │                                       │
          ▼                                       │ (residual connection)
   Multi-Head Causal                              │
   Self-Attention                                  │
          │                                       │
          ▼                                       │
         (+)◀────────────────────────────────────┘
          │
          ▼
     LayerNorm  ──▶ X'  ──┬─────────────────────────────┐
                            │                              │
                            ▼                              │ (residual connection)
                    Feedforward (GELU)                     │
                            │                              │
                            ▼                              │
                           (+)◀────────────────────────────┘
                            │
                            ▼
                       LayerNorm  ──▶ X''  (feeds the NEXT block, or the final LM head)
```

**Stacked blocks building increasingly abstract representations (analogous to CNN depth, Lesson 3):**
```
Block 1: local syntax, adjacent-token relationships
Block 2-4: phrase-level structure, simple semantic relationships
Block 5+: long-range dependencies, abstract/compositional meaning
(exact behavior per layer is empirically studied via probing research, not architecturally guaranteed)
```

---

## 9. Practical Examples

**Simple:** implement `layer_norm` and verify it produces zero mean, unit variance across the feature dimension for a sample input.
**Medium:** assemble a full `TransformerBlock` (Section 5) and verify the output shape matches the input shape (a Transformer block is shape-preserving, allowing arbitrary stacking).
**Real-world:** train the `MiniTransformer` (Section 5) on a small character-level text corpus (e.g., a public domain book excerpt) to predict the next character, generate sample text via the Section 6 sampling loop at a few different temperatures, and qualitatively compare output coherence — directly, hands-on preparation for Phase 6's full-scale LLM pretraining lesson.

---

## 10. Real Industry Use Cases

- **Every major LLM in 2026** (GPT-family, Claude, Llama, Gemini, Mistral): decoder-only Transformer stacks, exactly this lesson's architecture scaled to dozens/hundreds of layers and billions/trillions of parameters.
- **BERT and encoder-only models**: bidirectional Transformer encoders (no causal masking) remain widely used for embedding generation, classification, and retrieval tasks (directly relevant to Phase 7's RAG systems).
- **Vision Transformers (ViT)**: apply the identical block structure to sequences of image patches instead of text tokens, demonstrating the architecture's modality-agnostic generality.
- **Multimodal models**: increasingly unify text, image, and audio processing within variants of this same Transformer block structure, with modality-specific tokenization/embedding layers feeding a shared architectural backbone.

---

## 11. Common Mistakes

- Forgetting residual connections when stacking many Transformer blocks — without them, gradients vanish through depth exactly as in any sufficiently deep network lacking skip connections (Phase 5 Lesson 1/3's vanishing gradient concerns).
- Confusing Pre-LN and Post-LN placement, or being unaware the distinction matters — a real, measurable architectural choice affecting training stability at scale, not an arbitrary implementation detail.
- Applying the causal mask inconsistently or incorrectly when building a decoder-only model — silently allows information leakage from future tokens, producing deceptively good training metrics that don't reflect real generation-time behavior (Lesson 6's exact warning, now at the full-model level).
- Using an inappropriate temperature for text generation — very low temperature produces repetitive, deterministic-feeling text; very high temperature produces incoherent output; the right value is task-dependent and usually needs empirical tuning.

---

## 12. Best Practices (2026)

- Use Pre-LN (not the original paper's Post-LN) for any new Transformer implementation intended to scale to significant depth — the now-standard choice in virtually every major modern LLM architecture.
- Rely on Hugging Face Transformers' pre-built, pretrained model classes for any real application rather than training a Transformer from scratch — Phase 6 will cover fine-tuning these pretrained models, the standard, resource-efficient approach in practice.
- Use modern positional encoding schemes (RoPE, Phase 6 Lesson 3) rather than the original paper's fixed sinusoidal or simple learned positional embeddings, for better length generalization.
- When generating text, use appropriate sampling strategies (nucleus/top-p sampling, not just raw temperature) for the best balance of coherence and diversity — Hugging Face's `.generate()` API exposes these directly.

---

## 13. Exercises

**Easy:** Implement layer normalization and verify its output has zero mean and unit variance along the feature axis.
**Medium:** Implement the full `TransformerBlock` (Section 5) and verify that a forward pass through 6 stacked blocks doesn't produce `NaN`/`Inf` values, confirming basic numerical stability.
**Hard:** Implement both Pre-LN and Post-LN variants of the Transformer block and empirically compare gradient norms (via gradient checking or direct inspection) across 20 stacked layers, demonstrating Pre-LN's improved stability at depth.
**Mathematical:** Derive the approximate total parameter count formula ($\approx 12Ld^2$) for a decoder-only Transformer, given $L$ layers and model dimension $d$, assuming a $4d$-dimensional feedforward hidden layer.
**Coding:** Implement top-p (nucleus) sampling from scratch (select the smallest set of tokens whose cumulative probability exceeds $p$, then sample only from that set) as an alternative to the simple temperature-based sampling in Section 6.

---

## 14. Mini Project

**Fully train the `MiniTransformer` (Section 5) as a character-level language model**: assemble the complete architecture with backpropagation (extending Lesson 2's techniques through the attention and feedforward sublayers — either by hand or, more practically, by reimplementing in PyTorch with autograd, previewing Lesson 8), train on a modest text corpus, generate sample text at several temperatures, and write a short report comparing this fully-from-scratch small Transformer's behavior against what you'd expect from Phase 6's much larger, pretrained production LLMs — directly setting up Phase 6's pretraining and fine-tuning content as a natural continuation of exactly this architecture at vastly greater scale.

---

## 15. Interview Preparation

- Explain the full Transformer block's data flow: self-attention, residual connections, layer normalization, and the position-wise feedforward network.
- Why are residual connections essential for training deep Transformer stacks?
- What's the difference between Pre-LN and Post-LN, and why does it matter in practice?
- Explain the autoregressive text generation loop and the role of temperature in sampling.

---

## 16. Summary

The Transformer assembles Lesson 6's self-attention mechanism with position-wise feedforward layers, residual connections, and layer normalization into a fully parallelizable, recurrence-free architecture — residual connections directly address the vanishing-gradient risk of stacking many such blocks (echoing ResNet's insight in CNNs, Lesson 3), while Pre-LN placement (the modern standard over the original Post-LN) further stabilizes training at the depth and scale used by real LLMs. This lesson's `MiniTransformer`, built entirely from scratch across Lessons 5-7, is architecturally identical in kind — just vastly smaller — to every production LLM Phase 6 will build on, making the transition from here directly a matter of scale (more layers, larger model dimension, far more training data) rather than fundamentally new architecture.

---

## 17. References

- Vaswani et al. — "Attention Is All You Need" (2017, the original Transformer paper)
- Xiong et al. — "On Layer Normalization in the Transformer Architecture" (2020, the Pre-LN vs Post-LN analysis)
- Radford et al. — "Improving Language Understanding by Generative Pre-Training" (2018, GPT-1, establishing the decoder-only paradigm)
- Alammar, J. — "The Illustrated GPT-2" (jalammar.github.io, an excellent visual walkthrough of the decoder-only architecture)
