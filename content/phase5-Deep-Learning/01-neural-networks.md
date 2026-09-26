# Phase 5 · Lesson 1 — Neural Networks

> Prerequisite: Phase 3 (Linear Algebra, Calculus, Optimization), Phase 4 Lesson 1 (Supervised Learning)

---

## 1. Introduction

### What is a neural network?
A function approximator built from layers of simple computational units ("neurons"), each performing a weighted sum of inputs followed by a non-linear activation function, stacked so that the composition of many simple layers can approximate extremely complex functions. Where Phase 4's models (linear/logistic regression, trees) each have a specific, hand-designed structure suited to particular problems, neural networks are a general-purpose function approximator whose *architecture* is designed but whose specific function is learned entirely from data.

### Why does it exist?
Biologically inspired (McCulloch & Pitts, 1943, modeling neurons as simple threshold logic units) but its modern relevance is purely mathematical: the Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) proves a sufficiently wide single-hidden-layer network can approximate any continuous function on a compact domain arbitrarily well — a remarkable existence proof, though it says nothing about *how many* parameters or how much data are needed in practice (depth, it turns out empirically and increasingly theoretically, matters enormously for tractable learning, not just width).

### Historical background
Perceptrons (Rosenblatt, 1958) could only learn linearly separable functions — a limitation proven rigorously by Minsky & Papert (1969), contributing to the first "AI winter." Backpropagation's rediscovery/popularization (Rumelhart, Hinton, Williams, 1986 — directly Phase 3 Lesson 2's territory) made training multi-layer networks practical, but it took until the 2010s (better hardware — GPUs, more data, and architectural/optimization refinements) for deep networks to decisively outperform classical ML on perceptual tasks (ImageNet/AlexNet, 2012), triggering the deep learning era this entire curriculum sits within.

### Real-world motivation
Every model in Phase 6 (LLMs) and much of Phase 7 (embeddings, RAG components) is, at its core, exactly this lesson's architecture — layers, weights, activations, trained via backpropagation — scaled up enormously and specialized (Lessons 3-7, this phase) for particular data structures (images, sequences, text).

---

## 2. Theory

### The single neuron (perceptron/logistic unit)

$$
z = w^Tx + b, \qquad a = \phi(z)
$$

where $\phi$ is a non-linear **activation function**. Without $\phi$ being non-linear, stacking layers would be pointless — a composition of purely linear functions is still linear (a single matrix), so non-linearity is what gives depth its expressive power.

### Common activation functions
| Function | Formula | Notes |
|---|---|---|
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | historically common, suffers vanishing gradients at extremes |
| Tanh | $\tanh(z)$ | zero-centered (unlike sigmoid), still saturates |
| ReLU | $\max(0,z)$ | dominant default since ~2012 — cheap, non-saturating for $z>0$, but "dying ReLU" risk for $z<0$ |
| GELU/SiLU | smooth approximations of ReLU | standard in modern transformers (Phase 5 Lessons 6-7, Phase 6) |
| Softmax | $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$ | used at the OUTPUT layer for multi-class classification, converts logits to a valid probability distribution |

### Feedforward architecture (Multi-Layer Perceptron, MLP)

$$
h^{(1)} = \phi(W^{(1)}x + b^{(1)}), \quad h^{(2)} = \phi(W^{(2)}h^{(1)} + b^{(2)}), \quad \dots, \quad \hat y = \text{output layer}(h^{(L)})
$$

Each layer's output feeds the next — a direct composition of linear transformations (Phase 3 Lesson 1) and non-linearities, with the network's total parameter count growing with both width (units per layer) and depth (number of layers).

### Why depth matters (not just width)
Empirically (and with growing theoretical support), deep networks can represent certain function classes exponentially more compactly than shallow-but-wide networks — each layer can be understood as learning increasingly abstract, composed features from the previous layer's representation (e.g., in vision: edges → textures → parts → objects), a hierarchical feature-learning capability entirely absent from Phase 4's flat feature-engineering approach.

---

## 3. Mathematical Foundations

### Universal Approximation Theorem (statement, not full proof)
For any continuous function $f$ on a compact subset of $\mathbb{R}^n$ and any $\epsilon > 0$, there exists a single-hidden-layer network with a finite number of units and a suitable (non-polynomial) activation function $\phi$ such that the network approximates $f$ within $\epsilon$ everywhere on that domain. **Caveat, critically**: the theorem is non-constructive — it doesn't say how many units are needed (potentially astronomically many for complex $f$) or how to *find* the right weights via a tractable optimization procedure; depth is what makes both of these practically tractable in ways width alone often isn't.

### Weight initialization theory (why it's not arbitrary)
Naive initialization (all zeros) creates a **symmetry problem**: every neuron in a layer receives identical gradients and stays identical forever — no learning happens. Random initialization breaks symmetry, but the *scale* matters enormously:

$$
\text{Xavier/Glorot init: } W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)
$$


$$
\text{He init (for ReLU): } W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

Both are derived by requiring the *variance* of activations (and gradients, during backpropagation) to remain roughly constant across layers — without this, activations either vanish toward zero or explode toward infinity as they propagate through many layers, a foundational numerical-stability concern.

### Vanishing/exploding gradients (directly connecting to Phase 3 Lesson 2's chain rule)
Backpropagating through $L$ layers multiplies $L$ Jacobian terms together (the chain rule, applied repeatedly). If each layer's Jacobian has eigenvalues consistently $<1$, the product shrinks exponentially with depth (**vanishing gradients** — early layers barely learn); if consistently $>1$, it grows exponentially (**exploding gradients** — training diverges). This is *precisely why* activation choice (ReLU vs. saturating sigmoid/tanh), careful initialization, normalization layers (BatchNorm/LayerNorm), and residual connections (Lesson 7, this phase) all matter so much for training genuinely deep networks.

### Universal approximation vs. generalization (a crucial distinction)
Being able to *represent* a target function (approximation) is entirely different from being able to *learn* good parameters for it from finite data with a tractable optimizer (generalization) — a network with enough parameters can perfectly memorize training data (zero training error) while generalizing terribly (Phase 4 Lesson 1's bias-variance tradeoff, now in its highest-variance-capacity regime), which is why regularization (dropout, weight decay, early stopping) remains essential even for enormously expressive models.

---

## 4. Algorithm — Forward Pass Through an MLP (fully specified)

```
GIVEN input x, weights W^(1)...W^(L), biases b^(1)...b^(L), activation phi:
h^(0) = x
FOR l = 1 to L:
    z^(l) = W^(l) @ h^(l-1) + b^(l)
    h^(l) = phi(z^(l))          # apply non-linearity, EXCEPT typically at the final output layer
                                  # (which uses softmax for classification, or identity for regression)
RETURN h^(L) as the network's prediction
```
Complexity: $O(\sum_l n_l \times n_{l-1})$ (total number of weight parameters) per forward pass — dominated by matrix multiplications, exactly Phase 3 Lesson 1's $O(n^3)$-style cost per layer, now the actual computational bottleneck of neural network training and inference at scale.

---

## 5. Python Implementation

```python
"""neural_networks_core.py — a from-scratch MLP using only NumPy"""
import numpy as np


def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))   # numerical stability: subtract max BEFORE exponentiating
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class MLP:
    """A 2-hidden-layer MLP, weights initialized via He initialization (Section 3)."""
    def __init__(self, layer_sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)))   # He init
            self.biases.append(np.zeros(n_out))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """Returns final output AND cached intermediate activations (needed for backprop, Lesson 2)."""
        activations = [X]
        z_values = []
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            z_values.append(z)
            is_last_layer = (i == len(self.weights) - 1)
            h = softmax(z) if is_last_layer else relu(z)
            activations.append(h)
        return h, {"activations": activations, "z_values": z_values}


# Example: a 3-layer MLP classifying 4 features into 3 classes
model = MLP(layer_sizes=[4, 16, 8, 3])
X_sample = np.random.default_rng(1).normal(size=(10, 4))
predictions, cache = model.forward(X_sample)
print("Predictions (should sum to 1 per row):", predictions.sum(axis=1).round(4))
```

---

## 6. Build From Scratch

Section 5's `MLP` class already is the from-scratch implementation — the forward pass and initialization are built with no library beyond NumPy. Backpropagation (training this network) is deliberately deferred to Lesson 2, which formalizes it fully. Here, we add a from-scratch **dropout** layer (a key regularization technique for controlling the overfitting risk raised in Section 3):

```python
def dropout(h: np.ndarray, rate: float, training: bool, rng: np.random.Generator) -> np.ndarray:
    """Randomly zeroes a fraction `rate` of activations during training; scales at TEST time (inverted dropout)."""
    if not training or rate == 0:
        return h
    mask = (rng.random(h.shape) > rate).astype(float)
    return h * mask / (1 - rate)    # "inverted dropout": scale during TRAINING so no change needed at test time
```
**Why it works, intuitively**: forcing the network to make correct predictions even when a random subset of units is "missing" prevents any single unit (or small co-adapted group of units) from becoming a fragile, over-relied-upon shortcut — a form of implicit ensembling (each dropout mask defines a slightly different sub-network, and training implicitly averages over this exponentially large ensemble, directly echoing Phase 4 Lesson 5's bagging variance-reduction logic).

---

## 7. Library Implementation (Comparison)

| From scratch | PyTorch |
|---|---|
| `MLP.forward` (manual matrix multiplies + activations) | `torch.nn.Sequential(nn.Linear(...), nn.ReLU(), ...)` — GPU-accelerated, integrated autograd |
| Manual He initialization | `torch.nn.init.kaiming_normal_` — identical formula, standard library function |
| `dropout` (manual mask) | `torch.nn.Dropout(p=...)` — automatically handles train/eval mode switching |
| No backprop yet (Lesson 2) | `loss.backward()` — full automatic differentiation (Phase 3 Lesson 2's autograd engine), not hand-derived gradients |

---

## 8. Visual Explanations

**MLP architecture (fully connected layers):**
```
Input       Hidden 1      Hidden 2      Output
 x1 ─┬────▶ (h1_1) ─┬───▶ (h2_1) ─┬───▶ (softmax) ──▶ class probs
 x2 ─┼────▶ (h1_2) ─┼───▶ (h2_2) ─┤
 x3 ─┼────▶ (h1_3) ─┼───▶ (h2_3) ─┤
 x4 ─┴────▶ (h1_4) ─┴───▶ (h2_4) ─┘
     (every unit connects to EVERY unit in the next layer -- "fully connected"/"dense")
```

**Vanishing gradients through depth (Jacobian eigenvalues < 1, compounding):**
```
Layer:     1        2        3        4        5   (gradient magnitude, illustrative)
Gradient: 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 -> ...  (shrinks EXPONENTIALLY with depth)
          (early layers receive a vanishingly small training signal -- barely update)
```

---

## 9. Practical Examples

**Simple:** implement a single-layer perceptron (no hidden layers) and confirm it can only solve linearly separable problems (fails on XOR).
**Medium:** implement the 2-hidden-layer MLP (Section 5) and forward-pass a batch of synthetic data, verifying output shapes and that softmax outputs sum to 1.
**Real-world:** design (architecture only, forward pass) an MLP for a tabular actuarial classification task (mortality prediction) with an input layer matching your engineered feature count (Phase 2 Lesson 6), comparing its representational flexibility conceptually against the XGBoost model from Phase 4 Lesson 1 — anticipating Lesson 2's full training implementation.

---

## 10. Real Industry Use Cases

- **Every deep learning system** in this curriculum's remaining phases (CNNs, RNNs, Transformers) is built from this lesson's fundamental building blocks (linear layers + non-linear activations), specialized via architectural constraints (Lessons 3-7).
- **Tabular deep learning** (a genuinely contested area in 2026): MLPs and specialized tabular architectures (TabNet, FT-Transformer) compete with XGBoost/gradient boosting on tabular data, though gradient boosting still frequently wins on small-to-medium tabular datasets — directly relevant to whether you'd choose a neural network or XGBoost for your actuarial work.
- **Every embedding/representation-learning system** (Phase 6-7): built from stacked linear+non-linear layers, exactly this lesson's architecture, just deeper and wider.

---

## 11. Common Mistakes

- Initializing all weights to zero (or the same value) — creates the symmetry problem (Section 3), preventing any effective learning.
- Using sigmoid/tanh activations in very deep networks without careful initialization/normalization — a direct route to vanishing gradients.
- Forgetting that dropout must be disabled at test/inference time (or properly scaled, as in "inverted dropout," Section 6) — a common source of a train/inference mismatch bug.
- Assuming more layers/parameters always helps — without enough data or proper regularization, additional capacity often just increases overfitting (Section 3's approximation-vs-generalization distinction) without improving real-world performance.

---

## 12. Best Practices (2026)

- Default to ReLU or its modern smooth variants (GELU, SiLU) over sigmoid/tanh for hidden layers in any reasonably deep network.
- Always use principled initialization (He for ReLU-family, Xavier/Glorot for tanh/sigmoid) — never naive random or zero initialization.
- Use dropout and/or weight decay (L2 regularization, Phase 3 Lesson 1) as standard regularization for any network with enough capacity to plausibly overfit its training data.
- For tabular data specifically, benchmark any neural network approach against a well-tuned gradient-boosted tree ensemble (Phase 4 Lesson 1) — don't assume deep learning wins by default outside of its traditionally dominant domains (vision, language, audio).

---

## 13. Exercises

**Easy:** Implement a single perceptron with a step-function activation and manually verify it correctly classifies a simple linearly separable 2D dataset.
**Medium:** Implement the forward pass for a 3-hidden-layer MLP from scratch (extending Section 5) and verify output shapes at every layer.
**Hard:** Implement both Xavier and He initialization from scratch, and empirically compare the variance of activations across 10 layers of a deep, purely-forward (untrained) network under each initialization scheme, confirming Section 3's variance-preservation claim.
**Mathematical:** Derive why the sum of pre-activation variances stays roughly constant across layers under He initialization, assuming ReLU activations and independent, zero-mean inputs.
**Coding:** Implement inverted dropout (Section 6) and empirically verify that the expected value of a dropped-out layer's output matches the non-dropout (full) layer's output, confirming the scaling correction is mathematically consistent.

---

## 14. Mini Project

Build a **from-scratch MLP architecture (forward pass only, training deferred to Lesson 2) for tabular actuarial classification**: design an architecture with 2-3 hidden layers sized appropriately for your engineered feature set (Phase 2 Lesson 6), implement He initialization, ReLU activations, dropout, and a softmax output layer, verify all shapes and numerical stability (no NaNs/overflow) across a large synthetic batch, and write a short architectural justification (why this depth/width, why these regularization choices) — setting up Lesson 2's full backpropagation-based training of this exact architecture.

---

## 15. Interview Preparation

- Explain why non-linear activation functions are necessary for a multi-layer network to be more expressive than a single linear layer.
- What is the vanishing gradient problem, and what techniques address it?
- Why does weight initialization matter, and what problem does Xavier/He initialization solve?
- Explain how dropout works and why it helps prevent overfitting.

---

## 16. Summary

Neural networks stack linear transformations (Phase 3 Lesson 1) with non-linear activations to build universal function approximators (Cybenko/Hornik's theorem), with depth providing empirically and increasingly theoretically superior compactness over pure width for learning genuinely complex, hierarchical functions. Careful weight initialization (Xavier/He) and non-saturating activations (ReLU family) directly address the vanishing/exploding gradient problem that plagues naive deep networks, while regularization techniques like dropout control the very real overfitting risk that comes with such high representational capacity — every architecture in the remainder of Phase 5 (CNNs, RNNs, Transformers) is a structured specialization of exactly these foundational building blocks.

---

## 17. References

- Goodfellow, Bengio, Courville — *Deep Learning* (the definitive textbook, freely available online)
- Cybenko, G. — "Approximation by Superpositions of a Sigmoidal Function" (1989, the original Universal Approximation Theorem)
- Glorot & Bengio — "Understanding the Difficulty of Training Deep Feedforward Neural Networks" (2010, Xavier initialization)
- He et al. — "Delving Deep into Rectifiers" (2015, He initialization)
- Srivastava et al. — "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
