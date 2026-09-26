# Phase 5 · Lesson 4 — Recurrent Neural Networks (RNNs)

> Prerequisite: Neural Networks, Backpropagation (Lessons 1–2). Phase 4 Lesson 6 (Time Series) provides useful context.

---

## 1. Introduction

### What is an RNN?
A neural network architecture designed for **sequential** data (text, time series, audio), where the network maintains a **hidden state** that gets updated at each time step, carrying information forward from earlier elements of the sequence to influence processing of later ones — in contrast to CNNs' fixed, spatially-local receptive fields, RNNs are built specifically to handle variable-length sequences with potentially long-range dependencies.

### Why does it exist?
Feedforward networks (Lesson 1) and CNNs (Lesson 3) have no inherent notion of sequential order or memory across time steps — each input is processed independently (CNNs have local spatial context but no temporal recurrence). Language, speech, and many time series problems fundamentally require understanding that the meaning of an element depends on what came before it — RNNs introduce a recurrent connection specifically to capture this.

### Historical background
RNNs' basic recurrent formulation traces to the 1980s-90s (Elman networks, 1990), but training them via **Backpropagation Through Time (BPTT)** exposed a severe vanishing/exploding gradient problem (Lesson 1's Section 3 concern, dramatically amplified over long sequences) that made learning long-range dependencies nearly impossible with vanilla RNNs — directly motivating LSTMs (Lesson 5), which solved this problem well enough to dominate sequence modeling from the mid-2010s until transformers (Lesson 7) largely superseded both starting around 2017-2020.

### Real-world motivation
Though transformers have displaced RNNs for most large-scale language modeling (Phase 6), RNN concepts (hidden state, sequential processing, the vanishing gradient problem they exposed) remain essential intellectual scaffolding for understanding *why* attention (Lesson 6) and transformers were specifically designed the way they were — this lesson is as much about understanding a historically pivotal architecture as using one directly.

---

## 2. Theory

### The recurrence relation

$$
h_t = \phi(W_{hh}h_{t-1} + W_{xh}x_t + b_h), \qquad \hat y_t = W_{hy}h_t + b_y
$$

The **same weights** ($W_{hh}, W_{xh}, W_{hy}$) are reused at *every* time step — a weight-sharing principle directly analogous to CNNs' spatial weight sharing (Lesson 3), but here shared across *time* rather than *space*. The hidden state $h_t$ acts as a compressed summary of everything seen up to time $t$.

### Unrolling through time
An RNN processing a sequence of length $T$ can be "unrolled" into an equivalent feedforward network with $T$ layers, each sharing the same weights — this unrolled view is precisely what makes Backpropagation Through Time (BPTT) just an application of Lesson 2's ordinary backpropagation, treating the unrolled network as a (very deep, weight-tied) feedforward graph.

### Sequence modeling variants
- **Many-to-one**: sequence in, single output (e.g., sentiment classification of a review).
- **Many-to-many (aligned)**: one output per input time step (e.g., part-of-speech tagging).
- **Many-to-many (unaligned/seq2seq)**: encode a full input sequence, then decode an output sequence of potentially different length (e.g., machine translation) — the direct historical precursor to the encoder-decoder transformer architecture (Lesson 7).
- **Bidirectional RNNs**: process the sequence both forward and backward, concatenating both hidden states — useful when the *entire* sequence is available at once (not for real-time/streaming prediction, since the backward pass requires future context).

---

## 3. Mathematical Foundations

### Backpropagation Through Time (BPTT), derived
Since $h_t$ depends on $h_{t-1}$, the gradient of the loss (summed across all time steps) with respect to $W_{hh}$ involves a sum over all the ways $W_{hh}$ influenced the loss — through every subsequent time step:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t}\left(\prod_{k=t}^{2}\frac{\partial h_k}{\partial h_{k-1}}\right)\frac{\partial h_1}{\partial W_{hh}}
$$

The product term $\prod \partial h_k/\partial h_{k-1}$ is the crux of the vanishing/exploding gradient problem: each factor is (roughly) $W_{hh}^T \text{diag}(\phi'(z_k))$, and multiplying $T$ such matrices together (for a long sequence) causes the product's magnitude to shrink toward zero (vanishing) or grow without bound (exploding) exponentially in $T$ — far more severe than the depth-related vanishing gradient problem in Lesson 1, since $T$ (sequence length) can be far larger than any practical feedforward network's depth $L$.

### Why vanishing gradients specifically cripple long-range dependency learning
If gradients from time step $T$ back to time step $1$ have shrunk to near-zero by the time they reach $W_{hh}$'s update at step 1, the network effectively **cannot learn** dependencies spanning more than a handful of time steps — even if such a dependency is genuinely present and important in the data (e.g., a pronoun in a sentence referring back to a noun many words earlier) — a concrete, formalizable limitation, not just an empirical observation.

### Gradient clipping (the standard mitigation for exploding gradients)

$$
\text{if } \|g\| > \tau: \quad g \leftarrow \tau \cdot \frac{g}{\|g\|}
$$

Rescaling the gradient vector to have norm at most $\tau$ whenever it exceeds that threshold — a simple, direct fix for the *exploding* half of the problem (does nothing for vanishing gradients, which require an architectural fix — exactly LSTMs'/GRUs' purpose, Lesson 5).

---

## 4. Algorithm — Vanilla RNN Forward and BPTT (fully specified)

```
FORWARD PASS (unrolling through time):
  h_0 = zeros (initial hidden state)
  FOR t = 1 to T:
      z_t = W_hh @ h_{t-1} + W_xh @ x_t + b_h
      h_t = tanh(z_t)              # cache z_t, h_{t-1}, h_t for backward pass
      y_hat_t = W_hy @ h_t + b_y
  RETURN all y_hat_t, and cached h_t/z_t values

BACKWARD PASS (BPTT -- accumulate gradients across ALL time steps for the SHARED weights):
  dW_hh, dW_xh, dW_hy = 0, 0, 0     # accumulators -- weights are SHARED, so gradients SUM across time
  dh_next = zeros                    # gradient flowing in from the FUTURE time step
  FOR t = T down to 1:
      dy_t = y_hat_t - y_t (or other loss gradient)
      dW_hy += dy_t @ h_t.T
      dh_t = W_hy.T @ dy_t + dh_next               # gradient from THIS step's output + FUTURE step's hidden state
      dz_t = dh_t * (1 - h_t**2)                    # tanh derivative
      dW_hh += dz_t @ h_{t-1}.T
      dW_xh += dz_t @ x_t.T
      dh_next = W_hh.T @ dz_t                        # propagate to the PREVIOUS time step
  CLIP all gradients to norm tau (Section 3)
  RETURN dW_hh, dW_xh, dW_hy
```

---

## 5. Python Implementation

```python
"""rnn_core.py — from-scratch vanilla RNN + PyTorch equivalent"""
import numpy as np
import torch
import torch.nn as nn


class VanillaRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.W_xh = rng.normal(0, scale, (hidden_size, input_size))
        self.W_hh = rng.normal(0, scale, (hidden_size, hidden_size))
        self.W_hy = rng.normal(0, scale, (output_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """X: (T, input_size) -- a single sequence. Returns outputs (T, output_size) and cache."""
        T = X.shape[0]
        h = np.zeros(self.hidden_size)
        cache = {"h": [h.copy()], "z": [], "x": []}
        outputs = []
        for t in range(T):
            z = self.W_hh @ h + self.W_xh @ X[t] + self.b_h
            h = np.tanh(z)
            y = self.W_hy @ h + self.b_y
            outputs.append(y)
            cache["z"].append(z); cache["h"].append(h.copy()); cache["x"].append(X[t])
        return np.array(outputs), cache

    def backward(self, d_outputs: np.ndarray, cache: dict, clip_norm: float = 5.0):
        T = len(cache["x"])
        dW_hh = np.zeros_like(self.W_hh)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hy = np.zeros_like(self.W_hy)
        dh_next = np.zeros(self.hidden_size)

        for t in reversed(range(T)):
            dy = d_outputs[t]
            dW_hy += np.outer(dy, cache["h"][t + 1])
            dh = self.W_hy.T @ dy + dh_next
            dz = dh * (1 - cache["h"][t + 1] ** 2)          # tanh'(z) = 1 - tanh(z)^2
            dW_hh += np.outer(dz, cache["h"][t])
            dW_xh += np.outer(dz, cache["x"][t])
            dh_next = self.W_hh.T @ dz

        for grad in (dW_hh, dW_xh, dW_hy):                    # GRADIENT CLIPPING (Section 3)
            norm = np.linalg.norm(grad)
            if norm > clip_norm:
                grad *= clip_norm / norm
        return dW_hh, dW_xh, dW_hy


# --- PyTorch equivalent, for comparison ---
class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)          # out: (batch, seq_len, hidden_size)
        return self.fc(out)            # (batch, seq_len, output_size)
```

---

## 6. Build From Scratch

Section 5's `VanillaRNN` already implements forward and backward (BPTT) fully from scratch. The natural extension here is **empirically demonstrating the vanishing gradient problem**, to make Section 3's math concrete and visible:

```python
import numpy as np

def measure_gradient_magnitude_by_lag(rnn: "VanillaRNN", seq_length: int = 50):
    """Feeds a long sequence, backprops a loss ONLY at the FINAL time step,
    and measures how much gradient signal reaches EARLIER time steps."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(seq_length, rnn.W_xh.shape[1]))
    outputs, cache = rnn.forward(X)

    d_outputs = np.zeros_like(outputs)
    d_outputs[-1] = np.ones(outputs.shape[1])   # gradient ONLY from the final time step's output

    # Manually track ||dh_t|| at each step during backward pass (modified BPTT loop)
    T = len(cache["x"])
    dh_next = np.zeros(rnn.hidden_size)
    gradient_norms = []
    for t in reversed(range(T)):
        dy = d_outputs[t]
        dh = rnn.W_hy.T @ dy + dh_next
        gradient_norms.append(np.linalg.norm(dh))
        dz = dh * (1 - cache["h"][t + 1] ** 2)
        dh_next = rnn.W_hh.T @ dz
    return list(reversed(gradient_norms))   # index 0 = earliest time step

rnn = VanillaRNN(input_size=4, hidden_size=16, output_size=2)
norms = measure_gradient_magnitude_by_lag(rnn, seq_length=50)
print("Gradient norm at step 0 (earliest):", norms[0])
print("Gradient norm at step 49 (latest):", norms[-1])
# Expect: norms[0] << norms[-1] -- direct empirical evidence of vanishing gradients over long sequences
```

---

## 7. Library/Tool Comparison

| From scratch | PyTorch |
|---|---|
| `VanillaRNN.forward`/`backward` | `torch.nn.RNN` — handles batching, variable-length sequences (via padding/packing), GPU acceleration |
| Manual BPTT loop | Automatic — `.backward()` unrolls the computation graph exactly as Section 4 describes, but automatically |
| Manual gradient clipping | `torch.nn.utils.clip_grad_norm_` — same formula, standard library function |
| No LSTM/GRU shown here | `torch.nn.LSTM`/`torch.nn.GRU` — Lesson 5's architectural fix for vanishing gradients, drop-in replacements for `nn.RNN` |

---

## 8. Visual Explanations

**RNN unrolled through time (weight sharing across time steps):**
```
       x1        x2        x3        x4
       │         │         │         │
       ▼         ▼         ▼         ▼
h0 ──▶[RNN]──▶h1──▶[RNN]──▶h2──▶[RNN]──▶h3──▶[RNN]──▶h4
       │(W shared) │(W shared) │(W shared) │(W shared)
       ▼         ▼         ▼         ▼
      y1        y2        y3        y4
  (SAME W_hh, W_xh, W_hy reused at every time step -- weight sharing across TIME, not space)
```

**Vanishing gradient over a long sequence (gradient magnitude by lag, schematic):**
```
Gradient norm
  reaching   │█
  each step  │██
  (backward  │███
   from the  │████
   final     │██████
   loss)     │████████████████████████  <- barely any signal reaches EARLY time steps
             └──────────────────────────  time step (0=earliest ... T=latest, loss computed at T)
```

---

## 9. Practical Examples

**Simple:** implement the vanilla RNN forward pass and process a short synthetic sequence, inspecting hidden state evolution.
**Medium:** train a many-to-one vanilla RNN (Section 5) on a simple sequence classification task and empirically observe its difficulty learning dependencies beyond ~10-15 time steps.
**Real-world:** apply an RNN-based approach to your DZD exchange-rate or claims time-series data as a direct architectural comparison against the 1D CNN (Lesson 3) and ARIMA/XGBoost (Phase 4 Lesson 6) approaches, under identical walk-forward validation — building toward a complete, multi-architecture forecasting comparison.

---

## 10. Real Industry Use Cases

- **Historical dominance (pre-2017-2020)**: RNNs/LSTMs (Lesson 5) powered the first generation of production neural machine translation, speech recognition, and text generation systems before transformers (Lesson 7) took over.
- **Still-relevant niches**: RNNs remain used in some latency-sensitive streaming/real-time applications (online, single-token-at-a-time processing) where transformers' full-sequence attention computation is less naturally suited, though transformer variants increasingly address this too.
- **Time series forecasting**: RNN/LSTM-based forecasters remain a viable, sometimes-competitive option alongside classical (ARIMA) and modern (transformer-based) approaches, particularly for genuinely long, irregularly-sampled sequences.
- **Conceptual foundation for Transformers**: understanding *why* RNNs struggle with long-range dependencies (Section 3's vanishing gradient analysis) is the direct motivation for attention (Lesson 6), which solves the same problem by an entirely different, non-recurrent mechanism.

---

## 11. Common Mistakes

- Expecting a vanilla RNN to learn genuinely long-range dependencies (50+ time steps) without architectural help (LSTM/GRU, Lesson 5) — Section 3's math shows this is a fundamental limitation, not a hyperparameter-tuning problem.
- Forgetting gradient clipping when training RNNs — exploding gradients (the "other half" of Section 3's problem) can cause `NaN` losses and completely derail training within a few iterations.
- Using a vanilla RNN's *final* hidden state as a fixed-size sequence summary for very long sequences, then being surprised when early-sequence information is effectively lost — a direct consequence of the vanishing gradient problem limiting what actually gets encoded into that final state.
- Processing sequences of very different lengths in a batch without proper padding/masking — silently including padding tokens in loss computation or hidden-state updates, corrupting training.

---

## 12. Best Practices (2026)

- Default to LSTM/GRU (Lesson 5) over vanilla RNNs for essentially any real application — vanilla RNNs are primarily of historical/pedagogical interest at this point.
- For new sequence modeling projects, seriously evaluate transformer-based approaches (Lesson 7) first, given their now-dominant performance and superior parallelization (RNNs' sequential dependency prevents parallelizing across time steps during training, a major practical disadvantage transformers eliminate).
- Always use gradient clipping when training any recurrent architecture.
- Use padding + masking (or PyTorch's `pack_padded_sequence` utilities) correctly when batching variable-length sequences.

---

## 13. Exercises

**Easy:** Implement the vanilla RNN forward pass (Section 5) and process a batch of short sequences, inspecting output shapes.
**Medium:** Implement the full BPTT backward pass and train the RNN on a simple sequence-to-label task (e.g., predicting whether a sequence of numbers sums to more than a threshold).
**Hard:** Implement the Section 6 gradient-magnitude-by-lag measurement and empirically characterize at what sequence length vanishing gradients become severe enough to prevent effective learning, for a few different hidden-state sizes.
**Mathematical:** Derive the BPTT gradient formula for $\partial L/\partial W_{hh}$ explicitly for a 3-time-step sequence, showing the product-of-Jacobians term explicitly.
**Coding:** Implement gradient clipping from scratch and empirically demonstrate it prevents `NaN` losses on a synthetic RNN training run deliberately set up to have exploding gradients (e.g., via poor weight initialization scale).

---

## 14. Mini Project

Build a **complete RNN-based time series forecaster** for your DZD exchange-rate or Brent oil dataset: implement (or use `torch.nn.RNN` for a properly comparable, GPU-capable version) a many-to-one RNN predicting the next time step from a fixed-length window of past values, train under proper walk-forward validation (Phase 4 Lesson 6), and produce a final three-way comparison table (ARIMA/naive baseline, 1D CNN, RNN) with bootstrap confidence intervals on forecast error — directly completing the multi-architecture forecasting comparison begun in Lesson 3's mini project.

---

## 15. Interview Preparation

- Explain the RNN recurrence relation and how weight sharing across time steps works.
- Derive why vanilla RNNs suffer from vanishing/exploding gradients over long sequences.
- What is Backpropagation Through Time, and how does it relate to ordinary backpropagation?
- Why can't RNN training be parallelized across time steps the way CNN or transformer computation can be?

---

## 16. Summary

RNNs introduce a recurrent hidden state, weight-shared across time steps, to process variable-length sequences — but Backpropagation Through Time's repeated multiplication of Jacobian terms across potentially many time steps causes a severe, mathematically explicable vanishing (or exploding, addressed via gradient clipping) gradient problem that fundamentally limits vanilla RNNs' ability to learn long-range dependencies. This specific, well-understood limitation is the direct historical motivation for both LSTMs/GRUs (Lesson 5's architectural fix) and, ultimately, attention mechanisms and transformers (Lessons 6-7), which abandon recurrence altogether in favor of a fundamentally different way of relating distant sequence elements.

---

## 17. References

- Elman, J. — "Finding Structure in Time" (1990, foundational RNN paper)
- Werbos, P. — "Backpropagation Through Time: What It Does and How to Do It" (1990)
- Pascanu, Mikolov, Bengio — "On the Difficulty of Training Recurrent Neural Networks" (2013, rigorous vanishing/exploding gradient analysis + gradient clipping)
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 10 (Sequence Modeling: Recurrent and Recursive Nets)
