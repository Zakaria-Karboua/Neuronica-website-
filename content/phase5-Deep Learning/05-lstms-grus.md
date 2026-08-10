# Phase 5 · Lesson 5 — LSTMs & GRUs

> Prerequisite: RNNs (Lesson 4) — this lesson directly addresses the vanishing gradient problem formalized there

---

## 1. Introduction

### What are LSTMs and GRUs?
Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks are architectural refinements of the vanilla RNN (Lesson 4), introducing explicit **gating mechanisms** that allow the network to selectively retain, forget, or update information over long sequences — directly and specifically solving the vanishing gradient problem that made vanilla RNNs unable to learn long-range dependencies.

### Why does it exist?
Lesson 4 showed mathematically that vanilla RNNs' repeated Jacobian multiplication through Backpropagation Through Time causes gradients (and thus learnable dependencies) to decay exponentially with sequence length. LSTMs (Hochreiter & Schmidhuber, 1997) introduced a separate "cell state" with an **additive** (rather than purely multiplicative) update path specifically designed to let gradients flow essentially unchanged across many time steps when needed.

### Historical background
LSTMs were proposed remarkably early (1997) but only became widely practical and dominant roughly 15 years later (2013-2016) once sufficient compute and training data existed to exploit their advantages — a notable example of a foundational idea's practical value depending on infrastructure catching up to the theory. GRUs (Cho et al., 2014) later offered a simplified, often comparably effective alternative with fewer parameters and gates.

### Real-world motivation
Before Transformers (Lesson 7) became dominant, LSTMs were the standard architecture behind production machine translation (early Google Translate), speech recognition, and text generation systems throughout the mid-2010s — understanding their gating mechanism is both historically important and directly illuminates the "selective information flow" idea that attention (Lesson 6) generalizes even further.

---

## 2. Theory

### The LSTM cell — four components working together
- **Cell state** $C_t$: a separate "memory highway" that information can flow along with minimal transformation — the key structural innovation enabling long-range gradient flow.
- **Forget gate** $f_t$: decides what fraction of the previous cell state to discard.
- **Input gate** $i_t$: decides what new information to write into the cell state.
- **Output gate** $o_t$: decides what part of the (updated) cell state to expose as the hidden state output.

Each gate is a sigmoid-activated layer producing values in $[0,1]$, acting as a *soft, learnable switch* — 0 means "block completely," 1 means "pass completely," with everything in between representing partial information flow, all differentiable and trainable via ordinary backpropagation.

### The GRU — a simplified alternative
GRUs merge the forget and input gates into a single **update gate**, and eliminate the separate cell state entirely (folding its role into the hidden state directly) — fewer parameters, often comparable performance to LSTMs, and in practice, the choice between the two is frequently empirical (try both, keep whichever validates better) rather than theoretically predetermined.

---

## 3. Mathematical Foundations

### LSTM equations, fully specified
$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}
$$
$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}
$$
$$
\tilde C_t = \tanh(W_C[h_{t-1}, x_t] + b_C) \quad \text{(candidate cell update)}
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t \quad \text{(cell state update — THE key equation)}
$$
$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \qquad h_t = o_t \odot \tanh(C_t)
$$

### Why the cell state update solves vanishing gradients
The crucial term is $C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t$ — this is an **additive**, not purely multiplicative, update. Taking the gradient of $C_t$ with respect to $C_{t-1}$:
$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t \quad \text{(plus additional terms from } i_t, \tilde C_t \text{'s own dependence on } C_{t-1} \text{ through } h_{t-1}\text{)}
$$
If the forget gate $f_t \approx 1$ (the network has learned "remember this"), gradients flow backward through the cell state **almost unchanged**, avoiding the repeated small-Jacobian-multiplication decay that plagued vanilla RNNs' purely multiplicative $h_t = \tanh(W_{hh}h_{t-1} + \dots)$ recurrence (Lesson 4, Section 3). This is the precise mathematical mechanism — not just an empirical claim — behind LSTMs' ability to learn dependencies across hundreds of time steps where vanilla RNNs fail within tens.

### GRU equations
$$
z_t = \sigma(W_z[h_{t-1},x_t]) \quad \text{(update gate)}, \qquad r_t = \sigma(W_r[h_{t-1},x_t]) \quad \text{(reset gate)}
$$
$$
\tilde h_t = \tanh(W[r_t \odot h_{t-1}, x_t]), \qquad h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t
$$
The update-gate equation for $h_t$ has the same additive-blending structure as the LSTM's cell-state update — $(1-z_t)$ directly preserves the old hidden state when $z_t \approx 0$, providing the same essential gradient-flow benefit with one fewer gate and no separate cell state.

### Parameter count comparison
For hidden size $h$ and input size $x$: vanilla RNN has $O(h(h+x))$ parameters; LSTM has $4\times$ that (four gates, each its own weight matrix); GRU has $3\times$ (three gates) — a genuine, quantifiable efficiency argument for GRU when compute/memory is constrained and empirical performance is comparable.

---

## 4. Algorithm — LSTM Forward Pass (fully specified)

```
INITIALIZE: h_0 = 0, C_0 = 0
FOR t = 1 to T:
    combined = concatenate(h_{t-1}, x_t)
    f_t = sigmoid(W_f @ combined + b_f)
    i_t = sigmoid(W_i @ combined + b_i)
    C_tilde_t = tanh(W_C @ combined + b_C)
    C_t = f_t * C_{t-1} + i_t * C_tilde_t        # THE additive gradient-preserving update
    o_t = sigmoid(W_o @ combined + b_o)
    h_t = o_t * tanh(C_t)
    CACHE all gate values, C_t, h_t for backward pass
RETURN all h_t (and/or a final output layer applied to each h_t)
```

---

## 5. Python Implementation

```python
"""lstm_gru_core.py — from-scratch LSTM forward pass + PyTorch equivalents"""
import numpy as np
import torch
import torch.nn as nn


def sigmoid(z): return 1 / (1 + np.exp(-z))


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        concat_size = input_size + hidden_size
        scale = 0.1
        # One combined weight matrix per gate, operating on [h_{t-1}, x_t] concatenated
        self.W_f = rng.normal(0, scale, (hidden_size, concat_size))
        self.W_i = rng.normal(0, scale, (hidden_size, concat_size))
        self.W_C = rng.normal(0, scale, (hidden_size, concat_size))
        self.W_o = rng.normal(0, scale, (hidden_size, concat_size))
        self.b_f = np.ones(hidden_size)     # bias initialized to 1: START by remembering (a known good practice)
        self.b_i = np.zeros(hidden_size)
        self.b_C = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = X.shape[0]
        h = np.zeros(self.hidden_size)
        C = np.zeros(self.hidden_size)
        hidden_states = []
        for t in range(T):
            combined = np.concatenate([h, X[t]])
            f = sigmoid(self.W_f @ combined + self.b_f)
            i = sigmoid(self.W_i @ combined + self.b_i)
            C_tilde = np.tanh(self.W_C @ combined + self.b_C)
            C = f * C + i * C_tilde                          # THE key additive update (Section 3)
            o = sigmoid(self.W_o @ combined + self.b_o)
            h = o * np.tanh(C)
            hidden_states.append(h.copy())
        return np.array(hidden_states), C


# --- PyTorch equivalents (production-grade, GPU-capable) ---
class TorchLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)    # out: all hidden states; h_n, c_n: final hidden/cell state
        return self.fc(out[:, -1, :])      # use the LAST time step's hidden state for a many-to-one task


class TorchGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.gru(x)             # GRU has NO separate cell state -- only hidden state
        return self.fc(out[:, -1, :])
```

**Note the forget-gate bias initialization (`b_f = ones`)**: a well-known practical trick — initializing the forget gate to favor "remember" ($f_t$ close to 1 initially) helps gradient flow from the very start of training, before the network has learned anything, directly exploiting Section 3's insight that $f_t \approx 1$ is what preserves gradient signal.

---

## 6. Build From Scratch

**Empirically comparing vanilla RNN vs. LSTM gradient flow over a long sequence (directly extending Lesson 4 Section 6's measurement):**
```python
import numpy as np

def measure_lstm_gradient_by_lag(lstm: "LSTMCell", seq_length: int = 100):
    """A simplified numerical (not full analytic) check: perturb the input at each early time step
    slightly and measure how much the FINAL hidden state changes -- a proxy for gradient magnitude."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(seq_length, lstm.W_f.shape[1] - lstm.hidden_size))
    baseline_hidden, _ = lstm.forward(X)
    baseline_final = baseline_hidden[-1]

    sensitivities = []
    epsilon = 1e-3
    for t in [0, seq_length // 4, seq_length // 2, seq_length - 1]:
        X_perturbed = X.copy()
        X_perturbed[t] += epsilon
        perturbed_hidden, _ = lstm.forward(X_perturbed)
        sensitivity = np.linalg.norm(perturbed_hidden[-1] - baseline_final) / epsilon
        sensitivities.append((t, sensitivity))
    return sensitivities

lstm = LSTMCell(input_size=4, hidden_size=16)
print(measure_lstm_gradient_by_lag(lstm, seq_length=100))
# Expect: sensitivity at t=0 (earliest) remains MEASURABLY non-negligible,
# a qualitatively different result than the vanilla RNN's near-zero sensitivity at long lags (Lesson 4)
```

---

## 7. Library/Tool Comparison

| From scratch | PyTorch |
|---|---|
| `LSTMCell.forward` | `torch.nn.LSTM` — highly optimized (often fused CUDA kernels), full backward pass via autograd, supports multi-layer/bidirectional variants directly |
| No GRU from scratch shown (structurally analogous) | `torch.nn.GRU` — drop-in alternative, fewer parameters |
| Manual forget-gate bias trick | Some frameworks/tutorials apply this automatically; worth verifying/setting explicitly regardless |

---

## 8. Visual Explanations

**LSTM cell state as a "gradient highway" (additive updates, minimal transformation):**
```
C_{t-1} ──────(×f_t)──────┬──────▶ C_t ──────(×f_{t+1})──────┬──────▶ C_{t+1} ──▶ ...
                           │                                   │
                    (+ i_t*C_tilde_t)                  (+ i_{t+1}*C_tilde_{t+1})
   (mostly ADDITIVE flow when f_t≈1 -- gradients pass through with minimal shrinkage,
    unlike the vanilla RNN's purely multiplicative h_t = tanh(W_hh h_{t-1} + ...) recurrence)
```

**LSTM vs GRU structural comparison:**
```
LSTM:  separate cell state C_t + hidden state h_t;  4 gates (forget, input, candidate, output)
GRU:   single hidden state h_t only;                 3 gates (update, reset, candidate)
       (GRU = a simpler, often comparably effective, lower-parameter-count alternative)
```

---

## 9. Practical Examples

**Simple:** implement the LSTM forward pass (Section 5) and process a short sequence, inspecting how the cell state evolves differently from the hidden state.
**Medium:** train an LSTM on the same long-range-dependency synthetic task that a vanilla RNN (Lesson 4) struggled with, and empirically confirm the LSTM succeeds where the vanilla RNN failed.
**Real-world:** replace the vanilla RNN forecaster from Lesson 4's mini project with an LSTM (and separately, a GRU) on your DZD/Brent oil time series, comparing forecasting performance and training stability under identical walk-forward validation — extending the growing multi-architecture comparison table.

---

## 10. Real Industry Use Cases

- **Pre-2017 production NLP/speech systems**: LSTMs powered the first generation of genuinely useful neural machine translation (early Google Neural Machine Translation used a deep LSTM encoder-decoder), speech recognition, and text generation systems.
- **Time series forecasting**: LSTMs remain a viable, sometimes-competitive option for demand forecasting, financial time series, and sensor/IoT data, particularly for genuinely long sequences where classical ARIMA struggles.
- **Anomaly detection in sequential data**: LSTM-based autoencoders remain used for detecting anomalies in time series (e.g., predictive maintenance, fraud pattern detection over transaction sequences).
- **Legacy production systems**: many deployed systems built in the 2015-2020 window still run on LSTM architectures, making understanding them directly relevant to maintaining/improving existing production ML systems, not purely historical interest.

---

## 11. Common Mistakes

- Assuming LSTMs completely eliminate the vanishing gradient problem — they substantially mitigate it (enabling learning across hundreds, not just tens, of time steps) but extremely long sequences (thousands+ of steps) can still pose challenges, part of why attention/transformers (Lessons 6-7) were further developed.
- Forgetting the forget-gate bias initialization trick (Section 5) — a small change with a real, measurable impact on early training dynamics.
- Using an LSTM when a simpler GRU would perform comparably with less computation/memory — always worth empirically comparing both rather than defaulting to LSTM out of habit.
- Not using bidirectional LSTMs/GRUs when the full sequence is available at training/inference time (e.g., offline text classification) — leaving useful "future context" information unused.

---

## 12. Best Practices (2026)

- Try both LSTM and GRU empirically on your specific task/dataset — neither dominates universally, and the "extra" LSTM cell state sometimes helps, sometimes doesn't, depending on the specific dependency structure in the data.
- Apply the forget-gate bias initialization trick (Section 5) as a small, well-established improvement for training stability.
- For any new sequence modeling project in 2026, seriously evaluate transformer-based approaches (Lesson 7) first — LSTMs/GRUs are now primarily chosen for specific latency/resource-constrained deployment scenarios or for maintaining/extending existing legacy systems, rather than as a default choice for new large-scale projects.
- Use bidirectional variants when the full sequence is available (not for real-time streaming prediction).

---

## 13. Exercises

**Easy:** Implement the LSTM forward pass (Section 5) and verify the cell state and hidden state have different values at each time step.
**Medium:** Implement a GRU cell from scratch (following the equations in Section 3) and verify its parameter count is $3/4$ that of an equivalently-sized LSTM.
**Hard:** Train both a vanilla RNN (Lesson 4) and an LSTM on the identical long-range-dependency synthetic task, and produce a plot showing task accuracy as a function of the dependency's lag distance for each architecture — directly, empirically demonstrating the LSTM's advantage.
**Mathematical:** Derive $\partial C_t/\partial C_{t-1}$ fully (including all paths through $i_t$ and $\tilde C_t$'s dependence on $h_{t-1}$, not just the direct $f_t$ term) and discuss under what conditions this derivative stays close to 1.
**Coding:** Implement the forget-gate bias initialization trick and empirically compare training loss curves (first 20 epochs) with vs. without this initialization on a synthetic sequence task.

---

## 14. Mini Project

Complete the **multi-architecture time series forecasting comparison** (extending Lessons 3-4's mini projects): add LSTM and GRU forecasters to your DZD/Brent oil comparison table (alongside ARIMA/naive baseline, 1D CNN, vanilla RNN), evaluate all under identical walk-forward validation with bootstrap confidence intervals, and write a final comprehensive analysis of which architecture performs best on your specific data characteristics (sequence length, noise level, seasonality) — a genuinely thorough, publication-quality forecasting methodology comparison spanning classical statistics through modern deep learning.

---

## 15. Interview Preparation

- Explain the LSTM's cell state and why its additive update helps solve the vanishing gradient problem.
- What's the difference between an LSTM and a GRU, and when might you choose one over the other?
- Derive (at a high level) why $f_t \approx 1$ preserves gradient flow across many time steps.
- Why might you still choose an LSTM/GRU over a Transformer for a specific production use case in 2026?

---

## 16. Summary

LSTMs solve vanilla RNNs' vanishing gradient problem (Lesson 4) through an explicit, gated cell state whose additive (rather than purely multiplicative) update allows gradients to flow across many time steps largely unchanged when the forget gate favors retention — a precise, derivable mathematical mechanism, not just an empirical fix. GRUs offer a simplified, often comparably effective alternative with fewer parameters and no separate cell state. Both architectures dominated production sequence modeling for roughly a decade (mid-2010s to ~2020) and remain relevant for specific deployment constraints and legacy systems today, while also serving as essential conceptual scaffolding — the notion of "selectively gating information flow across a sequence" — for understanding why attention mechanisms (Lesson 6) were subsequently developed as an even more powerful, fully parallelizable alternative.

---

## 17. References

- Hochreiter, S. & Schmidhuber, J. — "Long Short-Term Memory" (1997, the original LSTM paper)
- Cho et al. — "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014, the original GRU paper)
- Olah, C. — "Understanding LSTM Networks" (colah's blog — an exceptionally clear, widely-cited visual explanation)
- Jozefowicz, Zaremba, Sutskever — "An Empirical Exploration of Recurrent Network Architectures" (2015, systematic LSTM/GRU comparison)
