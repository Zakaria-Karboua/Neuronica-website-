# Phase 5 · Lesson 8 — PyTorch

> Prerequisite: All prior Phase 5 lessons — this lesson shows how the framework automates everything built by hand so far

---

## 1. Introduction

### What is PyTorch?
An open-source deep learning framework (Meta AI, first released 2016) providing a `Tensor` object (NumPy-like, but GPU-capable and autograd-enabled), automatic differentiation (Phase 3 Lesson 2's reverse-mode autodiff, fully generalized and productionized), and a rich ecosystem of neural network building blocks (`torch.nn`), optimizers (`torch.optim`), and data-loading utilities (`torch.utils.data`).

### Why does it exist?
Every technique built by hand across this phase — forward passes, backpropagation, gradient descent, attention, Transformer blocks — is mathematically correct but painfully slow in pure NumPy (no GPU support, no automatic differentiation, manual gradient derivation required for every new architecture). PyTorch exists to make exactly this mathematics fast (GPU/TPU-accelerated), automatic (autograd derives gradients for *any* composition of differentiable operations, not just the ones you've manually derived), and composable (a large ecosystem of pretrained models, layers, and tools).

### Historical background
PyTorch emerged from Torch (a Lua-based framework) and gained rapid adoption over early alternatives (Theano, and to a significant degree TensorFlow 1.x) specifically because of its "define-by-run" (eager execution) design — the computation graph is built dynamically as Python code executes, making debugging and experimentation dramatically easier than TensorFlow 1.x's original static-graph-then-execute paradigm (a design difference TensorFlow 2.x later largely adopted in response). By the mid-2020s, PyTorch is the dominant framework for both research and, increasingly, production deep learning, including nearly all major LLM training codebases.

### Real-world motivation
Everything from Phase 5 Lessons 1-7 that you built with raw NumPy — you will now rebuild (better, faster, GPU-capable) in a fraction of the code, using PyTorch's autograd instead of hand-derived backpropagation. This is the framework Phase 6's LLM fine-tuning content assumes fluency in.

---

## 2. Theory

### The `Tensor` — NumPy's `ndarray`, generalized
A `torch.Tensor` behaves almost identically to a NumPy array (same broadcasting rules, Phase 2 Lesson 1) but additionally: can live on a GPU (`tensor.to("cuda")`), and can track gradients (`requires_grad=True`), building a dynamic computation graph as operations are applied — directly the `Value` autodiff engine from Phase 3 Lesson 2, but generalized to full tensors and implemented in optimized C++/CUDA.

### Autograd — automatic reverse-mode differentiation, productionized
Calling `.backward()` on a scalar tensor (typically a loss) triggers exactly Phase 5 Lesson 2's backpropagation algorithm, automatically, for *any* sequence of PyTorch operations — no manual gradient derivation required, regardless of architecture complexity.

### `torch.nn.Module` — the building block of every model
Every layer and full model is a subclass of `nn.Module`, implementing `__init__` (define sub-layers/parameters) and `forward` (define the computation) — directly the OOP composition patterns from Phase 1 Lessons 3 and 10 (Composite pattern: a `Module` can contain other `Module`s, forming a tree, with operations like `.parameters()` or `.to(device)` recursing through the whole tree).

### The standard training loop
Every PyTorch training loop follows the same skeleton: forward pass → compute loss → zero previous gradients → backward pass → optimizer step — a pattern you will write (or call a library that writes) essentially every time you train any model in Phase 6 onward.

### `Dataset` and `DataLoader`
`torch.utils.data.Dataset` defines how to access one example; `DataLoader` wraps a `Dataset` to provide batching, shuffling, and (crucially, for large datasets) multiprocessing-based prefetching (Phase 1 Lesson 2's multiprocessing concepts, applied directly) so data loading overlaps with GPU computation rather than blocking it.

---

## 3. Mathematical Foundations

### Autograd's computation graph, formalized (directly extending Phase 3 Lesson 2)
Every PyTorch tensor operation records a `grad_fn` — a reference to the operation that produced it and its inputs, exactly the `Value` class's `_prev`/`_backward` structure from Phase 3 Lesson 2, but now supporting full tensor-valued (not just scalar-valued) nodes, and implemented with production-grade memory management (the graph is freed after `.backward()` by default, unless `retain_graph=True` is specified).

### Why `optimizer.zero_grad()` is mandatory
PyTorch **accumulates** gradients into `.grad` by default (rather than overwriting them) on each `.backward()` call — a deliberate design choice enabling gradient accumulation across multiple mini-batches (useful for effectively larger batch sizes than fit in GPU memory) but requiring an explicit `zero_grad()` call before each new optimization step, or gradients from previous steps silently corrupt the current update — one of the most common PyTorch bugs for newcomers, and a direct, mechanical consequence of this accumulation-by-default design.

### Mixed-precision training (a practical, real optimization)
Using 16-bit floating point (`float16`/`bfloat16`) instead of 32-bit for most operations roughly halves memory usage and can substantially speed up training on modern GPUs (which have specialized hardware for lower-precision matrix multiplication), while keeping certain numerically sensitive operations (like loss scaling and gradient accumulation) in 32-bit precision to avoid the instability that naive full-16-bit training would cause — directly connecting to Phase 1 Lesson 1's floating-point precision discussion, now at production training scale.

### Batch normalization statistics (train vs. eval mode)
`BatchNorm` layers compute batch statistics (mean/variance) *during training* but use a running average of those statistics *during evaluation/inference* (since a single inference example, or a differently-sized batch, shouldn't have its normalization depend on whatever other examples happen to be in that particular inference batch) — precisely why `model.train()` vs. `model.eval()` mode matters, a real, easy-to-forget source of bugs (forgetting `model.eval()` before inference silently uses training-mode batch statistics).

---

## 4. Algorithm — The Standard PyTorch Training Loop (fully specified)

```
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

FOR each epoch:
    model.train()                          # enables dropout, uses BATCH statistics for BatchNorm
    FOR each batch (X, y) in train_loader:
        optimizer.zero_grad()               # CRITICAL: clear accumulated gradients from the previous step
        predictions = model(X)               # forward pass (calls model.forward(X) via __call__)
        loss = loss_fn(predictions, y)
        loss.backward()                      # autograd computes ALL parameter gradients in one backward pass
        optimizer.step()                     # update parameters using those gradients (Phase 3 Lesson 5's Adam, etc.)

    model.eval()                            # disables dropout, uses RUNNING statistics for BatchNorm
    WITH torch.no_grad():                    # disables gradient tracking -- saves memory, not needed for inference
        FOR each batch (X, y) in val_loader:
            predictions = model(X)
            val_loss = loss_fn(predictions, y)
    (log metrics, check early stopping, etc. -- Phase 4 Lesson 3's evaluation discipline)
```

---

## 5. Python Implementation

```python
"""pytorch_core.py — the Phase 5 Lesson 1 MLP, rebuilt properly in PyTorch"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ActuarialDataset(Dataset):
    """A custom Dataset -- defines how to access ONE example; DataLoader handles batching/shuffling."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MortalityMLP(nn.Module):
    """Directly mirrors Phase 5 Lesson 1's from-scratch MLP -- same architecture, autograd handles the rest."""
    def __init__(self, input_dim: int, hidden_dims: list[int], n_classes: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)   # returns LOGITS -- CrossEntropyLoss applies softmax internally


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  n_epochs: int = 20, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)   # AdamW: Phase 3 Lesson 5
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += loss_fn(logits, y_batch).item() * len(X_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader.dataset):.4f}, "
                  f"val_loss={val_loss/len(val_loader.dataset):.4f}, "
                  f"val_acc={correct/len(val_loader.dataset):.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    n, n_features, n_classes = 4000, 6, 2
    X = torch.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).long()   # a non-linear, interaction-driven target

    train_size = int(0.8 * n)
    train_ds = ActuarialDataset(X[:train_size], y[:train_size])
    val_ds = ActuarialDataset(X[train_size:], y[train_size:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MortalityMLP(input_dim=n_features, hidden_dims=[32, 16], n_classes=n_classes)
    train_model(model, train_loader, val_loader, n_epochs=20, device=device)
```

---

## 6. Build From Scratch

Rather than re-deriving anything (PyTorch *is* the production tool here), the valuable "from scratch" exercise is writing a **custom `autograd.Function`** — demonstrating exactly how PyTorch's autograd extends to operations it doesn't natively provide, directly bridging back to Phase 3 Lesson 2's `Value` engine:

```python
import torch

class CustomReLU(torch.autograd.Function):
    """Manually implementing forward AND backward for an operation -- rarely needed
    (PyTorch has nn.ReLU already) but essential to understand for genuinely custom operations."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # cache what's needed for the backward pass (Phase 5 Lesson 2's caching)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0              # EXACTLY Phase 5 Lesson 1's relu_derivative, applied to the incoming gradient
        return grad_input

x = torch.randn(5, requires_grad=True)
y = CustomReLU.apply(x)
loss = y.sum()
loss.backward()
print(x.grad)   # matches (x > 0).float(), confirming the custom backward pass is correct
```
This is precisely how you would implement a genuinely novel differentiable operation not already in PyTorch — and, notably, is exactly the mechanism you'd use to gradient-check (Phase 5 Lesson 2) against `torch.autograd.gradcheck` when writing anything non-standard.

---

## 7. Library Implementation (Comparison)

Since this lesson *is* about the library, the more useful comparison is **NumPy-from-scratch vs. PyTorch**, summarizing the whole phase:

| From scratch (NumPy, Lessons 1-7) | PyTorch |
|---|---|
| Manual forward pass, manual caching | `nn.Module.forward` + autograd's automatic graph construction |
| Hand-derived backpropagation (Lesson 2) | `.backward()` — works for ANY architecture without manual derivation |
| Manual gradient descent/Adam (Phase 3 Lesson 5) | `torch.optim.SGD`/`Adam`/`AdamW` — identical formulas, GPU-accelerated |
| CPU-only, single-threaded | GPU/TPU acceleration via `.to(device)`, essential for any non-trivial model |
| No automatic batching/data pipeline | `Dataset`/`DataLoader` — batching, shuffling, multiprocess prefetching |

---

## 8. Visual Explanations

**PyTorch's dynamic computation graph (built as code executes, freed after `.backward()`):**
```
x (requires_grad=True) ──▶ [Linear] ──▶ [ReLU] ──▶ [Linear] ──▶ loss
      │                        │             │           │
      └── each op RECORDS its grad_fn, building the graph DYNAMICALLY, step by step
             (contrast with old-style STATIC graphs, built entirely before any data flows through)

loss.backward()  -->  walks the graph in REVERSE (Phase 5 Lesson 2's algorithm), populates .grad on every leaf
```

**Train vs. eval mode (BatchNorm/Dropout behavioral switch):**
```
model.train():                          model.eval():
  Dropout: randomly zeroes units          Dropout: PASSES EVERYTHING THROUGH (no zeroing)
  BatchNorm: uses THIS BATCH's stats      BatchNorm: uses RUNNING AVERAGE stats (from training)
     (forgetting to switch modes is a common, subtle, silent-failure-mode bug)
```

---

## 9. Practical Examples

**Simple:** create a `torch.Tensor` with `requires_grad=True`, perform a few operations, call `.backward()`, and inspect `.grad`.
**Medium:** rebuild the Phase 5 Lesson 1 from-scratch MLP as a PyTorch `nn.Module` and confirm it trains successfully using `.backward()` + `optimizer.step()`, with dramatically less code than the manual NumPy version.
**Real-world:** rebuild your Lesson 2 mini project (the actuarial mortality MLP) in PyTorch, using a proper `Dataset`/`DataLoader` pipeline, `AdamW`, and GPU acceleration if available — directly comparing training speed and code simplicity against the from-scratch NumPy version.

---

## 10. Real Industry Use Cases

- **Virtually every LLM training codebase** (Phase 6): built on PyTorch (or JAX, a less common but present alternative), using exactly the `nn.Module` + autograd + optimizer pattern from this lesson, just at vastly larger scale (distributed training across thousands of GPUs).
- **Hugging Face Transformers**: built entirely on top of PyTorch's `nn.Module` system — every pretrained model class you'll use in Phase 6 IS a PyTorch module.
- **Production inference serving**: PyTorch models are commonly exported (via TorchScript or ONNX) for optimized, framework-agnostic deployment, though increasingly, PyTorch itself is used directly in production inference servers (Phase 8).
- **Research**: PyTorch's dynamic graph/eager execution remains the dominant choice for research code specifically because of its debuggability (standard Python debugging tools work naturally, unlike historically with static-graph frameworks).

---

## 11. Common Mistakes

- Forgetting `optimizer.zero_grad()` — gradients silently accumulate across steps, corrupting training (Section 3's explicit warning).
- Forgetting `model.eval()` before inference/validation — BatchNorm/Dropout behave as if still training, giving incorrect (often visibly worse, sometimes bizarrely inconsistent) predictions.
- Forgetting `with torch.no_grad()` during inference — wastes memory building an unnecessary computation graph for tensors that will never call `.backward()`.
- Moving only the model (not the data) to a GPU (or vice versa) — causes a device-mismatch runtime error; both model and data tensors must be on the same device for any operation between them.

---

## 12. Best Practices (2026)

- Always use `AdamW` (not plain `Adam`) as your default optimizer for most deep learning tasks — the decoupled weight decay is now the standard, empirically preferred variant (Phase 3 Lesson 5).
- Use `torch.utils.data.DataLoader` with appropriate `num_workers` for any dataset large enough that data loading could bottleneck GPU utilization.
- Wrap training with mixed-precision (`torch.cuda.amp.autocast`/`GradScaler`, or the newer `torch.amp` API) for meaningful speed/memory improvements on modern GPUs, when training larger models.
- Use `torch.compile` (introduced in PyTorch 2.0, standard by 2026) to just-in-time compile models for significant inference/training speedups with minimal code changes.

---

## 13. Exercises

**Easy:** Create a tensor with `requires_grad=True`, compute $y = x^2 + 3x$, call `.backward()`, and verify `x.grad` matches the analytic derivative $2x+3$.
**Medium:** Rebuild the Phase 5 Lesson 3 CNN (`SimpleCNN`) and train it on a small image dataset, comparing training speed on CPU vs. GPU (if available).
**Hard:** Implement a custom `autograd.Function` (Section 6) for an operation not natively in PyTorch (e.g., a custom activation function of your choosing) and validate it via `torch.autograd.gradcheck`.
**Mathematical:** Explain, using the chain rule (Phase 3 Lesson 2), why gradient accumulation (calling `.backward()` multiple times before `optimizer.step()`, without `zero_grad()` in between) is mathematically equivalent to training with a larger effective batch size.
**Coding:** Implement a full PyTorch training loop with early stopping (halt training when validation loss stops improving for a set number of epochs) and learning-rate scheduling (Phase 3 Lesson 5's warmup+decay).

---

## 14. Mini Project

**Rebuild every from-scratch Phase 5 model in PyTorch and benchmark them**: reimplement the MLP (Lesson 1), the CNN (Lesson 3), and the tiny Transformer (Lesson 7) as proper `nn.Module` classes, train each using the standard PyTorch training loop (Section 4), and produce a comparison report covering (a) lines of code required vs. the NumPy-from-scratch versions, (b) training speed on CPU (and GPU, if available), and (c) final task performance — a comprehensive, hands-on demonstration of exactly what a production deep learning framework buys you over hand-derived implementations, closing out Phase 5 by connecting every prior lesson's manual mathematics to the tool you'll use for the rest of this curriculum.

---

## 15. Interview Preparation

- Explain what `optimizer.zero_grad()` does and why it's necessary.
- What's the difference between `model.train()` and `model.eval()`, and which layers does it affect?
- Explain PyTorch's dynamic (define-by-run) computation graph and how it differs from a static graph framework.
- How would you implement a custom differentiable operation not natively supported by PyTorch?

---

## 16. Summary

PyTorch productionizes every technique built by hand across Phase 5: `Tensor` generalizes NumPy's `ndarray` with GPU support and autograd, `nn.Module` formalizes the composable layer/model structure explored via OOP (Phase 1) throughout this phase's architectures, and the standard training loop (`zero_grad` → `forward` → `loss.backward()` → `optimizer.step()`) automates exactly Phase 5 Lesson 2's hand-derived backpropagation for architectures of arbitrary complexity, including every CNN, RNN, LSTM, and Transformer this phase has covered. Fluency here — understanding *why* each step of the training loop exists, not just memorizing the four-line incantation — is the direct prerequisite for Phase 6's LLM pretraining and fine-tuning content, which assumes this exact framework as its working substrate.

---

## 17. References

- Paszke et al. — "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019, the official PyTorch paper)
- Official PyTorch documentation and tutorials (pytorch.org/tutorials) — exceptionally well-maintained and current
- Stevens, Antiga, Viehmann — *Deep Learning with PyTorch* (comprehensive, widely-used practical reference)
- Hugging Face — "The Transformers Course" (huggingface.co/course, directly bridging this lesson into Phase 6's territory)
