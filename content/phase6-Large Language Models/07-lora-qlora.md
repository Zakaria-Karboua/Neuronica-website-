# Phase 6 · Lesson 7 — LoRA & QLoRA

> Prerequisite: Fine-Tuning (Lesson 6), Phase 3 Linear Algebra

---

## 1. Introduction

### What are LoRA and QLoRA?
LoRA (Low-Rank Adaptation, Hu et al., 2021) is a parameter-efficient fine-tuning (PEFT) technique that freezes a pretrained model's original weights entirely and instead trains a small number of additional low-rank matrices injected alongside them — achieving fine-tuning results comparable to full fine-tuning while updating often less than 1% of the total parameter count. QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantization (Lesson 8) of the frozen base model, enabling fine-tuning of very large models on a single consumer or prosumer GPU.

### Why does it exist?
Full fine-tuning (Lesson 6) of a modern LLM requires storing gradients and optimizer states for *every* parameter — for a 70-billion-parameter model, this can require hundreds of gigabytes of GPU memory, far beyond what all but the largest, most expensive hardware setups provide. LoRA exists specifically to make fine-tuning economically and computationally accessible, dramatically reducing memory requirements while empirically matching full fine-tuning's performance on most tasks.

### Historical background
LoRA (2021) built on earlier observations that fine-tuning updates to large pretrained models tend to have low "intrinsic rank" — the actual change needed to specialize a model for a new task can be well-approximated by a much lower-dimensional update than the full weight matrix's dimensionality would suggest. QLoRA (2023) extended this by showing the *frozen* base model weights could be aggressively quantized (Lesson 8) to 4-bit precision with minimal quality loss, since only the small LoRA matrices (kept in higher precision) actually receive gradient updates — a combination that made fine-tuning genuinely large (65B+ parameter) open-weight models feasible on a single GPU, a widely-cited turning point for accessible LLM customization.

### Real-world motivation
If you ever fine-tune an open-weight model for SANAVIR or an actuarial application without access to a large GPU cluster, LoRA/QLoRA is almost certainly the technique you'll use — understanding exactly why and how it works is directly, practically relevant, not just theoretical background.

---

## 2. Theory

### The low-rank hypothesis
For a pretrained weight matrix $W_0 \in \mathbb{R}^{d\times k}$, LoRA hypothesizes that the *update* needed during fine-tuning, $\Delta W$, has low intrinsic rank $r \ll \min(d,k)$ — meaning $\Delta W$ can be well-approximated as the product of two much smaller matrices, $\Delta W \approx BA$ where $B \in \mathbb{R}^{d\times r}$ and $A \in \mathbb{R}^{r\times k}$, with $r$ often as small as 4-64 even for weight matrices with thousands of rows/columns.

### The LoRA mechanism
Rather than fine-tuning $W_0$ directly, LoRA **freezes** $W_0$ entirely and adds a parallel low-rank path:
$$
h = W_0x + \Delta W x = W_0x + BAx
$$
Only $A$ and $B$ are trained (with $A$ typically initialized randomly and $B$ initialized to **zero**, so the model's initial behavior exactly matches the original pretrained model before any training occurs — a deliberate, important initialization choice). The number of trainable parameters becomes $r(d+k)$ instead of $dk$ — for typical Transformer weight matrix dimensions and small $r$, a reduction of 100-1000×.

### QLoRA — combining LoRA with quantization
QLoRA quantizes the frozen base model $W_0$ to 4-bit precision (Lesson 8's territory) while keeping the LoRA adapter matrices $A, B$ in higher precision (e.g., bfloat16) — since $W_0$ never receives gradients (it's frozen), its reduced numerical precision has minimal impact on the *trainable* part of the model, while dramatically shrinking the memory footprint of the (otherwise dominant) frozen base weights.

### Where LoRA is typically applied
Empirically, applying LoRA to the attention mechanism's query and value projection matrices ($W_Q, W_V$, Phase 5 Lesson 6) captures most of the benefit at minimal parameter cost; some configurations extend LoRA to all linear layers (including the feedforward layers, Phase 5 Lesson 7) for a further quality/parameter-count tradeoff — a genuinely empirical choice, tunable per task and model.

---

## 3. Mathematical Foundations

### Parameter count reduction, quantified
For a weight matrix $W_0 \in \mathbb{R}^{d\times k}$ (e.g., $d=k=4096$ for a mid-sized LLM's attention projections), full fine-tuning requires updating $dk = 16{,}777{,}216$ parameters. With LoRA rank $r=8$:
$$
\text{LoRA parameters} = r(d+k) = 8 \times (4096+4096) = 65{,}536
$$
a **256× reduction** in trainable parameters for this single matrix — and since optimizer states (e.g., Adam's first and second moment estimates, Phase 3 Lesson 5) must be stored *per trainable parameter*, this reduction directly and proportionally shrinks the dominant memory cost of fine-tuning (not just the parameter storage itself, but the 2-3× additional memory Adam-family optimizers require per parameter).

### Why zero-initializing $B$ matters
At the start of training, $\Delta W = BA$; if $B$ is initialized to all zeros, $\Delta W = 0$ regardless of $A$'s (random) initialization — guaranteeing the model's *initial* output during fine-tuning is identical to the original pretrained model's output, a clean, controlled starting point from which the LoRA update can then learn a meaningful, task-specific adjustment via gradient descent, rather than starting from an arbitrary random perturbation of pretrained behavior.

### Rank $r$ as a genuine expressiveness/efficiency tradeoff
$r$ directly bounds $\Delta W$'s rank, and thus how expressive an update LoRA can represent — very small $r$ (e.g., $r=1$ or $2$) may be insufficient for tasks requiring substantial behavioral change, while larger $r$ approaches full fine-tuning's expressiveness (and cost) as $r \to \min(d,k)$. Empirically, quite small ranks ($r=8$ to $64$) suffice for most practical fine-tuning tasks — a genuinely important, non-obvious empirical finding validating the low-rank hypothesis (Section 2) across a wide range of real applications.

### Quantization error and QLoRA's numerical justification (preview of Lesson 8)
Quantizing $W_0$ introduces some representational error $\epsilon$ in the frozen weights, but since $W_0$'s *gradients* are never computed (it's entirely frozen), this error simply becomes a small, fixed perturbation to the forward-pass computation — it does not compound or amplify through a training process the way quantization error affecting *trainable* weights and their optimizer states would, which is precisely why QLoRA's combination (quantize what's frozen, keep what's trained in higher precision) works so well in practice.

---

## 4. Algorithm — LoRA Fine-Tuning (fully specified)

```
GIVEN a pretrained model with weight matrices W_0 (frozen), rank r, target modules (e.g., W_Q, W_V):
FOR each target weight matrix W_0 (shape d x k):
    INITIALIZE A (r x k) with small random values (e.g., Gaussian)
    INITIALIZE B (d x r) with ALL ZEROS                          # guarantees delta_W = 0 initially (Section 3)
    FREEZE W_0 entirely (no gradients computed for it)
    MARK A, B as trainable

FORWARD PASS at any layer with a LoRA-augmented matrix:
    h = W_0 @ x + (B @ A) @ x                                     # equivalently: h = W_0 @ x + B @ (A @ x)
                                                                    # (compute A@x FIRST -- cheaper, avoids
                                                                     ever materializing the full d x k delta_W)

TRAINING: standard fine-tuning loop (Lesson 6's SFT or DPO), but ONLY A and B receive gradient updates;
          W_0 remains frozen throughout

INFERENCE / DEPLOYMENT: EITHER
  (a) keep A, B separate and add their contribution at each forward pass (flexible: easy to swap adapters), OR
  (b) MERGE: W_merged = W_0 + BA, producing a single dense matrix identical in shape to the original
      (no extra inference-time computation, but loses the ability to easily swap/remove the adapter)
```

---

## 5. Python Implementation

```python
"""lora_core.py — a from-scratch LoRA linear layer + PyTorch integration pattern"""
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear layer with a trainable low-rank adapter."""
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False                            # FREEZE the original weights (Section 4)

        d_out, d_in = base_layer.weight.shape
        self.A = nn.Parameter(torch.randn(rank, d_in) * (1 / math.sqrt(rank)))   # small random init
        self.B = nn.Parameter(torch.zeros(d_out, rank))                           # ZERO init (Section 3)
        self.scaling = alpha / rank                                                # a common scaling convention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = (x @ self.A.T) @ self.B.T                     # compute A@x FIRST (cheaper, Section 4)
        return base_output + self.scaling * lora_output

    def merge_weights(self) -> nn.Linear:
        """Produces a single dense layer with LoRA's contribution folded in (Section 4, option b)."""
        merged = nn.Linear(self.base_layer.in_features, self.base_layer.out_features)
        with torch.no_grad():
            merged.weight.copy_(self.base_layer.weight + self.scaling * (self.B @ self.A))
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        return merged


def apply_lora_to_model(model: nn.Module, target_module_names: list[str], rank: int = 8) -> nn.Module:
    """Replaces specified nn.Linear layers with LoRALinear wrappers throughout a model."""
    for name, module in model.named_modules():
        for target in target_module_names:
            if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                original_layer = getattr(module, target)
                setattr(module, target, LoRALinear(original_layer, rank=rank))
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Example: wrap a single linear layer and verify parameter reduction
base = nn.Linear(4096, 4096)
lora_layer = LoRALinear(base, rank=8)
trainable, total = count_trainable_parameters(lora_layer)
print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
```

---

## 6. Build From Scratch

**Verifying the low-rank hypothesis empirically (checking how well a low-rank approximation captures a "true" fine-tuning update):**
```python
import numpy as np

def verify_low_rank_approximation(d: int = 512, k: int = 512, true_rank: int = 10, test_rank: int = 8):
    """Simulates a 'true' fine-tuning delta that genuinely IS low-rank, and checks how well
    a LoRA-style rank-r factorization recovers it via SVD (Phase 3 Lesson 1)."""
    rng = np.random.default_rng(0)
    true_B = rng.normal(size=(d, true_rank))
    true_A = rng.normal(size=(true_rank, k))
    true_delta_W = true_B @ true_A                                  # a genuinely rank-`true_rank` matrix

    U, S, Vt = np.linalg.svd(true_delta_W, full_matrices=False)     # Phase 3 Lesson 1's SVD, reused directly
    approx_delta_W = U[:, :test_rank] @ np.diag(S[:test_rank]) @ Vt[:test_rank, :]

    relative_error = np.linalg.norm(true_delta_W - approx_delta_W) / np.linalg.norm(true_delta_W)
    return relative_error

# If test_rank >= true_rank, reconstruction should be (near) PERFECT
print("Error with test_rank=10 (matches true rank):", verify_low_rank_approximation(true_rank=10, test_rank=10))
print("Error with test_rank=8 (UNDER true rank):", verify_low_rank_approximation(true_rank=10, test_rank=8))
print("Error with test_rank=50 (over true rank, real weights are rarely EXACTLY low-rank):",
      verify_low_rank_approximation(true_rank=10, test_rank=50))
```
This directly demonstrates, via Phase 3 Lesson 1's SVD, exactly why LoRA's low-rank assumption is reasonable: if a fine-tuning update genuinely has low intrinsic rank (as empirically observed across many real fine-tuning tasks), a rank-$r$ factorization captures it essentially perfectly once $r$ reaches that true rank — the mathematical crux of why LoRA works at all.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `LoRALinear`/`apply_lora_to_model` | Hugging Face `peft` library — production-grade LoRA/QLoRA implementation, integrated with `transformers` and `trl`, supporting many more target-module configurations and adapter-merging utilities |
| Manual parameter counting | `peft`'s built-in `print_trainable_parameters()` utility |
| No quantization shown (Lesson 8's territory) | `bitsandbytes` library — provides the 4-bit quantization QLoRA specifically relies on, integrated directly with `peft`/`transformers` |

---

## 8. Visual Explanations

**LoRA's parallel low-rank path (frozen base + trainable adapter):**
```
Input x
   │
   ├──────────────▶ [Frozen W_0] ──────────────┐
   │                (NO gradients computed)      │
   │                                              ▼
   └──▶ [A: r x k, small random init] ──▶ [B: d x r, ZERO init] ──▶ (+) ──▶ Output h
        (trainable, small)                (trainable, small)
   (at initialization: B=0, so the adapter path contributes NOTHING -- output = W_0 @ x exactly)
```

**Parameter count comparison (full fine-tuning vs. LoRA):**
```
Full fine-tuning:  [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]  ALL of W_0's d×k parameters trainable
LoRA (rank r=8):   [■■]                                  only r(d+k) parameters trainable
                    (a tiny fraction -- often <1% of the full matrix's parameter count)
```

---

## 9. Practical Examples

**Simple:** implement `LoRALinear` (Section 5) and verify that at initialization (before any training), its output exactly matches the base frozen layer's output.
**Medium:** apply LoRA to a small pretrained model's attention projections (using Hugging Face `peft`) and verify the reported trainable parameter percentage is a small fraction of the total.
**Real-world:** fine-tune (via `peft`'s LoRA integration) a small open-weight model on an actuarial/insurance-domain instruction dataset, comparing training memory usage and final task performance against a full-fine-tuning baseline (if feasible) or simply reporting the dramatic memory savings if full fine-tuning isn't practical on your available hardware.

---

## 10. Real Industry Use Cases

- **Virtually every open-weight LLM fine-tuning project outside major labs**: LoRA/QLoRA is the default, practically necessary approach given typical available GPU memory constraints.
- **Multi-tenant LLM serving**: companies serving many customized model variants (e.g., a different fine-tune per customer) often store only the small LoRA adapter weights per customer, dynamically swapping them on top of one shared frozen base model — a substantial storage and serving-infrastructure efficiency gain over storing a full fine-tuned copy per customer.
- **QLoRA's democratizing effect**: enabled fine-tuning of 65B+ parameter open-weight models on a single consumer/prosumer GPU (as demonstrated in the original QLoRA paper), a widely-cited factor in the explosion of community fine-tuned model variants since 2023.
- **Rapid experimentation**: LoRA's small trainable parameter count and fast training turnaround make it the default choice for quick fine-tuning experiments/iterations even at organizations with ample compute for full fine-tuning.

---

## 11. Common Mistakes

- Choosing an inappropriately small rank $r$ for a task genuinely requiring substantial behavioral change, resulting in underfitting relative to what full fine-tuning could achieve — rank should be empirically tuned per task, not assumed universally sufficient at a fixed small value.
- Forgetting to zero-initialize $B$ (or the equivalent in a given implementation) — breaks the clean "identical to pretrained model at initialization" property, adding unnecessary training instability.
- Applying LoRA only to a subset of layers that turns out insufficient for the target task, without empirically testing whether extending to additional target modules (e.g., feedforward layers, not just attention) improves results.
- Confusing the *adapter-swappable* deployment pattern (keeping LoRA weights separate) with the *merged* deployment pattern (Section 4) — each has different inference-time performance and flexibility tradeoffs, and conflating them can lead to unexpected serving-architecture decisions.

---

## 12. Best Practices (2026)

- Use Hugging Face `peft` for any real LoRA/QLoRA fine-tuning project rather than a from-scratch implementation — mature, well-tested, and integrates directly with the broader Hugging Face ecosystem (Lesson 10).
- Start with commonly-validated defaults (rank 8-16, applied to attention query/value projections) and empirically expand (higher rank, more target modules) only if initial results are insufficient for your task.
- Use QLoRA specifically when GPU memory is the binding constraint for fine-tuning a larger model than would otherwise fit.
- For production multi-tenant scenarios, seriously consider the adapter-swapping deployment pattern (keeping LoRA weights separate rather than merged) for storage and flexibility efficiency.

---

## 13. Exercises

**Easy:** Implement `LoRALinear` (Section 5) and verify its output exactly matches the base layer's output at initialization (before any training).
**Medium:** Train a LoRA-adapted small model on a toy task and compare final task performance against full fine-tuning of the same base model, at a few different rank values.
**Hard:** Implement the SVD-based low-rank verification (Section 6) using a "true" delta matrix derived from an actual full-fine-tuning run (rather than a synthetic random one) on a small model, and empirically assess how well the low-rank hypothesis holds for a real fine-tuning task.
**Mathematical:** Derive the exact parameter count formula for LoRA applied to a $d\times k$ matrix with rank $r$, and compute the reduction factor for several realistic $(d,k,r)$ combinations drawn from real LLM architectures.
**Coding:** Implement the weight-merging operation (Section 5's `merge_weights`) and verify that inference using the merged layer produces numerically identical output to inference using the separate (unmerged) LoRA-augmented layer.

---

## 14. Mini Project

**Fine-tune an open-weight model via QLoRA for an actuarial-domain application**: using Hugging Face `peft` + `bitsandbytes`, load a small-to-medium open-weight model in 4-bit quantization, apply LoRA adapters to its attention layers, fine-tune on an actuarial/insurance instruction or preference dataset (extending Lesson 6's mini project), measure GPU memory usage throughout training, and compare the trainable-parameter percentage and final task performance against what full fine-tuning would require — producing a genuinely practical demonstration of parameter-efficient fine-tuning applied to your own domain.

---

## 15. Interview Preparation

- Explain the low-rank hypothesis underlying LoRA and why it enables such a large parameter-count reduction.
- Why is the $B$ matrix initialized to zero in LoRA?
- How does QLoRA combine quantization with LoRA, and why does quantizing the frozen base model not hurt training stability?
- What are the tradeoffs between merging LoRA weights into the base model versus keeping them as a separate, swappable adapter at inference time?

---

## 16. Summary

LoRA exploits the empirical observation that fine-tuning updates to large pretrained models tend to have low intrinsic rank, replacing full weight-matrix updates with a much smaller trainable low-rank factorization ($BA$, with $B$ zero-initialized for a clean, pretrained-identical starting point) — reducing trainable parameters and, critically, optimizer memory overhead by orders of magnitude while matching full fine-tuning's quality on most practical tasks. QLoRA extends this by quantizing the (frozen, gradient-free) base model to 4-bit precision, since numerical error there doesn't compound through training the way it would for actively-updated weights — together, these techniques are precisely what makes fine-tuning large open-weight models economically accessible outside major labs, directly relevant to any domain-specific LLM customization project you undertake.

---

## 17. References

- Hu et al. — "LoRA: Low-Rank Adaptation of Large Language Models" (2021, the original LoRA paper)
- Dettmers et al. — "QLoRA: Efficient Finetuning of Quantized LLMs" (2023, the original QLoRA paper)
- Hugging Face — `peft` library documentation
- Aghajanyan et al. — "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020, the empirical low-rank-updates finding that directly motivated LoRA)
