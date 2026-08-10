# Phase 6 · Lesson 8 — Quantization

> Prerequisite: LoRA & QLoRA (Lesson 7), Phase 1 Lesson 1 (floating-point representation)

---

## 1. Introduction

### What is quantization?
The process of representing a model's weights (and sometimes activations) using fewer bits than the standard 32-bit or 16-bit floating-point formats used during training — commonly 8-bit or 4-bit integers — dramatically reducing memory footprint and often improving inference speed, at the cost of some numerical precision.

### Why does it exist?
A 70-billion-parameter model in 16-bit precision requires roughly 140GB just to store the weights — far beyond a single consumer GPU's memory, and a substantial cost even on data-center hardware at scale. Quantization exists to shrink this footprint (commonly 2-4× via 8-bit or 4-bit representation) enabling larger models to run on more modest hardware, faster inference (lower-precision arithmetic can be computed faster on modern hardware with dedicated support), and lower serving costs at production scale.

### Historical background
Quantization has a long history in classical signal processing and embedded systems (representing continuous signals with finite precision has always required some form of quantization). Its application to neural networks accelerated through the 2010s-2020s alongside the broader efficient-deep-learning research area, with LLM-specific quantization techniques (GPTQ, 2022; AWQ, 2023; and QLoRA's NF4 format, Lesson 7) emerging specifically to address the unique challenges of quantizing extremely large Transformer models with minimal quality loss.

### Real-world motivation
Quantization is precisely what makes it possible to run a capable LLM on a laptop, a phone, or a single affordable GPU rather than requiring a data-center cluster — directly relevant to any local/on-device deployment scenario, and to the cost economics of serving LLMs at scale in production (Phase 8).

---

## 2. Theory

### The core idea: mapping a continuous range to a discrete grid
Quantization maps floating-point values (continuous, high-precision) to a smaller set of discrete values (representable in fewer bits) via a scale factor and (optionally) a zero-point offset:
$$
x_{quantized} = \text{round}\left(\frac{x}{\text{scale}}\right), \qquad x_{dequantized} = x_{quantized} \times \text{scale}
$$
The **scale** is chosen based on the actual range of values being quantized (e.g., a weight tensor's min/max), so that the available discrete levels are used as effectively as possible for that specific tensor's value distribution.

### Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)
- **PTQ**: quantize an already-trained model's weights directly, with no further training — fast, simple, the standard approach for most LLM deployment scenarios.
- **QAT**: simulate quantization's effects *during* training (or fine-tuning), letting the model adapt its weights to be more robust to the eventual precision reduction — generally yields better quality at a given bit-width, at the cost of requiring a full (re-)training process rather than a quick post-hoc conversion.

### Symmetric vs. asymmetric quantization
- **Symmetric**: the quantization range is centered at zero (scale only, no zero-point offset) — simpler, works well when the value distribution is roughly symmetric around zero (common for weights).
- **Asymmetric**: uses both a scale and a zero-point offset, better suited for value distributions that aren't centered at zero (sometimes true of activations, which can have a skewed distribution after certain non-linearities).

### Per-tensor vs. per-channel (or per-group) quantization
Using a *single* scale factor for an entire weight matrix ("per-tensor") is simple but can be suboptimal if different rows/columns have very different value ranges; **per-channel** (a separate scale per output channel/row) or even finer **per-group** quantization (separate scales for small groups of consecutive weights) preserves more precision at a modest storage overhead (a small number of extra scale factors), a standard refinement in modern LLM quantization methods (GPTQ, AWQ).

---

## 3. Mathematical Foundations

### Quantization error and its sources
For a value $x$ quantized with scale $s$: $x_{dequantized} = \text{round}(x/s) \times s$, introducing a rounding error bounded by $|x - x_{dequantized}| \le s/2$. Total model quality degradation from quantization is the *aggregate* effect of this per-weight error across millions/billions of parameters — individually tiny, but the aggregate effect on model behavior is exactly what quantization research works to minimize through smarter scale/grouping choices, not just naive uniform rounding.

### Why weight distributions matter (outlier-aware quantization)
LLM weight (and especially activation) distributions are often **not** uniform — a small number of "outlier" values can have much larger magnitude than the bulk of the distribution. Naive uniform quantization must set its scale to accommodate these outliers, wasting most of the discrete levels on a narrow, densely-populated central region and representing it coarsely — a real, quantifiable quality problem. Modern techniques (AWQ specifically) identify and preserve precision for the *most important* weight channels (informed by activation magnitude, not just weight magnitude) rather than treating all weights uniformly.

### GPTQ's approach (a principled, error-compensating quantization method)
GPTQ (Frantar et al., 2022) quantizes weights **one at a time** (or in small groups), and after quantizing each weight, adjusts the *remaining, not-yet-quantized* weights slightly to compensate for the error just introduced — using second-order (Hessian-based, Phase 3 Lesson 2) information about the loss landscape to make this compensation as effective as possible, achieving much better quality at aggressive bit-widths (e.g., 4-bit or even lower) than naive independent-per-weight rounding.

### NF4 (NormalFloat4) — QLoRA's specialized 4-bit format
Rather than uniform quantization levels, NF4 (used in QLoRA, Lesson 7) chooses quantization levels specifically optimized for weights that follow a roughly Gaussian/normal distribution (empirically, a good match for typical neural network weight distributions after appropriate normalization) — placing more discrete levels where the distribution has more probability mass, a direct application of optimal quantizer design theory (information-theoretically, this minimizes expected quantization error for a known/assumed value distribution, echoing Phase 3 Lesson 6's entropy-optimal coding intuitions).

---

## 4. Algorithm — Simple Per-Channel Post-Training Quantization (fully specified)

```
GIVEN a weight matrix W (d_out x d_in), target bit-width b (e.g., 8 for INT8):
n_levels = 2^b                                    # e.g., 256 discrete levels for 8-bit
FOR each output channel (row) i of W:
    row = W[i, :]
    max_abs = max(abs(row))                        # per-channel scale, NOT a single global scale
    scale_i = max_abs / (n_levels / 2 - 1)          # symmetric quantization (Section 2)
    FOR each value x in row:
        quantized_value = round(x / scale_i)
        CLIP quantized_value to [-(n_levels/2 - 1), n_levels/2 - 1]
    STORE the quantized (low-bit-width integer) row AND scale_i (one float per row -- small overhead)

DEQUANTIZATION (at inference time, before/during the matrix multiplication):
    W_dequantized[i, :] = quantized_row[i, :] * scale_i   # reconstruct an approximate float value
```

---

## 5. Python Implementation

```python
"""quantization_core.py — from-scratch symmetric per-channel quantization"""
import numpy as np


def quantize_per_channel(W: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric, per-channel (per-row) quantization. Returns (quantized_int_array, scales)."""
    n_levels = 2 ** bits
    max_level = n_levels // 2 - 1
    scales = np.max(np.abs(W), axis=1, keepdims=True) / max_level
    scales = np.where(scales == 0, 1e-8, scales)          # avoid division by zero for all-zero rows
    quantized = np.round(W / scales).clip(-max_level - 1, max_level)
    return quantized.astype(np.int8 if bits <= 8 else np.int16), scales


def dequantize(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float32) * scales


def quantization_error_report(W: np.ndarray, bits_options: list[int] = [8, 4, 2]) -> None:
    for bits in bits_options:
        quantized, scales = quantize_per_channel(W, bits=bits)
        reconstructed = dequantize(quantized, scales)
        rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
        compression_ratio = 32 / bits    # comparing against a float32 baseline
        print(f"{bits}-bit: relative error={rel_error:.4f}, "
              f"compression ratio={compression_ratio:.1f}x, "
              f"memory for this tensor: {W.size * bits / 8 / 1024:.1f} KB "
              f"(vs {W.size * 32 / 8 / 1024:.1f} KB at float32)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate a realistic weight matrix WITH a few outlier values (Section 3's real-world concern)
    W = rng.normal(0, 0.02, size=(512, 512))
    outlier_mask = rng.random(W.shape) < 0.001
    W[outlier_mask] *= 50                                  # inject rare, large-magnitude outliers

    quantization_error_report(W)
```

**Expected finding:** the outliers (even though rare) force each affected row's `scale` to be large, coarsening precision for the *entire* row's non-outlier values — a concrete, runnable demonstration of exactly why outlier-aware methods (AWQ, Section 3) meaningfully outperform naive uniform quantization on real LLM weight distributions.

---

## 6. Build From Scratch

**A simplified GPTQ-style sequential, error-compensating quantization (to make Section 3's key idea concrete):**
```python
import numpy as np

def sequential_error_compensating_quantize(W_row: np.ndarray, bits: int = 4) -> np.ndarray:
    """Quantizes one row's values ONE AT A TIME, redistributing each step's rounding error
    to the NOT-YET-QUANTIZED remaining values -- a simplified illustration of GPTQ's core idea
    (real GPTQ uses Hessian information for optimal error redistribution; this is a naive version)."""
    n_levels = 2 ** bits
    max_level = n_levels // 2 - 1
    working_row = W_row.copy()
    scale = np.max(np.abs(working_row)) / max_level
    quantized = np.zeros_like(working_row)

    for i in range(len(working_row)):
        original_value = working_row[i]
        q = np.clip(np.round(original_value / scale), -max_level - 1, max_level)
        quantized[i] = q
        error = original_value - (q * scale)               # THE rounding error just introduced
        if i + 1 < len(working_row):
            working_row[i+1:] += error / (len(working_row) - i - 1)   # naive uniform error redistribution

    return quantized, scale

rng = np.random.default_rng(0)
row = rng.normal(0, 1, size=20)
naive_q, naive_scale = quantize_per_channel(row.reshape(1, -1), bits=4)
compensated_q, comp_scale = sequential_error_compensating_quantize(row, bits=4)

naive_error = np.linalg.norm(row - dequantize(naive_q, naive_scale).flatten())
comp_error = np.linalg.norm(row - compensated_q * comp_scale)
print(f"Naive quantization error: {naive_error:.4f}")
print(f"Error-compensating quantization error: {comp_error:.4f}")
```
This is a deliberately simplified illustration (real GPTQ uses genuine second-order/Hessian information for principled error redistribution, not naive uniform spreading) — but it directly demonstrates the core insight: compensating for each quantization decision's error in the *remaining* values reduces aggregate error compared to treating each value's quantization as fully independent.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `quantize_per_channel`/`dequantize` | `bitsandbytes` — production 8-bit and 4-bit (including NF4) quantization, tightly integrated with `transformers`/`peft` |
| `sequential_error_compensating_quantize` (naive) | `auto-gptq`/`optimum`'s GPTQ implementations — genuine Hessian-informed error compensation, extensively validated at real LLM scale |
| No outlier-aware quantization shown | `AutoAWQ` — implements Activation-aware Weight Quantization, specifically preserving precision for weight channels identified as most important via activation statistics |

---

## 8. Visual Explanations

**Quantization as mapping a continuous range onto a discrete grid:**
```
Continuous values:  ──●────●──●───────●────●─── (float32, effectively infinite precision)
                       │    │  │       │    │
                       ▼    ▼  ▼       ▼    ▼
Quantized grid:     [-7][-3][ 0][ 3][ 7]     (INT4 example: only 16 discrete levels, -8 to 7)
   (each continuous value SNAPS to its nearest available discrete level)
```

**Outlier impact on per-channel scale (why outlier-aware methods matter):**
```
Row WITHOUT outlier:                     Row WITH one outlier:
values: [-1, 0.5, 0.8, -0.3, 0.9]         values: [-1, 0.5, 0.8, -0.3, 50]  <- one outlier!
scale = max(|values|)/max_level            scale = max(|values|)/max_level
      = SMALL (fine-grained levels)              = LARGE (forced by the outlier -- COARSE levels for everyone else)
   (the outlier forces coarse quantization for ALL other, otherwise well-behaved, values in that row)
```

---

## 9. Practical Examples

**Simple:** implement `quantize_per_channel` (Section 5) and verify the reconstruction error decreases as bit-width increases (8-bit better than 4-bit better than 2-bit).
**Medium:** inject synthetic outliers into a weight matrix (Section 5) and empirically demonstrate how they degrade per-channel quantization quality compared to an outlier-free matrix.
**Real-world:** use `bitsandbytes` to load a real open-weight model in 8-bit and separately in 4-bit precision, compare memory usage (via `torch.cuda.memory_allocated()` or similar) and qualitatively assess output quality on a few test prompts against the full-precision baseline.

---

## 10. Real Industry Use Cases

- **QLoRA (Lesson 7)**: relies directly on NF4 quantization of the frozen base model, precisely the technique covered in Section 3.
- **Local/on-device LLM deployment** (e.g., running a 7B-13B parameter model on a laptop or phone via tools like `llama.cpp`): almost universally uses 4-bit or lower quantization to fit within consumer hardware memory constraints.
- **Production LLM serving cost optimization**: major API providers routinely use quantization (alongside other techniques, Phase 8) to reduce serving costs and increase throughput per GPU, especially for high-volume, latency-sensitive endpoints.
- **GPTQ/AWQ-quantized model releases**: it's now standard practice for popular open-weight model releases to be accompanied by community- or vendor-provided GPTQ/AWQ-quantized versions, specifically to broaden hardware accessibility.

---

## 11. Common Mistakes

- Using a single global scale factor for an entire weight matrix (rather than per-channel/per-group) — significantly worse precision preservation than the now-standard per-channel or per-group approaches, for negligible additional implementation complexity.
- Ignoring outliers when choosing quantization scale — a small number of large-magnitude values can force coarse quantization across an entire row/channel, a real and well-documented quality problem naive quantization schemes suffer from.
- Assuming all bit-widths degrade model quality proportionally — in practice, 8-bit quantization is often nearly lossless for many models/tasks, while pushing to 4-bit or below requires much more careful technique (GPTQ, AWQ, or QLoRA's NF4) to avoid substantial quality degradation.
- Quantizing activations naively without considering their (often different, sometimes more skewed) distribution compared to weights — activation quantization is generally a harder problem than weight-only quantization.

---

## 12. Best Practices (2026)

- Use established, well-validated quantization libraries (`bitsandbytes`, `auto-gptq`, `AutoAWQ`) rather than naive from-scratch quantization for any real deployment — these implement the outlier-aware and error-compensating refinements genuinely needed for good quality at aggressive bit-widths.
- Start with 8-bit quantization as a low-risk, often nearly-lossless default; move to 4-bit specifically when memory constraints require it, using a proper technique (GPTQ/AWQ/NF4) rather than naive uniform quantization.
- Always empirically evaluate quantized model quality on representative downstream tasks (Phase 4 Lesson 3's evaluation discipline) rather than assuming a quantization method's general reputation guarantees acceptable quality for your specific use case.
- For fine-tuning under memory constraints, combine quantization with LoRA (i.e., QLoRA, Lesson 7) rather than choosing between quantization and parameter-efficient fine-tuning — they compose naturally and address complementary bottlenecks.

---

## 13. Exercises

**Easy:** Implement per-tensor (single global scale) quantization and compare its reconstruction error against per-channel quantization (Section 5) on the same weight matrix.
**Medium:** Systematically vary the fraction and magnitude of injected outliers in a synthetic weight matrix and empirically characterize how quantization error degrades as outlier severity increases.
**Hard:** Implement the simplified error-compensating quantization (Section 6) and compare its aggregate error against naive independent quantization across many different rows/matrices, characterizing when the compensation provides the most benefit.
**Mathematical:** Derive the maximum possible quantization error ($s/2$, Section 3) for a given scale $s$, and compute the expected mean-squared quantization error under a uniform-distribution assumption for the rounding error.
**Coding:** Implement asymmetric quantization (using both a scale and a zero-point) and compare its precision against symmetric quantization on a weight distribution that is deliberately NOT centered at zero.

---

## 14. Mini Project

Build a **quantization quality/efficiency benchmarking tool**: using `bitsandbytes` (or `auto-gptq`), quantize a small open-weight model at 8-bit and 4-bit precision, measure and report memory footprint reduction, inference latency change, and quality degradation (via perplexity, Lesson 5, on a held-out text sample, and/or task-specific accuracy on a small benchmark) at each precision level — producing a genuinely practical, quantitative report on the precision/efficiency tradeoff for a real model, directly informing deployment decisions for resource-constrained applications.

---

## 15. Interview Preparation

- Explain the basic mechanism of quantization (scale, zero-point) and the difference between symmetric and asymmetric quantization.
- Why does per-channel quantization generally outperform per-tensor quantization?
- How do outliers in weight distributions specifically harm naive quantization, and what techniques address this?
- Explain, at a conceptual level, how GPTQ's error-compensation approach improves on naive independent-weight quantization.

---

## 16. Summary

Quantization reduces model memory footprint and often improves inference speed by representing weights with fewer bits than training precision, at the cost of some numerical error whose real-world impact depends heavily on *how* quantization is done — naive uniform, per-tensor quantization is simple but vulnerable to outlier-driven precision loss, while modern techniques (per-channel/per-group scaling, GPTQ's Hessian-informed error compensation, AWQ's activation-aware outlier handling, and QLoRA's NF4 format) achieve dramatically better quality at aggressive bit-widths like 4-bit. Combined with LoRA (Lesson 7), quantization is precisely what makes running and fine-tuning capable LLMs accessible on consumer and prosumer hardware rather than requiring data-center-scale infrastructure — a direct, practical enabler for any resource-constrained LLM project.

---

## 17. References

- Frantar et al. — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al. — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Dettmers et al. — "QLoRA: Efficient Finetuning of Quantized LLMs" (2023, introducing the NF4 format)
- Gholami et al. — "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021, comprehensive background)
