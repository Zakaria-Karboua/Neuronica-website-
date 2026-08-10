# Phase 6 · Lesson 5 — Pretraining

> Prerequisite: Lessons 1–4 (this phase), Phase 3 Optimization, Phase 4 Lesson 7 (RL Introduction, for RLHF context)

---

## 1. Introduction

### What is pretraining?
The initial, massively data- and compute-intensive training phase in which a Transformer (Phase 5 Lesson 7, refined per Lesson 4) learns general language modeling capability from a large, broad text corpus, via a simple **self-supervised** objective — predicting the next token given all previous tokens — before any task-specific fine-tuning (Lesson 6) occurs.

### Why does it exist?
Labeled data for every conceivable downstream task doesn't exist at the scale needed to train a model from random initialization for each task individually — but raw text (books, web pages, code) exists in effectively unlimited quantity, and next-token prediction requires no human labeling at all, since every piece of text is its own supervision signal (the actual next word). Pretraining exploits this abundance to learn broad linguistic, factual, and reasoning capability, which fine-tuning (Lesson 6) then specializes for particular tasks/behaviors at a fraction of the pretraining compute cost.

### Historical background
GPT-1 (2018) demonstrated that generative pretraining followed by task-specific fine-tuning outperformed training from scratch on each task. GPT-2 (2019) and GPT-3 (2020) showed that simply *scaling* pretraining (more data, more parameters, more compute) produced qualitatively new capabilities (few-shot learning, emergent reasoning) without architectural changes — an empirical trend formalized by the "scaling laws" (Kaplan et al., 2020; Hoffmann et al., 2022's "Chinchilla" refinement) that continue to guide how frontier labs allocate compute between model size and data quantity in 2026.

### Real-world motivation
Every LLM you interact with (Claude, GPT-series, Llama) went through exactly this process first — understanding pretraining's objective, scale, and data considerations is what makes concepts like "hallucination," "knowledge cutoff," and "emergent capability" concrete mechanical facts rather than vague marketing terms.

---

## 2. Theory

### The next-token prediction objective, and why it's so powerful
Given a sequence $x_1, \dots, x_n$, the model learns to maximize:
$$
P(x_1,\dots,x_n) = \prod_{t=1}^{n} P(x_t | x_1,\dots,x_{t-1})
$$
— the chain rule of probability (Phase 3 Lesson 3), decomposing the joint probability of an entire text sequence into a product of next-token conditional probabilities. Crucially, predicting the next token well *requires*, implicitly, learning grammar, facts, reasoning patterns, and style — a single simple objective that incentivizes learning an enormous range of latent capabilities, without any of them being explicitly labeled or specified in advance.

### Scaling laws — the empirical relationship between compute, data, and performance
Kaplan et al. (2020) found that pretraining loss follows a smooth power-law relationship with model size, dataset size, and compute — larger/more-trained models reliably achieve lower loss, with no signs (at the scales studied) of the relationship breaking down. Hoffmann et al.'s Chinchilla paper (2022) refined this: for a *fixed* compute budget, there is an optimal *balance* between model size and training data quantity — many earlier large models (GPT-3) were, in retrospect, undertrained relative to their size, and a smaller model trained on proportionally more data can outperform a larger undertrained one at the same total compute cost — a genuinely important, widely-adopted practical finding shaping how compute is allocated in modern training runs.

### Data quality and composition (increasingly recognized as equally important as quantity)
Pretraining corpora combine web text (heavily filtered/deduplicated), books, code, and increasingly curated/synthetic data — the specific mixture and quality filtering substantially affects downstream capability, an area of active, closely-guarded research effort at every major lab, with "just add more raw web text" long since recognized as insufficient on its own for state-of-the-art results.

### Emergent capabilities — a genuinely debated phenomenon
Certain capabilities (multi-step arithmetic, chain-of-thought reasoning) appear to emerge somewhat abruptly at particular scale thresholds rather than improving smoothly — though there is legitimate, ongoing research debate about how much of this "emergence" reflects real discontinuities in the underlying model capability versus artifacts of how a discontinuous evaluation metric (e.g., exact-match accuracy) is measured over an actually-smooth underlying improvement — a nuance worth knowing rather than treating "emergence" as a settled, fully-understood phenomenon.

---

## 3. Mathematical Foundations

### The pretraining loss, exactly (directly reusing Phase 3 Lesson 6)
$$
L = -\frac{1}{n}\sum_{t=1}^{n} \log P(x_t | x_{<t}; \theta)
$$
This is precisely cross-entropy loss (Phase 3 Lesson 6) applied at every position in the sequence simultaneously — the average negative log-likelihood the model assigns to the *actual* next token at each position, exactly the same objective Phase 4 Lesson 1's logistic regression minimizes, just applied to a much larger vocabulary and repeated at every sequence position.

### Perplexity as the standard pretraining evaluation metric
$$
\text{Perplexity} = \exp(L)
$$
Directly reusing Phase 3 Lesson 6's exponentiated-cross-entropy interpretation: a perplexity of, say, 20 means the model is (on average) about as "confused" as if uniformly guessing among 20 equally likely next tokens at each position — lower is better, and comparing perplexity across models (on the same evaluation set, with the same tokenizer) is a standard, if incomplete, pretraining quality signal.

### Chinchilla scaling laws, the core relationship
$$
L(N, D) \approx E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$
where $N$ = model parameters, $D$ = training tokens, $E, A, B, \alpha, \beta$ are empirically fit constants, and $E$ represents an irreducible loss floor. For a fixed compute budget $C \approx 6ND$ (a standard approximation for Transformer training FLOPs), Chinchilla's key finding was that the loss-minimizing allocation is roughly $N \propto C^{0.5}$ and $D \propto C^{0.5}$ — model size and data should scale **proportionally** with available compute, not model size alone, correcting a real, practically consequential earlier industry tendency to over-invest in parameter count relative to training data.

### Learning rate schedules for pretraining (directly extending Phase 3 Lesson 5)
Pretraining runs universally use a **warmup** period (gradually increasing the learning rate from near-zero) followed by a **decay** schedule (commonly cosine decay) over the remainder of training — warmup prevents early, large, poorly-informed gradient updates (when the model is still near random initialization) from destabilizing training; decay lets the model settle into a more precise optimum as training progresses, directly applying Phase 3 Lesson 5's optimization landscape reasoning at the scale of a training run spanning potentially trillions of tokens.

---

## 4. Algorithm — The Pretraining Loop (fully specified, conceptual)

```
GIVEN a large tokenized corpus, model architecture (Phase 5 Lesson 7 + Lesson 4 refinements):
INITIALIZE model parameters (He/Xavier-style initialization, Phase 5 Lesson 1)
SET UP learning rate schedule: linear warmup for W steps, then cosine decay to near-zero over remaining steps
FOR each training step (processing one large batch of token sequences):
    1. SAMPLE a batch of sequences from the corpus (each sequence: a contiguous chunk of tokens)
    2. FORWARD PASS: compute next-token logits at every position (Phase 5 Lesson 7)
    3. COMPUTE LOSS: cross-entropy between predicted logits and actual next tokens, averaged over
       ALL positions and ALL sequences in the batch
    4. BACKWARD PASS: compute gradients via backpropagation (Phase 5 Lesson 2), automated via autograd (Lesson 8)
    5. GRADIENT CLIPPING (Phase 5 Lesson 4's technique) -- essential at this scale for stability
    6. OPTIMIZER STEP: update parameters via AdamW (Phase 3 Lesson 5), using the CURRENT scheduled learning rate
    7. PERIODICALLY: evaluate perplexity on a held-out validation set, checkpoint model weights
CONTINUE until the planned token/compute budget (informed by Chinchilla-style scaling law calculations) is exhausted
```

---

## 5. Python Implementation

```python
"""pretraining_core.py — a small-scale, illustrative pretraining loop"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)                      # LINEAR WARMUP
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))                        # COSINE DECAY
    return LambdaLR(optimizer, lr_lambda)


def pretrain_step(model: nn.Module, batch: torch.Tensor, optimizer, scheduler, device: str = "cpu") -> float:
    """batch: (batch_size, seq_len) of token IDs. Predicts token[t+1] from token[0..t]."""
    model.train()
    batch = batch.to(device)
    inputs, targets = batch[:, :-1], batch[:, 1:]     # classic "shift by one" next-token setup

    optimizer.zero_grad()
    logits = model(inputs)                              # (batch, seq_len-1, vocab_size)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Phase 5 Lesson 4's gradient clipping
    optimizer.step()
    scheduler.step()
    return loss.item()


def compute_perplexity(model: nn.Module, val_loader, device: str = "cpu") -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)   # Section 3's perplexity formula


# Illustrative usage (assuming `model` is a Phase 5/6-style decoder-only Transformer)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=100_000)
# for batch in train_loader:
#     loss = pretrain_step(model, batch, optimizer, scheduler, device="cuda")
```

---

## 6. Build From Scratch

**A minimal empirical demonstration of the Chinchilla scaling relationship (fitting the power-law form to synthetic data, to make Section 3's abstract formula concrete):**
```python
import numpy as np
from scipy.optimize import curve_fit

def chinchilla_loss(ND, E, A, alpha, B, beta):
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

# Synthetic "experimental" data: (model_size, data_size) -> observed loss, following a KNOWN true law
rng = np.random.default_rng(0)
true_params = {"E": 1.5, "A": 400.0, "alpha": 0.35, "B": 200.0, "beta": 0.28}
N_vals = np.array([1e7, 1e8, 1e9, 1e10] * 4)
D_vals = np.repeat([1e8, 1e9, 1e10, 1e11], 4)
true_loss = chinchilla_loss((N_vals, D_vals), **true_params)
observed_loss = true_loss + rng.normal(0, 0.02, size=len(true_loss))   # add realistic measurement noise

fit_params, _ = curve_fit(chinchilla_loss, (N_vals, D_vals), observed_loss, p0=[1, 100, 0.3, 100, 0.3])
print("Fitted parameters:", dict(zip(["E", "A", "alpha", "B", "beta"], fit_params.round(3))))
print("True parameters:  ", true_params)
```
This mirrors, in simplified form, exactly the empirical methodology real scaling-law papers use: run many training experiments at varying $(N,D)$, fit the power-law form, then use the fitted relationship to predict the compute-optimal allocation for a *new*, larger compute budget than any individual experiment used.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| Manual pretraining loop | Distributed training frameworks (`torch.distributed`, DeepSpeed, Megatron-LM) — handle multi-GPU/multi-node parallelism, essential at real pretraining scale (thousands of GPUs) |
| Manual cosine schedule | `transformers.get_cosine_schedule_with_warmup` — identical formula, standard library utility |
| Small-scale illustrative loop | Real pretraining involves highly optimized data pipelines, checkpointing/fault-tolerance across long-running multi-week jobs, and careful mixed-precision (Phase 5 Lesson 8) training — engineering complexity far beyond this lesson's illustrative scope |

---

## 8. Visual Explanations

**Next-token prediction as the core self-supervised objective:**
```
Input:  "The  cat  sat  on  the"
Target: "cat  sat  on   the  mat"    <- EVERY position's target is just the NEXT token in the SAME sequence
   (no human labeling needed -- the raw text IS its own supervision, at every single position)
```

**Learning rate schedule (warmup + cosine decay):**
```
LR
 │      ___________
 │     /            ‾‾‾--___
 │    /                      ‾‾‾--___
 │   /                                ‾‾‾--_
 │  /                                        ‾
 └─────────────────────────────────────────────  training step
    warmup        cosine decay over remaining training
```

**Chinchilla-optimal allocation (schematic — model size vs data size for fixed compute):**
```
Loss
 │  ●  (too small model, too much data -- undertrained model capacity, wasted data)
 │      ●  (too large model, too little data -- undertrained MODEL, "GPT-3 mistake")
 │           ★ (Chinchilla-optimal: N and D both scaled proportionally to compute budget)
 └──────────────────────────  (N, D) combinations at FIXED total compute
```

---

## 9. Practical Examples

**Simple:** implement the cosine warmup+decay schedule (Section 5) and plot the resulting learning rate curve across training steps.
**Medium:** compute perplexity for a small pretrained model on a held-out text sample and interpret the resulting number using Section 3's "confused among N choices" intuition.
**Real-world:** using published scaling-law coefficients (from the Chinchilla paper or subsequent work), estimate the Chinchilla-optimal ratio of training tokens to parameters for a hypothetical compute budget, and compare it against the actual token/parameter ratios of several well-known open-weight models (whose training details are often published) to assess whether they were over/under-trained relative to that guideline.

---

## 10. Real Industry Use Cases

- **Every frontier LLM's initial training phase** (GPT-series, Claude, Llama, Gemini): pretraining is the single most compute-expensive phase of an LLM's development, often costing tens of millions of dollars in compute for the largest models.
- **Chinchilla scaling laws' industry impact**: directly influenced subsequent model releases (Llama's relatively smaller parameter counts trained on proportionally much more data compared to earlier-era models) once the paper's findings were widely adopted.
- **Data curation as competitive advantage**: the specific composition, filtering, and deduplication of pretraining data is now recognized as comparably important to raw compute scale, and is an area of significant, often proprietary, research investment at every major lab.
- **Knowledge cutoff dates**: directly determined by when a model's pretraining corpus was collected — a genuinely mechanical fact about pretraining, not an arbitrary limitation.

---

## 11. Common Mistakes

- Assuming "bigger model = always better" without accounting for the Chinchilla-style compute-optimal balance between model size and training data quantity.
- Treating perplexity as a complete measure of model quality — it measures next-token prediction confidence on a specific evaluation distribution, but doesn't directly capture downstream task performance, instruction-following ability, or safety properties (which fine-tuning, Lesson 6, and RLHF specifically target).
- Underestimating the importance of data quality/curation, assuming "more raw web text" alone drives capability improvements as reliably as model/data scale increases do.
- Confusing "emergent capability" claims with fully settled scientific fact — a genuinely still-debated research area regarding measurement artifacts versus real capability discontinuities.

---

## 12. Best Practices (2026)

- Use published scaling-law relationships to inform model size/data quantity decisions for any new large-scale pretraining effort, rather than defaulting to "as large as compute allows" without considering the data-quantity side of the tradeoff.
- Invest meaningfully in data quality, deduplication, and filtering — recognized industry-wide as comparably important to raw scale for achieving state-of-the-art results.
- Always evaluate perplexity alongside (not instead of) downstream task benchmarks — a full evaluation suite (Phase 4 Lesson 3's rigor, applied to LLM-specific benchmarks) is necessary to actually characterize model quality.
- Use warmup + cosine (or similar) decay learning rate schedules as a well-established default for any large-scale training run.

---

## 13. Exercises

**Easy:** Implement and plot the cosine warmup+decay learning rate schedule (Section 5) for a specified number of warmup and total steps.
**Medium:** Compute the perplexity of a small pretrained model on two different evaluation text samples (e.g., one similar to its training distribution, one quite different) and compare the results, interpreting what the difference implies.
**Hard:** Fit a simplified 2-parameter power-law scaling relationship (loss vs. model size alone, holding data roughly fixed) to a small set of your own from-scratch model training runs (Phase 5 Lesson 7's `MiniTransformer` at a few different sizes) and discuss how well (or poorly) it matches the smooth power-law behavior described in the scaling laws literature.
**Mathematical:** Derive, from the Chinchilla loss formula (Section 3), the compute-optimal scaling exponents for $N$ and $D$ under a fixed compute budget $C \approx 6ND$, using Lagrangian optimization (Phase 3 Lesson 5).
**Coding:** Implement gradient accumulation (processing several mini-batches before calling `optimizer.step()`, Phase 5 Lesson 8's Section 3) to simulate training with an effectively larger batch size than fits in available memory, and verify it produces equivalent results to true large-batch training on a small example.

---

## 14. Mini Project

**Fully pretrain the Phase 5/6 `MiniTransformer` (extended with Lesson 4's modern architectural refinements) on a modestly-sized real text corpus**: implement the complete pretraining loop (Section 5) with proper warmup+cosine learning rate scheduling and gradient clipping, track training and validation perplexity across training steps, generate sample text at several checkpoints to qualitatively observe capability improving over training, and write a short report relating your empirical loss curve's shape to the smooth power-law behavior described by scaling laws — a genuine, hands-on (if small-scale) rehearsal of the exact process underlying every production LLM's development.

---

## 15. Interview Preparation

- Explain the next-token prediction pretraining objective and why it requires no human labeling.
- What are scaling laws, and what did the Chinchilla paper specifically revise about earlier scaling assumptions?
- Why is perplexity used as a pretraining evaluation metric, and what are its limitations?
- Explain why pretraining uses a learning rate warmup period before decay.

---

## 16. Summary

Pretraining teaches a Transformer general language capability via the simple, label-free next-token prediction objective (exactly cross-entropy loss, Phase 3 Lesson 6, applied at every sequence position), with perplexity as the standard evaluation metric and empirically-derived scaling laws (culminating in the Chinchilla finding that model size and training data should scale proportionally under a fixed compute budget) guiding how massive training runs allocate resources. Every downstream capability of a modern LLM — and every limitation, from knowledge cutoffs to occasional factual errors — traces back to decisions made at exactly this stage: what data was included, how much compute was spent, and how that compute was balanced between model size and data quantity, setting the foundation that Lesson 6's fine-tuning and RLHF then specialize into a genuinely helpful, aligned assistant.

---

## 17. References

- Radford et al. — "Improving Language Understanding by Generative Pre-Training" (2018, GPT-1)
- Kaplan et al. — "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al. — "Training Compute-Optimal Large Language Models" (2022, the Chinchilla paper)
- Wei et al. — "Emergent Abilities of Large Language Models" (2022, and subsequent critical responses debating measurement artifacts)
