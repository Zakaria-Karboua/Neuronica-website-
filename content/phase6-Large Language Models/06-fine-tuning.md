# Phase 6 · Lesson 6 — Fine-Tuning

> Prerequisite: Pretraining (Lesson 5), Phase 4 Lesson 7 (RL Introduction)

---

## 1. Introduction

### What is fine-tuning?
The process of further training a pretrained model (Lesson 5) on a smaller, more specific dataset to specialize its behavior — adapting general language capability into a particular task, domain, or interaction style. This lesson covers the full spectrum: supervised fine-tuning (SFT) on labeled examples, instruction tuning (teaching a model to follow instructions), and RLHF (aligning a model's behavior using human preference signals, directly building on Phase 4 Lesson 7's RL foundations).

### Why does it exist?
A raw pretrained model (Lesson 5) is a powerful but *uncalibrated* next-token predictor — it will happily continue text in whatever style/direction the input suggests, with no inherent tendency toward being helpful, honest, or safe, and no natural format for "following an instruction" versus simply continuing a document. Fine-tuning exists specifically to reshape this raw capability into the behavior actually wanted from a production assistant.

### Historical background
GPT-1/2's original fine-tuning was straightforward supervised adaptation to specific downstream tasks (classification, etc.). The pivotal shift was InstructGPT (Ouyang et al., 2022), which introduced the now-standard three-stage pipeline — supervised fine-tuning on instruction-following demonstrations, training a reward model on human preference comparisons, then optimizing the policy against that reward model via PPO (Phase 4 Lesson 7) — the direct ancestor of the alignment process behind ChatGPT, Claude, and every modern instruction-following LLM.

### Real-world motivation
The difference between a raw pretrained model and Claude/ChatGPT-style assistants is almost entirely this lesson's content — understanding it demystifies why models follow instructions, refuse certain requests, and behave conversationally rather than simply auto-completing text.

---

## 2. Theory

### Supervised Fine-Tuning (SFT)
Continue training the pretrained model on a curated dataset of (instruction, ideal response) pairs, using the exact same next-token-prediction / cross-entropy loss as pretraining (Lesson 5) — but now the "correct" tokens are a human-written or human-vetted ideal response, not just naturally occurring web text. This single step already dramatically improves instruction-following behavior, even before any RL stage.

### Reward Modeling
Collect human preference data: given a prompt and two (or more) candidate model responses, a human labeler indicates which is better. Train a **reward model** — typically initialized from the SFT model itself, with a new output head — to predict this preference, using the Bradley-Terry formulation directly introduced in Phase 4 Lesson 7:
$$
P(\text{response } A \succ B) = \frac{e^{r_\theta(A)}}{e^{r_\theta(A)} + e^{r_\theta(B)}}
$$
trained via cross-entropy loss (Phase 3 Lesson 6) on the human comparison labels — this stage is, notably, *ordinary supervised learning* (Phase 4 Lesson 1), not RL, a distinction Phase 4 Lesson 7 emphasized and now made fully concrete.

### RLHF (Reinforcement Learning from Human Feedback)
Using the trained reward model as the reward signal, further optimize the SFT model (now called the "policy") via a reinforcement learning algorithm — almost universally **PPO** (Proximal Policy Optimization, a more stable, practical descendant of Phase 4 Lesson 7's REINFORCE) — to generate responses that score highly according to the learned reward model, **while** a KL-divergence penalty (Phase 3 Lesson 6, directly reused) keeps the policy from drifting too far from the original SFT model, preventing the well-documented failure mode of "reward hacking" (finding degenerate outputs that fool the reward model without actually being good responses).

### DPO (Direct Preference Optimization) — a simpler modern alternative
Rather than the full three-stage pipeline (SFT → reward model → PPO), DPO (Rafailov et al., 2023) shows that the RLHF objective can be reformulated into a **single supervised loss** directly on preference pairs, mathematically equivalent to the RLHF optimum under certain conditions but without needing a separate reward model or the engineering complexity of PPO — an increasingly popular, simpler alternative widely adopted since 2023-2024, precisely because it achieves similar alignment quality with meaningfully less implementation complexity.

---

## 3. Mathematical Foundations

### The RLHF objective, precisely (directly extending Phase 4 Lesson 7)
$$
\max_\pi \; E_{x\sim D, y\sim\pi(\cdot|x)}[r_\theta(x,y)] - \beta \cdot D_{KL}(\pi(\cdot|x)\|\pi_{SFT}(\cdot|x))
$$
The first term maximizes expected reward (as judged by the learned reward model); the second term — a KL-divergence penalty (Phase 3 Lesson 6) weighted by $\beta$ — keeps the policy $\pi$ close to the original SFT model $\pi_{SFT}$, preventing the optimization from drifting into reward-model blind spots. $\beta$ is a genuine, consequential hyperparameter: too small allows excessive drift/reward-hacking, too large prevents meaningful improvement over the SFT model.

### DPO's derivation, sketched (why it avoids needing an explicit reward model)
Starting from the same RLHF objective above, one can show the optimal policy $\pi^*$ under this objective has a closed form directly relating $\pi^*$ to the reward function $r$:
$$
r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{SFT}(y|x)} + \beta\log Z(x)
$$
Substituting this relationship back into the Bradley-Terry preference model (Section 2) and simplifying (the intractable partition function $Z(x)$ cancels out when comparing two responses to the *same* prompt) yields a loss expressed **directly in terms of the policy $\pi$** — no separate reward model, no RL optimization loop, just a supervised classification-style loss over preference pairs:
$$
L_{DPO} = -\log\sigma\left(\beta\log\frac{\pi(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta\log\frac{\pi(y_l|x)}{\pi_{SFT}(y_l|x)}\right)
$$
where $y_w$ is the preferred ("winning") response and $y_l$ the dispreferred ("losing") one — a genuinely elegant mathematical simplification, not merely an engineering shortcut, that directly explains DPO's growing popularity.

### Catastrophic forgetting during fine-tuning
Fine-tuning on a narrow dataset risks degrading capabilities learned during broad pretraining (Lesson 5) — a direct manifestation of the bias-variance/overfitting concerns from Phase 4 Lesson 1, now at the scale of "forgetting general knowledge while specializing." Mitigations include: keeping the fine-tuning learning rate low relative to pretraining, mixing in some general-purpose data during fine-tuning, and (Lesson 7's territory) parameter-efficient fine-tuning methods that modify only a small fraction of the model's parameters, inherently limiting how much can be overwritten/forgotten.

---

## 4. Algorithm — The Full InstructGPT-Style Pipeline (fully specified)

```
STAGE 1 -- Supervised Fine-Tuning (SFT):
  GIVEN: pretrained model, dataset of (instruction, ideal_response) pairs
  TRAIN via standard next-token cross-entropy loss (Lesson 5's exact objective)
  RESULT: an SFT model, noticeably better at following instructions than the raw pretrained model

STAGE 2 -- Reward Model Training:
  GIVEN: prompts, MULTIPLE candidate responses per prompt (often generated by the SFT model itself),
         human preference labels (which response is better, for many pairs)
  TRAIN a reward model (initialized from the SFT model + a new scalar output head) via the
  Bradley-Terry cross-entropy loss (Section 2) -- THIS STAGE IS ORDINARY SUPERVISED LEARNING
  RESULT: a reward model r_theta(x, y) scoring any (prompt, response) pair

STAGE 3 -- RL Policy Optimization (PPO) [or: DPO as a simpler single-stage alternative to Stages 2+3]:
  GIVEN: the SFT model as the initial policy, the trained reward model
  FOR many iterations:
      generate responses from the CURRENT policy for a batch of prompts
      score each response using the reward model
      compute the KL penalty against the ORIGINAL SFT model (Section 3)
      update the policy via PPO to increase expected (reward - KL penalty)
  RESULT: the final, RLHF-aligned model (e.g., what ends up deployed as a production assistant)
```

---

## 5. Python Implementation

```python
"""fine_tuning_core.py — SFT loop and a from-scratch DPO loss implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sft_step(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, optimizer) -> float:
    """Identical mechanically to Lesson 5's pretraining step -- SFT is just next-token
    prediction on a curated (instruction, response) dataset instead of raw web text."""
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    # ignore_index=-100 is used to MASK OUT the instruction/prompt tokens --
    # we only want to compute loss on the RESPONSE tokens the model should learn to generate
    loss.backward()
    optimizer.step()
    return loss.item()


def dpo_loss(policy_logps_chosen: torch.Tensor, policy_logps_rejected: torch.Tensor,
              ref_logps_chosen: torch.Tensor, ref_logps_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Directly implements Section 3's DPO loss formula."""
    policy_logratios = policy_logps_chosen - policy_logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected
    logits = beta * (policy_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def compute_sequence_logprob(model: nn.Module, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sums log P(token) over just the RESPONSE portion of a sequence (masked via response_mask)."""
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * response_mask).sum(dim=-1)   # sum over sequence length, per-example


def dpo_training_step(policy_model, ref_model, batch: dict, optimizer, beta: float = 0.1) -> float:
    """batch contains: chosen_ids, rejected_ids, chosen_mask, rejected_mask (all pre-tokenized)."""
    policy_model.train()
    optimizer.zero_grad()

    policy_chosen_logp = compute_sequence_logprob(policy_model, batch["chosen_ids"], batch["chosen_mask"])
    policy_rejected_logp = compute_sequence_logprob(policy_model, batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():                                    # reference model is FROZEN (Section 3's pi_SFT)
        ref_chosen_logp = compute_sequence_logprob(ref_model, batch["chosen_ids"], batch["chosen_mask"])
        ref_rejected_logp = compute_sequence_logprob(ref_model, batch["rejected_ids"], batch["rejected_mask"])

    loss = dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=beta)
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## 6. Build From Scratch

**A minimal reward-model training loop (Stage 2, from scratch), directly implementing the Bradley-Terry loss:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """A pretrained/SFT model's backbone + a new scalar reward head."""
    def __init__(self, backbone: nn.Module, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(input_ids)         # assume backbone returns final hidden states
        last_token_hidden = hidden_states[:, -1, :]        # use the LAST token's representation
        return self.reward_head(last_token_hidden).squeeze(-1)   # scalar reward per sequence

def reward_model_training_step(reward_model: RewardModel, chosen_ids: torch.Tensor,
                                  rejected_ids: torch.Tensor, optimizer) -> float:
    reward_model.train()
    optimizer.zero_grad()
    r_chosen = reward_model(chosen_ids)
    r_rejected = reward_model(rejected_ids)
    # Bradley-Terry loss (Section 2): P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    loss.backward()
    optimizer.step()
    return loss.item()
```
This makes explicit that reward modeling really is just binary classification (via `logsigmoid`, mathematically identical to logistic regression's loss, Phase 4 Lesson 1) over which of two responses a human preferred — the "reinforcement learning" branding of RLHF applies specifically to Stage 3, not this stage.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `sft_step` | Hugging Face `SFTTrainer` (from the `trl` library) — handles prompt masking, packing, and efficient batching automatically |
| `RewardModel`/reward training loop | `trl`'s `RewardTrainer` — production-grade reward model training utilities |
| `dpo_training_step` | `trl`'s `DPOTrainer` — handles reference model management, efficient batched log-probability computation |
| No PPO implementation shown (substantial additional complexity) | `trl`'s `PPOTrainer` — implements the full PPO-for-RLHF pipeline, including KL penalty scheduling and value function training |

---

## 8. Visual Explanations

**The three-stage InstructGPT/RLHF pipeline:**
```
Pretrained Model ──▶ [SFT: train on ideal (instruction,response) pairs] ──▶ SFT Model
                                                                                 │
                                                            ┌────────────────────┴─────────────────┐
                                                            ▼                                        ▼
                                          [Train Reward Model on human preferences]      [Used as REFERENCE for
                                                            │                              KL penalty in Stage 3]
                                                            ▼                                        │
                                          [PPO: optimize SFT model against reward model, ◀───────────┘
                                           penalized by KL-divergence from the reference]
                                                            │
                                                            ▼
                                              Final RLHF-Aligned Model
```

**DPO as a shortcut bypassing the explicit reward model + PPO stages:**
```
Pretrained Model ──▶ [SFT] ──▶ SFT Model ──▶ [DPO: single supervised loss DIRECTLY on preference pairs,
                                                using the SFT model itself as the reference] ──▶ Aligned Model
                     (Stages 2 and 3 COLLAPSED into one simpler supervised training step)
```

---

## 9. Practical Examples

**Simple:** implement the SFT training step (Section 5) with proper prompt-masking (only computing loss on response tokens) and verify the masking is applied correctly on a toy example.
**Medium:** implement the reward model training loop (Section 6) on a small synthetic preference dataset and verify the trained reward model correctly ranks held-out preference pairs.
**Real-world:** using the `trl` library's `DPOTrainer`, fine-tune a small open-weight model on a preference dataset relevant to a specific behavior you want to encourage/discourage (e.g., more concise responses, or a specific domain style), and qualitatively evaluate the before/after behavior difference.

---

## 10. Real Industry Use Cases

- **Every production instruction-following LLM** (ChatGPT, Claude, and virtually all deployed assistants): built via exactly this SFT → (reward model + RLHF, or DPO) pipeline on top of a pretrained base model (Lesson 5).
- **Domain-specific fine-tuning**: companies fine-tune open-weight models (often via LoRA, Lesson 7) on proprietary data for specialized applications (legal document analysis, medical Q&A, customer support) — directly relevant to a potential SANAVIR or actuarial-domain application.
- **Constitutional AI and related alignment techniques**: extend/modify the basic RLHF pipeline with additional principles-based feedback mechanisms, an active area of alignment research at labs including Anthropic.
- **DPO's rapid industry adoption**: since 2023, many open-weight model releases (and increasingly, production systems) have shifted from full PPO-based RLHF to DPO or DPO-variant pipelines specifically for their reduced implementation complexity.

---

## 11. Common Mistakes

- Forgetting to mask the loss to only the *response* portion of an SFT training example — naively computing loss on the instruction/prompt tokens too teaches the model to "predict" the input rather than focusing learning signal on generating good responses.
- Setting the KL-penalty coefficient $\beta$ (Section 3) too low during RLHF — allows the policy to drift too far from the SFT model, exploiting reward model blind spots ("reward hacking") rather than genuinely improving response quality.
- Using an insufficiently diverse/representative human preference dataset for reward modeling — the reward model can only be as good (and as unbiased) as the preference judgments it was trained on.
- Fine-tuning with too high a learning rate or on too narrow a dataset — risking catastrophic forgetting of broad pretrained capabilities (Section 3).

---

## 12. Best Practices (2026)

- Consider DPO (or similar simplified preference-optimization methods) as a strong, meaningfully simpler default over full PPO-based RLHF for most fine-tuning applications, reserving full RLHF for cases where its specific advantages (more flexible reward specification, established track record at extreme scale) are genuinely needed.
- Always mask prompt/instruction tokens out of the SFT loss computation.
- Use a low learning rate and, where feasible, mix in some general-purpose data during any narrow-domain fine-tuning to mitigate catastrophic forgetting.
- Use parameter-efficient fine-tuning methods (Lesson 7's LoRA/QLoRA) by default for most fine-tuning tasks in 2026 — full fine-tuning of a large model's every parameter is increasingly reserved for cases with substantial compute budgets and a clear need for maximum capacity change.

---

## 13. Exercises

**Easy:** Implement prompt-masking for an SFT training example (mask the instruction tokens, keep the response tokens) and verify the loss is computed only over the intended tokens.
**Medium:** Implement the Bradley-Terry reward model loss (Section 6) and train a tiny reward model on a synthetic preference dataset, verifying it learns to rank preferred responses higher.
**Hard:** Implement the full DPO loss (Section 5) and train a small model on a synthetic preference dataset, comparing its behavior before and after DPO training on held-out prompts.
**Mathematical:** Derive the DPO loss (Section 3) starting from the RLHF objective and the Bradley-Terry preference model, following the substitution argument sketched in Section 3.
**Coding:** Implement a simple KL-divergence monitoring utility that tracks how far a policy model's output distribution has drifted from a frozen reference model during training, and use it to empirically observe the effect of different $\beta$ values in a DPO training run.

---

## 14. Mini Project

**Fine-tune a small open-weight model using both SFT and DPO on a domain-specific task**: curate (or synthetically generate) a small instruction-following dataset relevant to your actuarial/insurance domain, perform SFT, then construct a preference dataset (e.g., preferring more precise/appropriately-hedged actuarial language over vague responses) and fine-tune further via DPO using the `trl` library, qualitatively and (where possible) quantitatively evaluating the resulting model's behavior against the SFT-only baseline — a genuine, hands-on demonstration of the full modern fine-tuning pipeline applied to your own domain.

---

## 15. Interview Preparation

- Explain the three stages of the InstructGPT-style RLHF pipeline, and which stage(s) are genuinely reinforcement learning versus supervised learning.
- Derive (at a conceptual level) how DPO avoids needing an explicit reward model.
- What is the purpose of the KL-divergence penalty in RLHF, and what happens if it's set too low?
- What is catastrophic forgetting, and what fine-tuning practices help mitigate it?

---

## 16. Summary

Fine-tuning transforms a raw pretrained model (Lesson 5) into a genuinely helpful, instruction-following assistant through supervised fine-tuning on curated demonstrations, followed by either the classical three-stage RLHF pipeline (reward modeling via ordinary supervised Bradley-Terry classification, then PPO-based policy optimization with a KL-divergence penalty against reward-hacking) or the mathematically-equivalent, substantially simpler DPO reformulation now widely adopted since 2023. Every behavioral property of a production assistant — following instructions, refusing certain requests, adopting a particular conversational style — traces directly to decisions made at exactly this stage, built on top of Lesson 5's pretraining and setting up Lesson 7's parameter-efficient techniques for doing this fine-tuning affordably.

---

## 17. References

- Ouyang et al. — "Training Language Models to Follow Instructions with Human Feedback" (2022, the InstructGPT/RLHF paper)
- Schulman et al. — "Proximal Policy Optimization Algorithms" (2017, PPO)
- Rafailov et al. — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023, the DPO paper)
- Bai et al. — "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (2022, Anthropic's RLHF work)
