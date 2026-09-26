# Phase 7 · Lesson 10 — Evaluation & Guardrails

> Prerequisite: All prior Phase 7 lessons, Phase 4 Lesson 3 (Model Evaluation) — this lesson closes out Phase 7

---

## 1. Introduction

### What does this lesson cover?
Systematic methods for measuring whether a generative AI application (RAG systems, agents, any LLM-based pipeline built across this phase) actually works well — and mechanisms for constraining its behavior within acceptable bounds (guardrails) to prevent harmful, incorrect, or off-policy outputs from reaching end users. Where Phase 4 Lesson 3 covered evaluating classical ML models against ground-truth labels, this lesson addresses the genuinely harder problem of evaluating open-ended, generative text output, where "correctness" is often not a simple binary match.

### Why does it exist?
Every technique built across this phase — RAG (Lesson 1), agents (Lesson 3), multi-agent systems (Lesson 8) — can fail in ways that are subtle, hard to detect via casual inspection, and genuinely costly in production (a RAG system confidently citing a source that doesn't actually support its claim; an agent taking a harmful or costly unintended action; a structured extraction silently fabricating data, Lesson 6's semantic-correctness gap). Rigorous evaluation and guardrails exist specifically to catch these failure modes systematically, before and during production deployment, rather than relying on hope or occasional manual spot-checks.

### Historical background
LLM evaluation matured rapidly through 2023-2026 from ad hoc manual review toward more systematic approaches: reference-based metrics (BLEU/ROUGE, borrowed from earlier machine translation/summarization research) proved poorly correlated with actual quality for open-ended generation, driving the field toward LLM-as-judge evaluation (using a separate, often more capable LLM to assess quality against a rubric) and specialized RAG-specific metrics (faithfulness, groundedness) — alongside a parallel, increasingly formalized guardrails discipline (input/output filtering, policy enforcement) as LLM applications moved from research demos into consequential production use.

### Real-world motivation
Every application built throughout this phase needs this lesson's discipline before being trusted in production — a RAG system, an agent, or a multi-agent pipeline that "seems to work" on a few manually-checked examples can still fail systematically and expensively at real scale without rigorous, ongoing evaluation.

---

## 2. Theory

### Why classical NLP metrics (BLEU/ROUGE) are usually insufficient for LLM evaluation
BLEU/ROUGE measure n-gram overlap between generated text and a single reference text — appropriate for tasks with a fairly constrained, narrow space of "correct" outputs (like early machine translation), but poorly suited to open-ended generation where many substantively different phrasings can be equally correct/good, and where these metrics can penalize genuinely excellent responses that simply don't share surface-level wording with a specific reference.

### LLM-as-judge evaluation
Using a separate LLM call (ideally a more capable model, or at minimum a differently-prompted one to reduce shared blind spots) to assess a generated response against an explicit rubric — e.g., "rate this response's helpfulness, accuracy, and conciseness on a 1-5 scale, with justification." This approach correlates substantially better with human judgment than n-gram-overlap metrics for open-ended tasks, though it introduces its own concerns (judge model biases, potential for the same failure modes the generator exhibits to go undetected by a similarly-limited judge).

### RAG-specific evaluation metrics
- **Faithfulness/groundedness**: does the generated response's content actually follow from the retrieved context (Lesson 1), or does it introduce unsupported claims (a specific, measurable form of hallucination)?
- **Answer relevance**: does the response actually address the user's question, independent of whether it's grounded?
- **Context relevance**: did the retrieval step (Lesson 1/2) actually surface relevant documents in the first place — a low-quality response might stem from poor retrieval rather than poor generation, a distinction Lesson 1 already flagged as important to diagnose separately.

### Guardrails — constraining behavior, not just measuring it after the fact
- **Input guardrails**: filtering/validating user input before it reaches the LLM (detecting prompt injection attempts, off-topic requests, or policy-violating content).
- **Output guardrails**: filtering/validating the LLM's response before it reaches the end user (checking for policy violations, verifying structured-output schema compliance beyond Lesson 6's syntactic guarantee, or checking generated content against a safety classifier).
- **Behavioral guardrails for agents**: constraining which tools/actions an agent is permitted to take autonomously (Lesson 3), potentially requiring human approval for high-stakes actions (sending an email, executing a financial transaction) rather than allowing fully autonomous execution.

---

## 3. Mathematical Foundations

### LLM-as-judge agreement with human judgment (an empirical validation concern)
Before trusting an LLM-judge's ratings as a stand-in for human evaluation at scale, it's standard practice to measure the judge's agreement with a smaller set of genuine human ratings — using metrics like Cohen's kappa (a chance-corrected agreement statistic, directly extending Phase 3 Lesson 4's statistical framework) or simple correlation, establishing an empirical basis for trusting the judge's scalability before relying on it as the primary evaluation signal.

### Faithfulness as an entailment/attribution problem
Formally, faithfulness checking asks: for each claim $c$ in the generated response, does the retrieved context $D$ entail $c$ (i.e., does $D$ logically/factually support $c$)? This can be operationalized as a **Natural Language Inference (NLI)**-style classification task — for each extracted claim, classify the relationship to the source context as entailment, contradiction, or neutral (unsupported) — directly a specialized instance of Phase 4 Lesson 1's classification framework, applied to fact-verification rather than a generic label.

### Statistical rigor in evaluation, directly reusing Phase 4 Lesson 3
Any claim that "system A performs better than system B" on an LLM evaluation metric requires the same statistical discipline as classical model comparison (Phase 4 Lesson 3): report confidence intervals (via bootstrap, Phase 3 Lesson 4) on the evaluation metric, not just a single point estimate, and use proper hypothesis testing rather than declaring a "winner" based on a small, potentially noisy sample of evaluated examples.

### Guardrail precision/recall tradeoffs (directly reusing Phase 4 Lesson 3)
A guardrail classifier (e.g., detecting policy-violating content) faces the exact same precision/recall tradeoff as any binary classifier (Phase 4 Lesson 3): overly aggressive guardrails (high recall, catching most violations) risk false positives (blocking legitimate, benign content), while overly permissive guardrails (high precision, rarely blocking legitimate content) risk missing genuine violations — the appropriate operating point depends entirely on the specific application's cost structure for each error type, exactly as in any classical classification threshold decision.

---

## 4. Algorithm — A Complete RAG Evaluation Pipeline (fully specified)

```
GIVEN a RAG system (Lesson 1) and a test set of (question, expected_answer_characteristics) examples:
FOR each test example:
    1. RUN the RAG pipeline: retrieve context, generate response
    2. EVALUATE RETRIEVAL:
         compute Recall@k / MRR against known-relevant documents (Lesson 1's metrics)
    3. EVALUATE FAITHFULNESS:
         extract individual claims from the generated response
         FOR each claim: check (via NLI-style classification, Section 3, often itself an LLM call)
             whether the retrieved context entails, contradicts, or is neutral toward that claim
         compute faithfulness score = fraction of claims ENTAILED by the context
    4. EVALUATE ANSWER RELEVANCE:
         use an LLM-judge (Section 2) to rate whether the response actually addresses the question
    5. LOG all scores, plus the FULL transcript (question, retrieved context, response, per-metric scores)
       for manual review of any low-scoring examples
AGGREGATE: compute mean + bootstrap confidence interval (Phase 3 Lesson 4) for each metric across the test set
COMPARE against previous system versions using proper statistical testing (Phase 4 Lesson 3), not just
  raw score differences
```

---

## 5. Python Implementation

```python
"""evaluation_guardrails_core.py — LLM-as-judge, faithfulness checking, and input/output guardrails"""
import re
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    justification: str = ""


def llm_judge_evaluate(question: str, response: str, judge_llm_call, rubric: str) -> EvaluationResult:
    """Uses a separate LLM call with an explicit rubric -- Section 2's LLM-as-judge pattern."""
    prompt = (
        f"Question: {question}\nResponse: {response}\n\n"
        f"Rubric: {rubric}\n\n"
        f"Rate the response from 1-5 and provide a brief justification. "
        f"Format: Score: <number>\\nJustification: <text>"
    )
    judge_output = judge_llm_call(prompt)
    score_match = re.search(r"Score:\s*(\d)", judge_output)
    justification_match = re.search(r"Justification:\s*(.*)", judge_output, re.DOTALL)
    score = float(score_match.group(1)) if score_match else 0.0
    justification = justification_match.group(1).strip() if justification_match else ""
    return EvaluationResult("llm_judge_quality", score / 5.0, justification)


def extract_claims(response: str, extraction_llm_call) -> list[str]:
    """Decomposes a response into individual, independently-checkable factual claims."""
    prompt = f"List each distinct factual claim made in this text, one per line:\n\n{response}"
    output = extraction_llm_call(prompt)
    return [line.strip("- ").strip() for line in output.split("\n") if line.strip()]


def check_faithfulness(claims: list[str], context: str, nli_llm_call) -> EvaluationResult:
    """Section 3's NLI-style entailment check, per claim."""
    entailed_count = 0
    for claim in claims:
        prompt = (
            f"Context: {context}\nClaim: {claim}\n\n"
            f"Does the context ENTAIL, CONTRADICT, or is NEUTRAL toward this claim? Answer with one word."
        )
        verdict = nli_llm_call(prompt).strip().upper()
        if verdict == "ENTAIL":
            entailed_count += 1
    score = entailed_count / len(claims) if claims else 1.0   # vacuously "faithful" if no claims extracted
    return EvaluationResult("faithfulness", score)


# --- Guardrails ---
def input_guardrail_check(user_input: str, blocked_patterns: list[str]) -> tuple[bool, str]:
    """A simple rule-based input guardrail (real systems typically ALSO use a classifier model)."""
    for pattern in blocked_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False, f"Input blocked: matched restricted pattern '{pattern}'"
    return True, ""


def output_guardrail_check(response: str, schema_validator=None, safety_classifier_call=None) -> tuple[bool, str]:
    """Output guardrail combining schema validation (Lesson 6) and a safety classifier."""
    if schema_validator is not None:
        try:
            schema_validator(response)
        except Exception as e:
            return False, f"Output failed schema validation: {e}"
    if safety_classifier_call is not None:
        verdict = safety_classifier_call(response)
        if verdict == "unsafe":
            return False, "Output flagged by safety classifier"
    return True, ""
```

---

## 6. Build From Scratch

**A minimal Cohen's kappa implementation (validating an LLM-judge against human ratings, Section 3):**
```python
import numpy as np

def cohens_kappa(rater_a: list[int], rater_b: list[int], n_categories: int) -> float:
    """Chance-corrected agreement -- essential BEFORE trusting an LLM-judge as a human-rating proxy."""
    n = len(rater_a)
    confusion = np.zeros((n_categories, n_categories))
    for a, b in zip(rater_a, rater_b):
        confusion[a, b] += 1

    observed_agreement = np.trace(confusion) / n
    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    expected_agreement = np.sum(row_marginals * col_marginals)   # agreement expected BY CHANCE ALONE

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)

# Example: 20 examples, human ratings (1-5 scale, 0-indexed as 0-4) vs LLM-judge ratings
human_ratings = [4, 3, 4, 2, 4, 3, 3, 4, 2, 4, 3, 4, 4, 2, 3, 4, 4, 3, 2, 4]
judge_ratings = [4, 3, 3, 2, 4, 4, 3, 4, 2, 3, 3, 4, 4, 2, 3, 4, 3, 3, 3, 4]
kappa = cohens_kappa(human_ratings, judge_ratings, n_categories=5)
print(f"Cohen's kappa (LLM-judge vs human): {kappa:.3f}")
# Rule of thumb: >0.6 substantial agreement, >0.8 near-perfect -- below ~0.4 suggests the
# judge shouldn't yet be trusted as a scalable proxy for human evaluation on this task
```

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `llm_judge_evaluate`/`check_faithfulness` (manual) | RAGAS, TruLens — dedicated RAG/LLM evaluation frameworks implementing faithfulness, answer relevance, and context relevance metrics out of the box |
| `cohens_kappa` | `sklearn.metrics.cohen_kappa_score` — identical formula, standard library function |
| Manual regex-based input guardrails | Dedicated guardrail frameworks (Guardrails AI, NeMo Guardrails) — provide configurable, extensible input/output validation pipelines combining rules and classifier models |
| No safety classifier shown | Provider-native content moderation APIs, or open-source safety classifiers (e.g., Llama Guard) for output screening |

---

## 8. Visual Explanations

**Diagnosing a poor RAG response: retrieval failure vs. generation failure:**
```
Poor final answer
       │
       ├── Was the RIGHT context retrieved? ── NO ──▶ RETRIEVAL problem (Lesson 1/2: chunking, embedding, index)
       │
       └── YES, right context retrieved
                  │
                  └── Did the response USE it faithfully? ── NO ──▶ GENERATION/faithfulness problem
                                                        │
                                                       YES ──▶ possibly a RELEVANCE problem
                                                                (right facts, but didn't answer the actual question)
```

**Guardrail precision/recall tradeoff (directly reusing Phase 4 Lesson 3):**
```
STRICT guardrail (high recall, catches most violations):     LENIENT guardrail (high precision):
  → blocks MORE borderline/benign content (false positives)     → MISSES more genuine violations (false negatives)
  → appropriate when the cost of a violation slipping           → appropriate when the cost of blocking
    through is very high                                          legitimate content is very high
```

---

## 9. Practical Examples

**Simple:** implement `llm_judge_evaluate` (Section 5) with a real LLM API and a simple rubric, testing it on a handful of manually-rated example responses to sanity-check the judge's ratings.
**Medium:** implement the faithfulness-checking pipeline (Section 5) on a real RAG system's outputs (from your Lesson 1 mini project) and identify any responses containing unsupported (non-entailed) claims.
**Real-world:** validate an LLM-judge against a small set of genuine human ratings (Section 6's Cohen's kappa) on your actuarial RAG/agent application's outputs before relying on the judge for larger-scale, ongoing evaluation.

---

## 10. Real Industry Use Cases

- **RAGAS and TruLens**: widely-adopted open-source frameworks specifically implementing the faithfulness/relevance metric suite covered in this lesson, used across the RAG industry for systematic evaluation.
- **Content moderation pipelines**: every major consumer-facing LLM product uses input/output guardrails combining rule-based filters and classifier models to prevent policy-violating content.
- **Agentic system safety**: production coding/browsing agents commonly require human approval for high-stakes actions (Section 2's behavioral guardrails), rather than fully autonomous execution of consequential operations.
- **Regulated-industry LLM applications** (directly relevant to insurance/actuarial use): evaluation rigor and guardrails aren't just good practice but often a genuine compliance requirement, with auditable evaluation logs and documented guardrail policies expected by regulators.

---

## 11. Common Mistakes

- Using BLEU/ROUGE-style n-gram overlap metrics for open-ended generation evaluation — poorly correlated with actual quality, potentially penalizing genuinely excellent, differently-phrased responses.
- Trusting an LLM-judge without first validating its agreement with human ratings (Section 6) — an unvalidated judge might share the same blind spots as the system it's evaluating, or simply be an unreliable proxy for the specific task/domain.
- Conflating retrieval quality and generation quality when diagnosing a poor RAG response — these require different fixes and should be evaluated (and debugged) separately, per Section 8's diagnostic flowchart.
- Deploying guardrails without considering their precision/recall tradeoff for the specific application's cost structure — a one-size-fits-all guardrail configuration is rarely appropriate across genuinely different risk contexts.

---

## 12. Best Practices (2026)

- Use LLM-as-judge evaluation for open-ended generation quality, but validate the judge's agreement with human ratings (Cohen's kappa or similar) before trusting it at scale.
- Evaluate RAG systems on faithfulness, answer relevance, AND context relevance separately, to correctly diagnose whether problems stem from retrieval or generation.
- Report evaluation metrics with confidence intervals (Phase 3 Lesson 4) and use proper statistical comparison (Phase 4 Lesson 3) rather than declaring a "winner" from a small, potentially noisy sample.
- Layer guardrails (input filtering, output validation, behavioral constraints for agentic actions) appropriately for your application's actual risk profile, and require human approval for genuinely high-stakes autonomous actions.

---

## 13. Exercises

**Easy:** Implement `llm_judge_evaluate` (Section 5) and manually verify its ratings align with your own intuitive judgment on 5-10 example responses.
**Medium:** Implement the faithfulness-checking pipeline (Section 5) and deliberately construct a response containing one fabricated (non-entailed) claim, verifying the pipeline correctly flags it.
**Hard:** Collect a small set of genuine human ratings for a batch of LLM responses, compute Cohen's kappa (Section 6) between an LLM-judge and the human ratings, and discuss whether the resulting agreement level would justify relying on the judge at scale for this task.
**Mathematical:** Derive why Cohen's kappa corrects for chance agreement (unlike raw percentage agreement), using a concrete example where two raters agree frequently but largely by chance (e.g., both raters heavily favor one category).
**Coding:** Implement a combined input+output guardrail pipeline (Section 5) for a specific application policy of your choosing, and test its precision/recall tradeoff across a set of both legitimate and policy-violating test inputs.

---

## 14. Mini Project

Build a **complete evaluation and guardrails suite for your actuarial RAG/agent application** (drawing on Lessons 1, 3, and 6-9's mini projects): implement faithfulness, answer relevance, and context relevance evaluation metrics; validate an LLM-judge against a small set of your own human ratings via Cohen's kappa before trusting it further; implement input guardrails (blocking off-topic or policy-violating requests) and output guardrails (schema validation plus a safety check); and produce a final evaluation report with confidence intervals on all metrics, plus a documented guardrail policy explaining the precision/recall tradeoff choices made for your specific application's risk profile.

---

## 15. Interview Preparation

- Why are classical NLP metrics like BLEU/ROUGE generally insufficient for evaluating open-ended LLM generation?
- What is LLM-as-judge evaluation, and what must be validated before trusting it at scale?
- Explain the difference between faithfulness, answer relevance, and context relevance as RAG evaluation metrics, and why they should be measured separately.
- How would you design guardrails for an LLM agent that can take real-world actions, and what tradeoffs are involved?

---

## 16. Summary

Rigorous evaluation and guardrails close out this phase by ensuring every technique built across it — RAG (Lesson 1), agents (Lesson 3), multi-agent systems (Lesson 8), memory (Lesson 9) — can actually be trusted in production rather than merely appearing to work on a handful of manually-checked examples. LLM-as-judge evaluation (validated against human agreement via Cohen's kappa) addresses open-ended generation's evaluation challenge where classical n-gram metrics fall short; faithfulness/relevance metrics specifically diagnose RAG systems' retrieval-versus-generation failure modes; and layered guardrails (input filtering, output validation, behavioral constraints on agentic actions) constrain real-world risk with the same precision/recall discipline as any classical classifier (Phase 4 Lesson 3). This evaluation-and-safety rigor is what separates a genuinely production-ready generative AI application from an impressive-looking demo — the essential final discipline underlying every system this phase has taught you to build.

---

## 17. References

- Es et al. — "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023, the foundational RAG-evaluation-framework paper)
- Zheng et al. — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023, foundational LLM-as-judge validation research)
- Cohen, J. — "A Coefficient of Agreement for Nominal Scales" (1960, the original Cohen's kappa paper)
- NeMo Guardrails and Guardrails AI official documentation
