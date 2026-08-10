# Phase 6 · Lesson 9 — Prompt Engineering

> Prerequisite: Fine-Tuning (Lesson 6) — understanding *why* instruction-tuned models respond the way they do

---

## 1. Introduction

### What is prompt engineering?
The practice of designing input text (prompts) to reliably elicit desired behavior from an LLM, without modifying the model's weights at all — in contrast to fine-tuning (Lessons 6-7), which changes model *parameters*, prompt engineering works entirely at inference time, exploiting the model's existing (pretrained + fine-tuned) capabilities through careful input construction.

### Why does it exist?
A single pretrained/fine-tuned model can be steered toward an enormous range of different behaviors and tasks purely through how it's prompted — no retraining needed. This makes prompt engineering the fastest, cheapest, most accessible lever for adapting LLM behavior, and for many applications, sufficient on its own without ever needing fine-tuning at all.

### Historical background
GPT-3's paper (Brown et al., 2020) demonstrated "few-shot learning" — providing a handful of examples directly in the prompt could elicit task performance rivaling fine-tuned models, without any weight updates, a genuinely surprising empirical finding at the time that launched prompt engineering as a distinct practical discipline. Chain-of-thought prompting (Wei et al., 2022) then showed that simply asking a model to "think step by step" substantially improved performance on multi-step reasoning tasks — a remarkably simple technique with an outsized, now well-replicated effect.

### Real-world motivation
Before reaching for fine-tuning (expensive, requires data curation, Lessons 6-7) or RAG (Phase 7, requires infrastructure), prompt engineering is almost always the first, cheapest thing to try for adapting an LLM's behavior to your specific task — and even after fine-tuning/RAG are in place, prompt design remains a critical, ongoing part of any production LLM application.

---

## 2. Theory

### Zero-shot, few-shot, and in-context learning
- **Zero-shot**: ask the model to perform a task directly, with no examples.
- **Few-shot**: provide a small number of example (input, output) pairs directly in the prompt before the actual query — the model infers the task pattern from these examples, entirely at inference time (no gradient updates occur; this is fundamentally different from fine-tuning despite the superficial similarity of "learning from examples").
- **In-context learning**: the general phenomenon underlying few-shot prompting — the model uses the prompt's content as a temporary, session-specific "context" to adjust its behavior, a capability that emerges from pretraining/fine-tuning rather than being an explicitly designed mechanism.

### Chain-of-thought (CoT) prompting
Explicitly prompting a model to generate intermediate reasoning steps before its final answer ("let's think step by step") substantially improves performance on tasks requiring multi-step reasoning (arithmetic, logic, multi-hop question answering) — plausibly because it gives the model more "computational steps" (more forward passes, effectively) to work through a problem, rather than requiring the entire solution to emerge from a single forward pass's worth of computation.

### System prompts and role-based prompting
Most production LLM APIs distinguish a "system" message (setting overall behavior, tone, constraints) from "user" messages (the actual conversational turns) — system prompts are the standard mechanism for establishing persistent behavior/persona/constraints across an entire conversation without repeating them in every user turn.

### Structured output prompting
Explicitly requesting a specific output format (JSON, XML, a particular schema) — often combined with providing an example of the desired format — substantially improves the reliability of extracting structured data from an LLM's response, directly relevant to any application needing to parse LLM output programmatically (Phase 7-8's agentic/tool-calling systems depend heavily on this).

---

## 3. Mathematical Foundations

### Why few-shot examples work (a probabilistic framing)
An LLM computes $P(\text{next token} | \text{context})$ (Lesson 5). Adding few-shot examples to the context changes the *conditioning* — the model's next-token distribution shifts to be consistent with the pattern demonstrated in the examples, exploiting the model's pretrained ability to recognize and continue patterns, without any parameter update. Formally, this is not "learning" in the gradient-descent sense (Phase 3 Lesson 5) at all — it's a form of pattern-matching/analogical inference performed entirely within a single forward pass over the (now longer) context.

### Temperature and its effect on the output distribution (directly reusing Phase 5 Lesson 7)
$$
P(\text{token}_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$
As covered in Phase 5 Lesson 7: low $T$ sharpens the distribution toward the highest-logit tokens (more deterministic, "confident" output); high $T$ flattens it (more diverse, sometimes less coherent output) — a direct, quantitative lever prompt engineers tune for tasks requiring precision (low $T$, e.g., factual extraction) versus creativity (higher $T$, e.g., brainstorming).

### Why prompt order/position can matter (a direct consequence of positional encoding, Lesson 3)
Empirically, information placed at the very beginning or very end of a long context is sometimes attended to more reliably than information in the middle (a phenomenon informally called "lost in the middle") — plausibly related to how attention patterns and positional encoding schemes (Lesson 3) interact with training data distributions (which may have systematically emphasized certain positions), though this remains an active empirical/research area rather than a fully settled theoretical result.

### Self-consistency (an ensembling technique applied to reasoning, directly reusing Phase 4 Lesson 5)
Rather than taking a single chain-of-thought reasoning path, **self-consistency** (Wang et al., 2022) samples *multiple* independent reasoning paths (via temperature-based sampling) for the same problem, then takes a majority vote over their final answers — directly analogous to Phase 4 Lesson 5's ensembling/bagging logic (reducing variance by aggregating multiple independent, imperfectly-correlated "models," where here each sampled reasoning path plays the role of one ensemble member).

---

## 4. Algorithm — Constructing an Effective Few-Shot Prompt (a practical procedure)

```
GIVEN a task and a target LLM:
1. Write a CLEAR task instruction (what exactly should the model do, in plain language)
2. SELECT 2-5 representative examples covering the range of inputs you expect
   (diverse enough to demonstrate the pattern, not so many that they bloat the prompt unnecessarily)
3. FORMAT examples CONSISTENTLY (same structure/delimiters for every example -- inconsistency
   confuses the pattern-matching the model relies on for in-context learning)
4. PLACE the actual query AFTER the examples, in the SAME format
5. IF the task involves multi-step reasoning: explicitly request step-by-step reasoning
   BEFORE the final answer ("think step by step", or a structured reasoning format)
6. IF structured output is needed: explicitly specify the exact format (e.g., "respond only in
   valid JSON matching this schema: ...") and, ideally, show one example of correctly formatted output
7. TEST across a range of realistic inputs, not just the examples used to construct the prompt itself
   -- iterate based on observed failure modes
```

---

## 5. Python Implementation

```python
"""prompt_engineering_core.py — few-shot prompt construction and self-consistency voting"""
from collections import Counter


def build_few_shot_prompt(task_instruction: str, examples: list[dict], query: str) -> str:
    prompt_parts = [task_instruction, ""]
    for ex in examples:
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['output']}")
        prompt_parts.append("")
    prompt_parts.append(f"Input: {query}")
    prompt_parts.append("Output:")
    return "\n".join(prompt_parts)


def build_chain_of_thought_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Let's think through this step by step, showing all reasoning, "
        "before giving the final answer.\n\nReasoning:"
    )


def self_consistency_vote(sampled_answers: list[str]) -> tuple[str, float]:
    """Majority vote over multiple independently-sampled reasoning paths (Section 3)."""
    counts = Counter(sampled_answers)
    winner, count = counts.most_common(1)[0]
    confidence = count / len(sampled_answers)
    return winner, confidence


# Example: few-shot prompt for a sentiment-classification-style task
task = "Classify the sentiment of each review as Positive, Negative, or Neutral."
examples = [
    {"input": "This product exceeded my expectations!", "output": "Positive"},
    {"input": "It broke after one use.", "output": "Negative"},
    {"input": "It's okay, does what it says.", "output": "Neutral"},
]
prompt = build_few_shot_prompt(task, examples, query="Absolutely fantastic service, will buy again!")
print(prompt)

cot_prompt = build_chain_of_thought_prompt(
    "A store had 120 items. It sold 35% on Monday and 20% of the REMAINDER on Tuesday. How many items are left?"
)
print("\n" + cot_prompt)

# Simulating self-consistency: 5 independently-sampled reasoning paths' FINAL answers
sampled = ["62", "62", "78", "62", "61"]   # (in practice: from 5 separate, temperature>0 LLM calls)
winner, confidence = self_consistency_vote(sampled)
print(f"\nSelf-consistency answer: {winner} (confidence: {confidence:.0%})")
```

---

## 6. Build From Scratch

**A minimal prompt-robustness testing harness (systematically probing sensitivity to prompt phrasing, a genuinely important practical practice):**
```python
def test_prompt_robustness(base_task: str, phrasing_variants: list[str], test_inputs: list[str],
                             mock_llm_call) -> dict:
    """Tests whether SEMANTICALLY EQUIVALENT prompt phrasings produce CONSISTENT outputs --
    a real, common failure mode: LLMs can be surprisingly sensitive to superficial wording changes."""
    results = {}
    for phrasing in phrasing_variants:
        outputs = []
        for test_input in test_inputs:
            full_prompt = f"{phrasing}\n\nInput: {test_input}\nOutput:"
            outputs.append(mock_llm_call(full_prompt))     # in practice: a real API call
        results[phrasing] = outputs
    return results

# Illustrative usage (mock_llm_call would be a real API call in practice):
variants = [
    "Classify the sentiment as Positive, Negative, or Neutral.",
    "What is the sentiment of this text? Answer with Positive, Negative, or Neutral.",
    "Determine whether the following expresses positive, negative, or neutral sentiment.",
]
# results = test_prompt_robustness(base_task, variants, test_inputs, real_llm_call_function)
# -> compare whether all three phrasings produce the SAME classifications across test_inputs
```
Running a real version of this (with an actual LLM API call substituted for `mock_llm_call`) on a representative test set is a genuinely important production practice: prompts that work well on a handful of manually-checked examples can still be surprisingly brittle to minor rephrasing or input variation — systematic testing catches this before deployment, directly echoing Phase 4 Lesson 3's evaluation rigor, now applied to prompts rather than model weights.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `build_few_shot_prompt`/`build_chain_of_thought_prompt` | Prompt templating libraries (LangChain's `PromptTemplate`, or simple Jinja2 templates) — add variable substitution, versioning, and reuse across an application |
| `self_consistency_vote` | Directly usable as-is; some frameworks (LangChain, LlamaIndex) provide built-in self-consistency/majority-voting utilities for agentic pipelines |
| `test_prompt_robustness` | Dedicated prompt evaluation/testing frameworks (e.g., `promptfoo`, or custom evaluation harnesses built on Phase 4 Lesson 3's evaluation principles) for systematic, repeatable prompt testing across model versions |

---

## 8. Visual Explanations

**Zero-shot vs. few-shot prompting:**
```
ZERO-SHOT:                              FEW-SHOT:
"Classify: 'Great product!'"            "Input: 'Terrible service' -> Output: Negative
                                          Input: 'Loved it!' -> Output: Positive
                                          Input: 'Great product!' -> Output:"
   (relies ENTIRELY on pretrained/       (demonstrates the EXACT input/output pattern
    fine-tuned understanding of           expected, exploiting in-context learning --
    the task from instructions alone)     often more reliable for less-common task formats)
```

**Self-consistency: multiple reasoning paths, majority vote:**
```
Same question, sampled 5 times (temperature > 0):
Path 1: ... reasoning ... -> Answer: 62
Path 2: ... DIFFERENT reasoning ... -> Answer: 62
Path 3: ... yet another path ... -> Answer: 78   (an error in THIS particular path)
Path 4: ... -> Answer: 62
Path 5: ... -> Answer: 61   (a different error)
                    │
                    ▼
          MAJORITY VOTE: 62  (3 out of 5 paths agree -- likely the correct answer)
```

---

## 9. Practical Examples

**Simple:** write a zero-shot and a few-shot prompt for the same classification task, and compare output consistency across several test inputs.
**Medium:** construct a chain-of-thought prompt for a multi-step arithmetic word problem and verify (via an actual LLM call) that it produces a more reliable correct answer than a direct zero-shot prompt.
**Real-world:** design a robust, well-tested prompt for extracting structured information (e.g., key terms from an insurance policy document) into a strict JSON schema, systematically testing it (Section 6's robustness-testing approach) across a diverse set of real policy documents to identify and fix failure modes before relying on it in a pipeline.

---

## 10. Real Industry Use Cases

- **Every production LLM application built on top of an API** (customer support bots, content generation tools, code assistants): prompt engineering is the primary, often only, customization mechanism used, given its zero-infrastructure cost compared to fine-tuning.
- **Agentic systems** (Phase 7): tool-calling and multi-step agent behavior rely heavily on carefully engineered system prompts specifying available tools, expected output formats, and behavioral constraints.
- **Structured data extraction pipelines**: legal, financial, and insurance document processing systems routinely use carefully engineered, extensively tested prompts to reliably extract structured data from unstructured text, often as a cheaper/faster alternative to training a dedicated extraction model.
- **Chain-of-thought and self-consistency in production reasoning systems**: used in math/coding-assistant applications and complex analytical tasks where reliability on multi-step reasoning is business-critical.

---

## 11. Common Mistakes

- Assuming a prompt that works on a handful of manually-checked examples is robust in general — LLMs can be surprisingly sensitive to superficial rephrasing, input length, or ordering, requiring systematic testing (Section 6) before production use.
- Overloading a single prompt with too many, inconsistent-format few-shot examples — confuses rather than clarifies the intended pattern.
- Forgetting to specify output format explicitly when downstream code needs to parse the response programmatically — leads to fragile, inconsistent parsing failures.
- Treating prompt engineering as a substitute for fine-tuning/RAG in cases genuinely requiring updated factual knowledge (RAG, Phase 7) or substantially different behavior than in-context learning can reliably achieve (fine-tuning, Lessons 6-7) — prompt engineering has real limits, and recognizing when to escalate to a more substantial technique is itself an important skill.

---

## 12. Best Practices (2026)

- Always test prompts systematically across a representative, diverse set of realistic inputs before production deployment — treat prompt development with the same evaluation rigor (Phase 4 Lesson 3) as model development.
- Use explicit chain-of-thought prompting for any task involving multi-step reasoning, arithmetic, or logic.
- Specify output format explicitly (and provide a format example) whenever downstream code needs to parse LLM output programmatically.
- Use self-consistency (multiple sampled reasoning paths + majority vote) for high-stakes reasoning tasks where reliability matters more than the added inference cost of multiple samples.
- Recognize prompt engineering's limits — escalate to RAG (Phase 7) for knowledge-freshness/grounding needs, or fine-tuning (Lessons 6-7) for substantial, persistent behavioral changes that in-context learning alone can't reliably achieve.

---

## 13. Exercises

**Easy:** Write both a zero-shot and a 3-example few-shot prompt for a simple text classification task, and manually compare their outputs on 5 test inputs.
**Medium:** Construct a chain-of-thought prompt for a multi-step logic puzzle and compare its success rate (across several trials) against a direct zero-shot prompt for the same puzzle.
**Hard:** Implement the self-consistency voting procedure (Section 5) using an actual LLM API, sampling 5-10 independent reasoning paths for a set of moderately difficult arithmetic/logic problems, and empirically measure the accuracy improvement over single-sample chain-of-thought.
**Mathematical:** Given a per-sample correctness probability $p > 0.5$ for a reasoning task, derive the probability that majority voting over $n$ independent samples produces the correct answer, and discuss how this compares to Phase 4 Lesson 5's ensemble variance-reduction mathematics.
**Coding:** Build the prompt-robustness testing harness (Section 6) using a real LLM API and systematically test 3-5 semantically equivalent phrasings of the same task across a diverse input set, reporting any inconsistencies found.

---

## 14. Mini Project

Build a **robust structured-extraction prompt pipeline for actuarial/insurance documents**: design a prompt that extracts key structured fields (policy type, coverage amount, key dates, exclusions) from unstructured policy text into a strict JSON schema, iteratively test and refine it (using the Section 6 robustness-testing methodology) across a diverse set of real or synthetic policy documents, incorporate chain-of-thought reasoning for any fields requiring interpretation/inference rather than direct extraction, and produce a final validated prompt along with a written report characterizing its failure modes and reliability across your test set.

---

## 15. Interview Preparation

- Explain the difference between zero-shot, few-shot, and fine-tuning as approaches to adapting LLM behavior, and when you'd choose each.
- Why does chain-of-thought prompting improve performance on multi-step reasoning tasks?
- What is self-consistency, and how does it relate to ensemble methods (Phase 4 Lesson 5)?
- How would you systematically test whether a prompt is robust before deploying it in production?

---

## 16. Summary

Prompt engineering exploits an LLM's existing pretrained and fine-tuned capabilities entirely through inference-time input construction — few-shot examples demonstrate a task pattern via in-context learning (no gradient updates), chain-of-thought prompting improves multi-step reasoning by eliciting intermediate computation, and self-consistency directly applies Phase 4 Lesson 5's ensembling logic to reasoning reliability. It is the cheapest, fastest lever for adapting LLM behavior — appropriately the first technique to reach for before escalating to RAG (Phase 7, for knowledge grounding) or fine-tuning (Lessons 6-7, for substantial persistent behavioral change) — but requires the same systematic, evaluation-driven rigor (Phase 4 Lesson 3) as any other engineering discipline to be genuinely reliable in production.

---

## 17. References

- Brown et al. — "Language Models are Few-Shot Learners" (2020, the GPT-3 paper establishing few-shot in-context learning)
- Wei et al. — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Wang et al. — "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022)
- Liu et al. — "Lost in the Middle: How Language Models Use Long Contexts" (2023, the empirical positional-attention-reliability study)
