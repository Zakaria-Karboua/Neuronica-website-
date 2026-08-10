# Phase 7 · Lesson 7 — Agentic Workflows

> Prerequisite: AI Agents, MCP, Tool Calling, Structured Outputs (Lessons 3–6)

---

## 1. Introduction

### What are agentic workflows?
Structured, often graph-based orchestration patterns for composing multiple LLM calls, tool invocations, and control-flow logic (conditionals, loops, parallel branches) into a coherent application — moving beyond Lesson 3's simple, purely reactive ReAct loop into more deliberate, engineered patterns (sequential chains, routing, parallelization, and explicit planning) suited to complex, production-grade tasks.

### Why does it exist?
Lesson 3's basic ReAct loop is powerful but unstructured — for genuinely complex applications, purely emergent step-by-step reasoning can be inefficient (missing opportunities for parallelization), unpredictable (no guaranteed structure for critical steps), and hard to debug/monitor. Agentic workflow patterns exist to impose deliberate, often more efficient and predictable structure on top of an agent's underlying capabilities, trading some flexibility for reliability, efficiency, and observability.

### Historical background
As agentic applications matured through 2023-2025, practitioners identified a recurring set of composition patterns that worked well across many applications — Anthropic's widely-referenced "Building Effective Agents" post (2024) crystallized much of this practical wisdom into a small set of named patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer), alongside frameworks like LangGraph that provide explicit graph-based tooling for building and executing such workflows.

### Real-world motivation
Production agentic applications rarely use a single, unstructured ReAct loop for everything — they compose several of this lesson's patterns together, using the simplest pattern sufficient for each sub-task, reserving full autonomous agentic loops for the specific sub-problems that genuinely require open-ended, adaptive behavior.

---

## 2. Theory

### Prompt chaining
Decompose a task into a fixed sequence of LLM calls, where each step's output feeds the next step's input — appropriate when a task can be cleanly decomposed into ordered sub-tasks (e.g., "extract key facts," then "verify facts against a source," then "compose a final summary") with limited need for the model to autonomously decide the sequence itself, trading some flexibility for predictability and easier debugging (each step's output can be checked independently).

### Routing
Use an initial LLM call (or a lightweight classifier) to determine which of several possible specialized downstream paths a given input should follow — e.g., classifying an incoming customer support query as "billing," "technical issue," or "general question," then routing to a specialized prompt/tool-set optimized for that specific category, rather than using one generic prompt/tool-set for all query types.

### Parallelization
Run multiple LLM calls (or tool invocations) **concurrently** rather than sequentially, when sub-tasks are independent — either **sectioning** (splitting a task into independent parallel sub-tasks whose results are later combined, e.g., analyzing different sections of a long document simultaneously) or **voting** (running the same task multiple times independently and aggregating results, directly Phase 6 Lesson 9's self-consistency technique, generalized here as a named workflow pattern).

### Orchestrator-workers
A central "orchestrator" LLM call dynamically decides how to break a task into sub-tasks and delegates each to "worker" LLM calls (which may themselves be simple prompts or full sub-agents), then synthesizes their results — more flexible than fixed prompt chaining (the orchestrator can adapt the decomposition to the specific input) while still more structured/predictable than a fully unstructured, single ReAct loop handling everything monolithically.

### Evaluator-optimizer
One LLM call generates a candidate response; a second, separate LLM call (or a rubric-based check) evaluates that response against specific criteria and provides feedback; the first call revises based on that feedback — iterating until the evaluator is satisfied or a maximum iteration count is reached — directly analogous to Lesson 6's retry-with-validation-feedback pattern, generalized here to open-ended quality criteria rather than strict schema validation alone.

---

## 3. Mathematical Foundations

### Latency and cost implications of each pattern (a genuine engineering tradeoff analysis)
- **Sequential chaining**: total latency $\approx \sum_i \text{latency}_i$ (each step waits for the previous).
- **Parallelization**: total latency $\approx \max_i(\text{latency}_i)$ for the parallel branches (plus a final combination step) — a direct, quantifiable latency improvement whenever sub-tasks are genuinely independent, echoing the concurrency benefits explored in Phase 1 Lesson 2's `asyncio` discussion.
- **Orchestrator-workers**: latency depends on the orchestrator's dynamically-chosen decomposition — potentially highly variable and harder to predict/bound in advance compared to a fixed chain or parallel structure.

### Voting/self-consistency's accuracy improvement, revisited formally (directly extending Phase 6 Lesson 9)
For $n$ independent samples each correct with probability $p > 0.5$, majority-vote accuracy increases toward 1 as $n$ grows (per Phase 6 Lesson 9's binomial-tail argument, itself an application of Phase 4 Lesson 5's ensembling mathematics) — but this benefit requires genuine independence/diversity across samples (e.g., via temperature-based sampling, Phase 5 Lesson 7); highly correlated samples (e.g., from a deterministic, zero-temperature call run multiple times) provide no such benefit, since they'd simply repeat the same answer/error every time.

### Evaluator-optimizer as bounded iterative refinement
Directly analogous to Phase 3 Lesson 5's iterative optimization (gradient descent refining toward a minimum) but operating in a discrete, LLM-judged quality space rather than a continuous, mathematically-defined loss landscape — convergence isn't guaranteed in the same formal sense, motivating the same practical safeguard as Lesson 3's agent loop: a bounded maximum iteration count, since there's no mathematical guarantee the evaluator-optimizer loop converges to a "good enough" result within any fixed number of steps for every possible task.

### Cost accounting across a composed workflow
For a workflow combining several patterns (e.g., routing into one of three specialized chains, each involving 2-4 sequential steps, with one parallelized voting sub-step), total cost is the sum of every individual LLM call's cost (Phase 6 Lesson 1's tokenization-based cost accounting) across whichever specific path the routing decision and orchestrator decisions actually traverse for a given input — meaning average production cost depends on the *distribution* of inputs across different workflow paths, not just a single worst-case or best-case path's cost.

---

## 4. Algorithm — Orchestrator-Workers Pattern (fully specified)

```
GIVEN a complex task and a set of available worker capabilities (specialized prompts/tools/sub-agents):
1. ORCHESTRATOR CALL: prompt an LLM with the task, asking it to:
     (a) decide how to decompose the task into sub-tasks, AND
     (b) assign each sub-task to an appropriate worker
   (this decomposition is DYNAMIC -- decided per-input, not a fixed hardcoded sequence)

2. FOR each assigned sub-task (potentially IN PARALLEL, if sub-tasks are independent):
     DISPATCH to the appropriate worker (a specialized prompt, tool call, or full sub-agent, Lesson 3)
     COLLECT the worker's result

3. SYNTHESIS CALL: prompt an LLM (often the orchestrator itself) with the ORIGINAL task
   plus ALL workers' collected results, asking it to synthesize a final, coherent response

4. (OPTIONAL) EVALUATOR CHECK: run the synthesized response through an evaluator step
   (Section 2's evaluator-optimizer pattern) before returning it, iterating if quality criteria aren't met

RETURN the final synthesized (and optionally evaluator-approved) response
```

---

## 5. Python Implementation

```python
"""agentic_workflows_core.py — prompt chaining, parallelization, and orchestrator-workers patterns"""
import concurrent.futures
from typing import Callable


def prompt_chain(initial_input: str, steps: list[Callable[[str], str]]) -> str:
    """Sequential chaining: each step's output feeds the next step's input."""
    current = initial_input
    for step in steps:
        current = step(current)
    return current


def parallel_sectioning(task_sections: list[str], worker_fn: Callable[[str], str],
                          combine_fn: Callable[[list[str]], str]) -> str:
    """Runs independent sub-tasks CONCURRENTLY, then combines results (Section 3's latency benefit)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_sections)) as executor:
        results = list(executor.map(worker_fn, task_sections))
    return combine_fn(results)


def voting_ensemble(task: str, sample_fn: Callable[[str], str], n_samples: int = 5) -> str:
    """Self-consistency / voting pattern (Phase 6 Lesson 9, generalized as a named workflow pattern)."""
    from collections import Counter
    samples = [sample_fn(task) for _ in range(n_samples)]   # each call should use temperature > 0 for diversity
    return Counter(samples).most_common(1)[0][0]


def orchestrator_workers(task: str, orchestrator_decompose_fn: Callable[[str], list[dict]],
                           worker_dispatch_fn: Callable[[dict], str],
                           synthesize_fn: Callable[[str, list[str]], str]) -> str:
    """Full orchestrator-workers pattern (Section 4)."""
    subtasks = orchestrator_decompose_fn(task)              # DYNAMIC decomposition, decided per-input

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as executor:
        worker_results = list(executor.map(worker_dispatch_fn, subtasks))

    return synthesize_fn(task, worker_results)


def evaluator_optimizer(task: str, generate_fn: Callable[[str, str | None], str],
                          evaluate_fn: Callable[[str, str], tuple[bool, str]],
                          max_iterations: int = 3) -> str:
    """Iterative refinement until an evaluator is satisfied, or max_iterations reached (Lesson 3's safeguard)."""
    feedback = None
    for _ in range(max_iterations):
        candidate = generate_fn(task, feedback)
        is_satisfactory, feedback = evaluate_fn(task, candidate)
        if is_satisfactory:
            return candidate
    return candidate   # return the LAST attempt even if not fully satisfactory -- graceful degradation


# --- Illustrative usage: a document analysis pipeline combining several patterns ---
def mock_extract_facts(section: str) -> str:
    return f"[extracted facts from: {section[:20]}...]"

def mock_combine_facts(results: list[str]) -> str:
    return " | ".join(results)

document_sections = ["Section 1 text...", "Section 2 text...", "Section 3 text..."]
combined_facts = parallel_sectioning(document_sections, mock_extract_facts, mock_combine_facts)
print(combined_facts)
```

---

## 6. Build From Scratch

**A minimal routing implementation (Section 2's routing pattern), directly extending the tool-calling/structured-output techniques from Lessons 5-6:**
```python
from enum import Enum

class QueryCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"

def classify_query(query: str, classify_llm_call: Callable[[str], str]) -> QueryCategory:
    """A lightweight classification call -- often a SMALLER/CHEAPER model than the main task model,
    since classification into a few known categories is a much simpler task than full generation."""
    category_str = classify_llm_call(query)   # in practice: structured output (Lesson 6) constrained to the Enum
    return QueryCategory(category_str)

def route_query(query: str, classify_llm_call, specialized_handlers: dict[QueryCategory, Callable]) -> str:
    category = classify_query(query, classify_llm_call)
    handler = specialized_handlers[category]
    return handler(query)

# Illustrative usage:
handlers = {
    QueryCategory.BILLING: lambda q: f"[Billing specialist response to: {q}]",
    QueryCategory.TECHNICAL: lambda q: f"[Technical specialist response to: {q}]",
    QueryCategory.GENERAL: lambda q: f"[General assistant response to: {q}]",
}
mock_classifier = lambda q: "billing" if "invoice" in q.lower() else "general"
print(route_query("Why was my invoice higher this month?", mock_classifier, handlers))
```
Using a smaller/cheaper model specifically for the routing/classification step (rather than the same large model used for full task generation) is a genuinely common, cost-effective production pattern — directly reusing this curriculum's broader "match model capability/cost to task difficulty" principle, here applied at the workflow-architecture level.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `prompt_chain`/`parallel_sectioning`/`orchestrator_workers` (manual) | LangGraph — explicit graph-based workflow definition with built-in state management, conditional edges, and visualization |
| `voting_ensemble` | Often built directly into agent frameworks as a configurable "self-consistency" or "best-of-n" sampling option |
| Manual `ThreadPoolExecutor` for parallelization | Async-native orchestration in modern agent frameworks, or `asyncio.gather` (Phase 1 Lesson 2) for I/O-bound LLM API calls specifically |
| `evaluator_optimizer` | LangGraph's cyclic graph support, or dedicated "self-refine"/"reflection" pattern implementations in various agent frameworks |

---

## 8. Visual Explanations

**The five core agentic workflow patterns (schematic):**
```
PROMPT CHAINING:        Step1 ──▶ Step2 ──▶ Step3 ──▶ Result   (sequential, predictable)

ROUTING:                          ┌──▶ Handler A
                        Classify ─┼──▶ Handler B                (one path chosen per input)
                                   └──▶ Handler C

PARALLELIZATION:        Task ──┬──▶ Worker1 ──┐
                                ├──▶ Worker2 ──┼──▶ Combine ──▶ Result   (concurrent, latency = max not sum)
                                └──▶ Worker3 ──┘

ORCHESTRATOR-WORKERS:    Orchestrator (decides decomposition DYNAMICALLY)
                                │
                       ┌────────┼────────┐
                       ▼        ▼        ▼
                   Worker A  Worker B  Worker C  ──▶ Synthesize ──▶ Result

EVALUATOR-OPTIMIZER:    Generate ──▶ Evaluate ──▶ (satisfied? RETURN : feedback ──▶ Generate again)
                              ▲_________________________________________|
```

---

## 9. Practical Examples

**Simple:** implement prompt chaining (Section 5) for a 3-step task (extract → summarize → format) and verify each step's output correctly feeds the next.
**Medium:** implement routing (Section 6) for a customer-query classification task and verify queries are correctly dispatched to the appropriate specialized handler.
**Real-world:** build an orchestrator-workers pipeline for analyzing a long actuarial/regulatory document — the orchestrator dynamically decides how to split the document into analyzable sections (rather than a fixed split), dispatches each to a worker for fact extraction, and synthesizes a final summary — comparing this dynamic approach against a simpler, fixed-chunking parallelization pattern.

---

## 10. Real Industry Use Cases

- **Anthropic's "Building Effective Agents"**: the direct, widely-cited source of this lesson's five named patterns, based on observed patterns across many real production agent deployments.
- **Customer support routing systems**: near-universally use the routing pattern to direct queries to specialized handling logic/prompts based on query category.
- **Document analysis and research synthesis tools**: commonly use orchestrator-workers or parallel-sectioning patterns to handle long documents that exceed convenient single-context-window processing.
- **Code review and content-generation quality pipelines**: frequently use the evaluator-optimizer pattern, with a dedicated "critic" LLM call reviewing and providing feedback on a "generator" LLM call's output before final delivery.

---

## 11. Common Mistakes

- Defaulting to a single, fully unstructured agentic loop (Lesson 3) for every task, even when a simpler, more predictable, cheaper pattern (a fixed prompt chain, or routing) would serve the task just as well with better reliability and lower cost/latency.
- Using voting/self-consistency with zero-temperature (fully deterministic) sampling — provides no accuracy benefit at all, since every "independent" sample would be identical.
- Not bounding evaluator-optimizer iteration counts — risking unbounded cost/latency on tasks the evaluator never judges satisfactory (directly Lesson 3's iteration-safeguard principle, resurfacing here).
- Parallelizing sub-tasks that aren't actually independent (where one sub-task's correct execution genuinely depends on another's result) — silently producing incorrect or incoherent combined results.

---

## 12. Best Practices (2026)

- Match the workflow pattern's structure/flexibility to the task's actual requirements — use the simplest pattern (a fixed chain) sufficient for predictable, well-understood sub-tasks, reserving more flexible/expensive patterns (full agentic loops, orchestrator-workers) for genuinely open-ended sub-problems.
- Use a smaller/cheaper model for routing/classification steps, reserving larger/more capable (and more expensive) models for the actual generation/reasoning steps that need them.
- Always bound iterative patterns (evaluator-optimizer, and any nested agentic sub-loops) with a maximum iteration count.
- Design workflows with observability in mind — log each step's input/output individually, so failures can be diagnosed to the specific step/pattern responsible, rather than only having visibility into the final end-to-end result.

---

## 13. Exercises

**Easy:** Implement prompt chaining for a 3-step text-processing task and verify correct sequential data flow.
**Medium:** Implement the routing pattern (Section 6) with 3 categories and verify correct dispatch on a set of test queries spanning all categories.
**Hard:** Implement the full orchestrator-workers pattern (Section 5) for a document-analysis task where the orchestrator's decomposition genuinely varies based on document structure (e.g., different numbers of sections for different documents), and verify the synthesis step correctly incorporates all workers' results.
**Mathematical:** Given $n$ independent parallel sub-tasks each taking latency drawn from a known distribution, derive the expected total latency for a parallelized workflow (max of $n$ draws) versus a sequential one (sum of $n$ draws), and compute the expected speedup for a few example distributions.
**Coding:** Implement the evaluator-optimizer pattern (Section 5) for a writing-quality-improvement task, with a genuine (not mocked) LLM-based evaluator providing specific, actionable feedback, and empirically measure how many iterations are typically needed before the evaluator is satisfied.

---

## 14. Mini Project

Build a **complete actuarial document analysis workflow combining multiple patterns**: use routing to classify an incoming document by type (policy wording, claims report, regulatory filing), dispatch to a type-specific orchestrator-workers pipeline that dynamically decomposes the document and extracts relevant structured data (Lesson 6) from each section in parallel, synthesize a final structured summary, and run it through an evaluator-optimizer pass checking for completeness/accuracy before final delivery — producing a genuinely production-shaped, multi-pattern agentic workflow directly relevant to your domain, with full logging of each step for observability.

---

## 15. Interview Preparation

- Explain the five core agentic workflow patterns (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) and give a use case for each.
- When would you choose a fixed prompt chain over a fully autonomous agentic loop, and vice versa?
- Why does voting/self-consistency require sampling diversity (non-zero temperature) to provide any accuracy benefit?
- How would you bound the cost/latency of an evaluator-optimizer loop in a production system?

---

## 16. Summary

Agentic workflows impose deliberate, engineered structure on top of Lesson 3's basic agent loop — prompt chaining for predictable sequential decomposition, routing for input-dependent specialization, parallelization for latency reduction on independent sub-tasks, orchestrator-workers for dynamic (but still structured) task decomposition, and evaluator-optimizer for bounded iterative quality refinement. Real production systems compose several of these patterns together, matching each sub-task's actual complexity and independence characteristics to the simplest sufficient pattern, reserving fully autonomous, unstructured agentic reasoning for the specific sub-problems that genuinely require it — directly setting up Lesson 8's multi-agent systems, which extend these composition patterns to multiple distinct, specialized agents working together.

---

## 17. References

- Anthropic — "Building Effective Agents" (2024, the primary source for this lesson's five named patterns)
- LangChain/LangGraph official documentation on graph-based agentic workflow construction
- Wang et al. — "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022, directly reused from Phase 6 Lesson 9 as the mathematical basis for the voting pattern)
- Madaan et al. — "Self-Refine: Iterative Refinement with Self-Feedback" (2023, closely related to the evaluator-optimizer pattern)
