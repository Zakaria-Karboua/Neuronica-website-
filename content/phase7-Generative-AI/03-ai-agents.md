# Phase 7 · Lesson 3 — AI Agents

> Prerequisite: RAG, Vector Databases (Lessons 1–2), Phase 6 Lesson 9 (Prompt Engineering), Phase 4 Lesson 7 (RL Introduction)

---

## 1. Introduction

### What is an AI agent?
A system built around an LLM that can **take actions** in the world (not just generate text) by reasoning about a goal, deciding which action to take next (including calling external tools, Lesson 5), observing the result, and iterating — a fundamentally different usage pattern from a single-turn prompt-response interaction, closer in structure to the sequential decision-making framework of Phase 4 Lesson 7's reinforcement learning, though the "policy" here is a pretrained/fine-tuned LLM reasoning in natural language rather than a trained-from-scratch RL policy.

### Why does it exist?
Many real tasks require more than a single LLM response: multi-step research requiring several searches, tasks requiring interaction with external systems (databases, APIs, calculators), or tasks whose full solution isn't knowable upfront and must be discovered through iterative exploration. Agents give an LLM the structure to plan, act, observe, and revise — extending its usefulness far beyond single-turn text generation.

### Historical background
Early "agent" patterns (2022-2023) emerged from combining LLMs with the classic **ReAct** framework (Yao et al., 2022) — interleaving *Reasoning* traces (chain-of-thought, Phase 6 Lesson 9) with *Acting* steps (tool calls) and *Observations* (tool results), in a repeated loop. This simple but powerful pattern remains the conceptual foundation of virtually every LLM agent framework in 2026, even as specific implementations (LangGraph, and others) have added more sophisticated state management and control flow around it.

### Real-world motivation
Every "AI assistant that can browse the web," "coding agent that can run and debug code," or "research agent that synthesizes information from multiple sources" is built on this lesson's ReAct-style loop — understanding it demystifies what's actually happening inside these systems rather than treating them as magic.

---

## 2. Theory

### The ReAct loop — the foundational agent pattern

$$
\text{Thought} \to \text{Action} \to \text{Observation} \to \text{Thought} \to \text{Action} \to \dots \to \text{Final Answer}
$$

At each step, the LLM generates a **thought** (reasoning about what to do next, exploiting chain-of-thought's benefits, Phase 6 Lesson 9), decides on an **action** (which tool to call and with what arguments, Lesson 5), receives an **observation** (the tool's actual result), and incorporates that observation into its next thought — continuing until it determines the task is complete and produces a final answer.

### Why interleaving reasoning and acting helps (versus reasoning-only or acting-only)
Pure chain-of-thought (no tool use) can only reason using information already in the model's context/parameters — it cannot look up a genuinely new fact or perform a precise calculation. Pure action-only (no explicit reasoning trace) loses the benefit of explicit intermediate reasoning shown to improve multi-step task performance (Phase 6 Lesson 9). ReAct's interleaving lets the model use reasoning to *decide* what to look up next, and lets tool results *inform* subsequent reasoning — a genuinely synergistic combination validated extensively since its introduction.

### Agent "scratchpad" / working memory
The full history of thoughts, actions, and observations within one agent episode is typically kept in the LLM's context window as a running transcript (a "scratchpad") — directly a context-window-budget concern (Lesson 1's constraint, resurfacing here), since a long-running agent task can accumulate a substantial transcript, eventually requiring summarization or truncation strategies (Lesson 8's memory systems territory).

### Planning strategies beyond simple ReAct
More sophisticated agent architectures separate **planning** (deciding a multi-step plan upfront) from **execution** (carrying out each step, possibly replanning if a step fails or reveals new information) — sometimes called "plan-and-execute" patterns, offering better structure for genuinely long, complex tasks compared to ReAct's more purely reactive, step-by-step approach, at the cost of additional architectural complexity.

---

## 3. Mathematical Foundations

### Framing agents within the MDP framework (directly connecting to Phase 4 Lesson 7)
An LLM agent's interaction loop maps naturally onto the Markov Decision Process formalism: the **state** is the current context/transcript (thoughts + actions + observations so far), the **action space** is the set of available tools (plus "produce final answer"), and the **policy** is the LLM's next-token generation process, implicitly selecting actions via its generated text. Unlike Phase 4 Lesson 7's trained-from-scratch RL policies, an LLM agent's "policy" comes from pretraining + fine-tuning (Phase 6 Lessons 5-6) rather than task-specific RL training — though RLHF-style techniques are increasingly applied specifically to improve agentic tool-use behavior.

### The cost of iteration — a genuine engineering tradeoff
Each ReAct loop iteration requires a full LLM forward pass (an API call, in practice) — for a task requiring $n$ iterations, total latency and cost scale roughly linearly with $n$ (each iteration's cost, per Phase 6 Lesson 4's inference-cost discussion, itself scaling with the accumulated context length, so total cost can actually scale *super*-linearly as the transcript grows across iterations). This directly motivates efficient tool selection (fewer, more effective iterations) and context management (Lesson 8) as genuine cost/latency engineering concerns, not just abstract elegance.

### Success/failure as a bounded random process
For agents performing multi-step tasks with per-step success probability $p$ (independent, simplifying assumption), the probability of completing an $n$-step task successfully is $p^n$ — even a high per-step success rate compounds unfavorably over many steps (e.g., $p=0.95$ over $n=20$ steps gives only $0.95^{20}\approx 0.36$ overall success probability) — a sobering, quantifiable argument for why minimizing the number of required steps, and building in error-recovery/retry mechanisms, matters enormously for reliable agent behavior at real task lengths.

---

## 4. Algorithm — The ReAct Agent Loop (fully specified)

```
GIVEN a task/goal, a set of available tools, and a max_iterations limit:
transcript = [initial task description]
FOR iteration = 1 to max_iterations:
    1. THOUGHT: prompt the LLM with the transcript so far, asking it to reason about
       what to do next (chain-of-thought style, Phase 6 Lesson 9)
    2. ACTION: parse the LLM's output for a chosen action (a tool call + arguments,
       Lesson 5's structured-output territory) OR a "final answer" signal
    3. IF final answer signaled:
         RETURN the final answer
    4. ELSE:
         EXECUTE the chosen tool with the given arguments
         OBSERVATION: capture the tool's actual result (success or error)
         APPEND (thought, action, observation) to the transcript
IF max_iterations reached without a final answer:
    RETURN a graceful failure/timeout response (NEVER silently loop forever)
```
The `max_iterations` safeguard is not optional in any real implementation — without it, a malfunctioning or genuinely stuck agent can loop indefinitely, incurring unbounded cost and latency, a critical, practical safety mechanism.

---

## 5. Python Implementation

```python
"""ai_agents_core.py — a minimal, from-scratch ReAct agent loop"""
import re
from typing import Callable


class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name, self.func, self.description = name, func, description

    def run(self, *args, **kwargs) -> str:
        try:
            return str(self.func(*args, **kwargs))
        except Exception as e:
            return f"ERROR: {e}"   # observations should surface errors gracefully, not crash the loop


def calculator_tool(expression: str) -> float:
    # NEVER use raw eval() in production -- shown simplified here; use a safe math parser in practice
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)


def mock_search_tool(query: str) -> str:
    """A stand-in for a real search/retrieval tool (Lesson 1's RAG, or a real web search API)."""
    knowledge = {
        "brent crude price": "Brent crude oil was trading around $78/barrel as of the latest data.",
        "dzd exchange rate": "The Algerian Dinar (DZD) exchange rate is approximately 134 DZD per USD.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found."


TOOLS = {
    "calculator": Tool("calculator", calculator_tool, "Evaluates a mathematical expression."),
    "search": Tool("search", mock_search_tool, "Searches for factual information."),
}


def parse_agent_output(llm_output: str) -> dict:
    """Parses a structured LLM response following a simple ReAct-style format convention."""
    action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", llm_output)
    final_match = re.search(r"Final Answer:\s*(.*)", llm_output)
    if final_match:
        return {"type": "final_answer", "content": final_match.group(1).strip()}
    if action_match:
        return {"type": "action", "tool": action_match.group(1), "input": action_match.group(2)}
    return {"type": "unknown", "content": llm_output}


def run_react_agent(task: str, llm_call: Callable[[str], str], max_iterations: int = 8) -> str:
    """llm_call: a function taking a prompt string and returning the LLM's response (Phase 6's territory)."""
    transcript = [f"Task: {task}"]
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS.values())

    for iteration in range(max_iterations):
        prompt = (
            f"You are an agent with access to these tools:\n{tool_descriptions}\n\n"
            f"Think step by step. Use 'Action: tool_name[input]' to use a tool, "
            f"or 'Final Answer: ...' when done.\n\n" + "\n".join(transcript)
        )
        llm_output = llm_call(prompt)
        parsed = parse_agent_output(llm_output)

        if parsed["type"] == "final_answer":
            return parsed["content"]

        if parsed["type"] == "action" and parsed["tool"] in TOOLS:
            observation = TOOLS[parsed["tool"]].run(parsed["input"])
            transcript.append(f"Thought/Action: {llm_output}")
            transcript.append(f"Observation: {observation}")
        else:
            transcript.append(f"Thought: {llm_output}")
            transcript.append("Observation: Could not parse a valid action. Please use the correct format.")

    return "Agent did not reach a final answer within the iteration limit."   # Section 4's safeguard
```

---

## 6. Build From Scratch

**A minimal success-probability simulator (making Section 3's compounding-failure math concrete):**
```python
import numpy as np

def simulate_multistep_task_success(per_step_success_prob: float, n_steps: int, n_trials: int = 10000) -> float:
    rng = np.random.default_rng(0)
    successes = 0
    for _ in range(n_trials):
        step_outcomes = rng.random(n_steps) < per_step_success_prob
        if step_outcomes.all():     # task succeeds ONLY if EVERY step succeeds
            successes += 1
    return successes / n_trials

for n_steps in [1, 5, 10, 20, 50]:
    empirical_rate = simulate_multistep_task_success(per_step_success_prob=0.95, n_steps=n_steps)
    theoretical_rate = 0.95 ** n_steps
    print(f"n_steps={n_steps}: empirical={empirical_rate:.3f}, theoretical={theoretical_rate:.3f}")
```
Running this concretely demonstrates Section 3's sobering compounding-failure math — even a 95%-reliable-per-step agent succeeds on barely more than a third of 20-step tasks, directly motivating both minimizing required steps and building explicit retry/error-recovery logic into any production agent system.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `run_react_agent` (manual parsing, manual loop) | LangGraph, LangChain's agent executors — handle robust output parsing, retries, streaming, and more sophisticated control flow (conditional branching, parallel tool calls) |
| Regex-based action parsing | Native structured/function-calling APIs (Lesson 5) — far more reliable than regex-parsing free-form text, the now-standard approach |
| Manual `max_iterations` safeguard | Built-in iteration/timeout limits and cost-tracking in production agent frameworks |
| No memory/summarization (Lesson 8) | LangGraph's persistent state management, or custom summarization strategies for long-running agent transcripts |

---

## 8. Visual Explanations

**The ReAct loop:**
```
  Task: "What is Brent crude price, converted to DZD per barrel?"
         │
         ▼
  Thought: "I need the Brent crude price first."
  Action: search[brent crude price]
  Observation: "Brent crude oil ~$78/barrel"
         │
         ▼
  Thought: "Now I need the DZD exchange rate."
  Action: search[dzd exchange rate]
  Observation: "~134 DZD per USD"
         │
         ▼
  Thought: "Now I can calculate: 78 * 134"
  Action: calculator[78 * 134]
  Observation: "10452"
         │
         ▼
  Final Answer: "Brent crude is approximately 10,452 DZD per barrel."
```

**Compounding failure probability across steps (Section 3/6):**
```
Success rate
  1.0 ●
      │  ╲___
  0.5 │      ╲___
      │          ╲______
  0.0 │                 ╲____________
      └──────────────────────────────  number of sequential steps
      1        5       10       20      (per-step success = 0.95)
```

---

## 9. Practical Examples

**Simple:** run the `run_react_agent` (Section 5) with a mocked LLM function that follows a scripted sequence of thoughts/actions, verifying the loop correctly executes tools and terminates.
**Medium:** integrate a real LLM API as the `llm_call` function and test the agent on a genuine multi-step task requiring both the search and calculator tools.
**Real-world:** build an actuarial research agent that can search a RAG-indexed collection of regulatory documents (Lesson 1) and perform calculations (e.g., computing a loss ratio from retrieved figures), testing it on realistic multi-step questions and measuring how often it succeeds within a reasonable iteration budget.

---

## 10. Real Industry Use Cases

- **Coding agents** (GitHub Copilot Workspace, Claude Code, and similar): use exactly this ReAct-style loop — reasoning about what code change is needed, running/testing it (the "action"), observing the result (test pass/fail, error messages), and iterating.
- **Research/browsing agents**: interleave web search actions with reasoning to synthesize multi-source answers to complex questions.
- **Customer support agents**: combine RAG-based knowledge retrieval (Lesson 1) with tool calls to backend systems (checking order status, processing a refund) within an agentic loop.
- **Data analysis agents**: iteratively query databases, run computations, and refine analysis based on intermediate results, directly relevant to potential SANAVIR agricultural-data or actuarial-analysis applications.

---

## 11. Common Mistakes

- No iteration limit (or an excessively high one) — risks runaway cost/latency if an agent gets stuck in an unproductive loop.
- Relying on fragile regex-based parsing of free-form LLM output to extract actions, rather than using native structured/function-calling APIs (Lesson 5) — a common source of silent parsing failures.
- Not handling tool errors gracefully — a tool failure should become an informative observation the agent can reason about and recover from, not a crash that halts the entire agent.
- Allowing the transcript to grow unboundedly across many iterations without any summarization/truncation strategy (Lesson 8), eventually exceeding the context window or degrading performance due to excessive irrelevant history.

---

## 12. Best Practices (2026)

- Use native function/tool-calling APIs (Lesson 5) rather than free-form-text action parsing whenever the underlying model supports it — dramatically more reliable.
- Always set a sensible `max_iterations` (and/or a cost/time budget) as a hard safeguard against runaway agent loops.
- Design tools to return clear, LLM-parseable error messages on failure, so the agent can reason about and recover from tool errors rather than getting stuck.
- Monitor and log full agent transcripts in production — essential for debugging unexpected agent behavior and for building evaluation datasets (Lesson 10).

---

## 13. Exercises

**Easy:** Implement a simple calculator-only agent loop and test it on a multi-step arithmetic word problem requiring 2-3 calculator calls.
**Medium:** Extend the Section 5 agent with a new tool (e.g., a unit converter) and test it on a task requiring all three tools in combination.
**Hard:** Implement a "plan-and-execute" agent variant (generate a full multi-step plan upfront, then execute each step, replanning if a step fails) and compare its behavior/reliability against the pure ReAct loop on a genuinely multi-step task.
**Mathematical:** Using the Section 6 simulator, empirically determine what per-step success probability is needed to achieve at least 90% overall success on a 15-step task, and compare against the theoretical $p^{15} \ge 0.9$ calculation.
**Coding:** Add a retry mechanism to the Section 5 agent loop (if a tool call fails, allow the agent up to 2 retries with revised input before giving up on that step) and measure whether it improves overall task success rate on a task with an intentionally flaky tool.

---

## 14. Mini Project

Build a **complete actuarial research agent**: give it access to a RAG-based search tool (Lesson 1, over a corpus of regulatory/actuarial reference documents), a calculator tool, and a unit-conversion tool; test it on a set of realistic multi-step actuarial questions (e.g., requiring looking up a rate, retrieving a formula, and computing a result); measure success rate, average iterations-to-completion, and failure modes across your test set; and write a short report on where the agent succeeds reliably versus where it needs better tooling, prompting, or error-recovery logic.

---

## 15. Interview Preparation

- Explain the ReAct pattern and why interleaving reasoning and acting outperforms either alone.
- How would you frame an LLM agent's interaction loop using the MDP formalism from reinforcement learning?
- Why does agent task success probability degrade so sharply as the number of required steps increases?
- What safeguards are essential in any production agent loop implementation?

---

## 16. Summary

AI agents extend LLMs from single-turn text generation into iterative, tool-using systems via the ReAct pattern — interleaving explicit reasoning (chain-of-thought, Phase 6 Lesson 9) with concrete actions (tool calls, Lesson 5) and observations (tool results), structurally mirroring the sequential decision-making framework of Phase 4 Lesson 7's reinforcement learning, though driven by a pretrained LLM's reasoning rather than a from-scratch-trained RL policy. The sobering mathematics of compounding per-step failure probability across multi-step tasks directly motivates this lesson's essential safeguards (iteration limits, graceful error handling) and sets up the need for the more structured tool-calling (Lesson 5), workflow orchestration (Lesson 7), and multi-agent (Lesson 8) patterns covered in the rest of this phase.

---

## 17. References

- Yao et al. — "ReAct: Synergizing Reasoning and Acting in Language Models" (2022, the foundational agent paper)
- Wei et al. — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022, directly reused from Phase 6 Lesson 9)
- LangChain/LangGraph official documentation on agent executors and control flow
- Anthropic — "Building Effective Agents" (engineering blog post, widely-cited practical guidance on agent architecture patterns)
