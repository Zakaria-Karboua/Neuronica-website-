# Phase 7 · Lesson 8 — Multi-Agent Systems

> Prerequisite: Agentic Workflows (Lesson 7), AI Agents (Lesson 3)

---

## 1. Introduction

### What are multi-agent systems?
Architectures where multiple distinct LLM-based agents — each potentially with different roles, tools, prompts, or even different underlying models — collaborate (or compete) to accomplish a task, communicating with each other rather than operating as a single monolithic agent. This extends Lesson 7's orchestrator-workers pattern into a more general framework where "workers" can themselves be full, autonomous agents (Lesson 3) rather than simple prompt calls.

### Why does it exist?
Some tasks benefit from genuine role specialization — a "researcher" agent optimized/prompted for information gathering, a "critic" agent optimized for quality evaluation, a "writer" agent optimized for synthesis — mirroring how human teams divide complex work among specialists rather than expecting one generalist to do everything equally well. Multi-agent systems also offer natural points for parallelization, redundancy/consensus, and modular development (each agent can be developed, tested, and improved somewhat independently).

### Historical background
Multi-agent LLM research and frameworks (AutoGen, CrewAI, and others) matured rapidly through 2023-2025, exploring various communication topologies (hierarchical, peer-to-peer, debate-style) and finding that, similar to Lesson 7's broader workflow-pattern lesson, genuine multi-agent complexity is often *not* necessary — many tasks that seem to call for multiple agents can be handled equally well (and more efficiently, reliably, and cheaply) by a single well-designed agent or simpler workflow, a nuance increasingly emphasized in 2025-2026 practitioner guidance after the initial multi-agent enthusiasm.

### Real-world motivation
Understanding both the genuine benefits AND the real overhead/failure-mode risks of multi-agent architectures is essential for making a sound engineering decision about whether your task actually needs one — a decision this lesson equips you to make deliberately rather than defaulting to multi-agent complexity because it sounds sophisticated.

---

## 2. Theory

### Communication topologies
- **Hierarchical (manager-worker)**: a manager/orchestrator agent (directly extending Lesson 7's orchestrator-workers pattern) delegates sub-tasks to specialized worker agents and synthesizes their results — the most common, most predictable topology.
- **Peer-to-peer**: agents communicate directly with each other without a central coordinator, useful for genuinely collaborative tasks but harder to control/predict and more prone to coordination failures.
- **Debate/adversarial**: two or more agents argue different positions or critique each other's work, with a final synthesis or judge step — sometimes used specifically to surface considerations or errors a single agent's reasoning might miss.

### Role specialization
Each agent in a multi-agent system is typically given a distinct **role** (via its system prompt, Phase 6 Lesson 9) and potentially a distinct **toolset** (Lesson 5) — a "researcher" agent might have search/RAG tools (Lesson 1) but not code-execution tools; a "coder" agent the reverse. This specialization can improve quality (each agent's prompt/context is more focused) but introduces coordination overhead (agents must communicate their findings to each other clearly, typically via structured outputs, Lesson 6).

### Shared vs. isolated context
A genuinely important architectural decision: do agents share one common context/transcript (simpler, but risks context-window bloat and each agent seeing potentially irrelevant information from others' work), or does each agent maintain its own isolated context, communicating only specific, deliberately-passed messages/results to others (more scalable, requires more careful message-passing design)?

### When multi-agent systems genuinely help versus add unnecessary overhead
Multi-agent architectures tend to help when: (a) sub-tasks genuinely benefit from different specialized prompting/tooling, (b) sub-tasks are substantially parallelizable, or (c) an adversarial/debate structure specifically improves quality for the task type (e.g., catching errors via critique). They tend to add unnecessary overhead when a single well-prompted agent (or a simpler Lesson 7 workflow) could handle the task with less coordination complexity, latency, and cost — a genuinely important, non-obvious distinction that 2025-2026 practitioner consensus increasingly emphasizes after early over-enthusiasm for multi-agent architectures.

---

## 3. Mathematical Foundations

### Coordination overhead, quantified
For $n$ agents communicating in a fully peer-to-peer topology, the number of potential communication pathways is $O(n^2)$ (every pair can potentially communicate) — versus $O(n)$ for a hierarchical topology (each worker communicates only with the manager) — directly, quantifiably explaining why hierarchical topologies are more common and more tractable to reason about/debug as the number of agents grows, echoing Phase 1 Lesson 4's general graph-complexity intuitions.

### Compounding failure probability, extended from Lesson 3
Lesson 3 showed that per-step failure probability compounds unfavorably across sequential steps ($p^n$ for $n$ sequential steps). In a multi-agent system, if a task requires $m$ agents to each succeed at their sub-task (with some independence assumption, itself often optimistic given shared context/error-propagation risk) AND requires correct **coordination** between them (with its own failure probability $q$ per hand-off), overall success probability is roughly:
$$
P(\text{success}) \approx p^m \times q^{m-1}
$$
(one fewer hand-off than agents) — directly showing that adding more specialized agents, while potentially improving each individual sub-task's quality, also multiplies the number of failure points requiring correct coordination, a genuine, quantifiable cost of increased architectural complexity that must be weighed against any quality benefit from specialization.

### Voting/consensus among multiple agents (directly extending Lesson 7's voting pattern)
When multiple agents independently attempt the same (sub-)task and a consensus/voting mechanism combines their outputs, the same majority-vote mathematics from Phase 6 Lesson 9/Phase 4 Lesson 5 applies — genuine diversity across agents (different prompts, different models, or different reasoning paths) is what makes this beneficial; using multiple instances of literally the same agent with the same deterministic configuration provides no such benefit.

---

## 4. Algorithm — Hierarchical Multi-Agent Task Execution (fully specified)

```
GIVEN a complex task, a manager agent, and a set of specialized worker agents:
1. MANAGER analyzes the task and produces a structured decomposition plan
   (which sub-tasks exist, which worker role each should be assigned to) -- Lesson 6's structured output

2. FOR each sub-task in the plan (potentially in parallel where independent, Lesson 7):
     DISPATCH to the assigned worker agent
     the worker agent runs its OWN full agentic loop if needed (Lesson 3's ReAct, with its OWN tools)
     the worker RETURNS a structured result (Lesson 6) to the manager

3. MANAGER reviews all worker results:
     IF results are sufficient and coherent: proceed to synthesis
     IF a result is insufficient/inconsistent: either re-dispatch to the SAME worker with feedback
        (Lesson 7's evaluator-optimizer pattern, applied at the multi-agent level), or reassign/replan

4. MANAGER synthesizes a final response from all (accepted) worker results

5. (OPTIONAL) a separate CRITIC/EVALUATOR agent reviews the final synthesized response
   before it's returned, potentially triggering another round of revision

RETURN the final response (with full transcript/logs of every agent's individual contribution, for observability)
```

---

## 5. Python Implementation

```python
"""multi_agent_core.py — a hierarchical manager-worker multi-agent system"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentResult:
    agent_role: str
    content: str
    success: bool = True
    metadata: dict = field(default_factory=dict)


class SpecializedAgent:
    """A worker agent with its OWN role/prompt/tools -- runs its own agentic loop if the task needs it."""
    def __init__(self, role: str, run_fn: Callable[[str], AgentResult]):
        self.role = role
        self.run_fn = run_fn   # in practice: wraps a full Lesson 3 ReAct loop, specialized per role

    def execute(self, subtask: str) -> AgentResult:
        return self.run_fn(subtask)


class ManagerAgent:
    def __init__(self, workers: dict[str, SpecializedAgent], decompose_fn: Callable[[str], list[dict]],
                  synthesize_fn: Callable[[str, list[AgentResult]], str]):
        self.workers = workers
        self.decompose_fn = decompose_fn
        self.synthesize_fn = synthesize_fn

    def run(self, task: str, max_retries_per_worker: int = 1) -> str:
        plan = self.decompose_fn(task)   # e.g., [{"role": "researcher", "subtask": "..."}, ...]
        results: list[AgentResult] = []

        for step in plan:
            worker = self.workers[step["role"]]
            result = worker.execute(step["subtask"])
            retries = 0
            while not result.success and retries < max_retries_per_worker:
                # Lesson 7's evaluator-optimizer pattern, applied at the agent-coordination level
                result = worker.execute(step["subtask"] + f"\n(Previous attempt failed: {result.content})")
                retries += 1
            results.append(result)

        return self.synthesize_fn(task, results)


# --- Illustrative example: a research + analysis + writing pipeline ---
def researcher_run(subtask: str) -> AgentResult:
    return AgentResult(agent_role="researcher", content=f"[researched facts for: {subtask}]")

def analyst_run(subtask: str) -> AgentResult:
    return AgentResult(agent_role="analyst", content=f"[analysis of: {subtask}]")

def writer_run(subtask: str) -> AgentResult:
    return AgentResult(agent_role="writer", content=f"[written summary for: {subtask}]")

def mock_decompose(task: str) -> list[dict]:
    return [
        {"role": "researcher", "subtask": f"gather facts about {task}"},
        {"role": "analyst", "subtask": f"analyze findings about {task}"},
        {"role": "writer", "subtask": f"write a summary about {task}"},
    ]

def mock_synthesize(task: str, results: list[AgentResult]) -> str:
    return f"Final report on '{task}':\n" + "\n".join(f"- [{r.agent_role}]: {r.content}" for r in results)


manager = ManagerAgent(
    workers={
        "researcher": SpecializedAgent("researcher", researcher_run),
        "analyst": SpecializedAgent("analyst", analyst_run),
        "writer": SpecializedAgent("writer", writer_run),
    },
    decompose_fn=mock_decompose,
    synthesize_fn=mock_synthesize,
)
print(manager.run("actuarial mortality trends"))
```

---

## 6. Build From Scratch

**A minimal debate/critique pattern (Section 2's adversarial topology), directly extending Lesson 7's evaluator-optimizer:**
```python
def debate_and_synthesize(question: str, position_a_fn, position_b_fn, judge_fn, rounds: int = 2) -> str:
    """Two agents argue different angles; a judge synthesizes a final, more robust answer."""
    transcript_a, transcript_b = [], []
    for round_num in range(rounds):
        argument_a = position_a_fn(question, transcript_b)   # sees the OTHER side's previous argument
        argument_b = position_b_fn(question, transcript_a)
        transcript_a.append(argument_a)
        transcript_b.append(argument_b)

    return judge_fn(question, transcript_a, transcript_b)

# Illustrative mocks
mock_position_a = lambda q, opposing: f"[Argument FOR, round {len(opposing)+1}, responding to: {opposing[-1] if opposing else 'nothing yet'}]"
mock_position_b = lambda q, opposing: f"[Argument AGAINST, round {len(opposing)+1}, responding to: {opposing[-1] if opposing else 'nothing yet'}]"
mock_judge = lambda q, a, b: f"Synthesized judgment considering both sides:\nFOR: {a}\nAGAINST: {b}"

result = debate_and_synthesize("Should this claim be approved?", mock_position_a, mock_position_b, mock_judge)
print(result)
```
This debate pattern is specifically useful when a single agent's reasoning might have systematic blind spots (e.g., favoring approval by default) — forcing explicit consideration of an opposing position can surface considerations a single, non-adversarial pass would miss, at the cost of additional LLM calls (Section 3's coordination-overhead accounting).

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `ManagerAgent`/`SpecializedAgent` | AutoGen, CrewAI — dedicated multi-agent orchestration frameworks with built-in role definitions, message-passing infrastructure, and conversation management |
| Manual retry-per-worker logic | LangGraph's cyclic graph support for retry/revision loops at the individual-agent-node level |
| `debate_and_synthesize` | Some frameworks provide built-in debate/critique patterns; otherwise typically hand-built on top of a general orchestration framework |

---

## 8. Visual Explanations

**Hierarchical multi-agent topology (manager-worker):**
```
                        Manager Agent
                       (decomposes, dispatches, synthesizes)
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              Researcher   Analyst    Writer
              (own tools,  (own tools, (own tools,
               own loop)    own loop)   own loop)
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                    Manager synthesizes final result
```

**Coordination overhead: hierarchical O(n) vs peer-to-peer O(n²):**
```
Hierarchical (n=4 workers):        Peer-to-peer (n=4 agents, fully connected):
   Manager                            A───B
   /  |  \  \                         │╲ ╱│
  W1  W2 W3  W4                       │ ╳ │
  (4 communication pathways)          │╱ ╲│
                                       C───D
                                    (6 communication pathways -- grows as n²)
```

---

## 9. Practical Examples

**Simple:** implement the hierarchical `ManagerAgent`/`SpecializedAgent` pattern (Section 5) with 2 simple mocked workers and verify correct dispatch and synthesis.
**Medium:** implement the debate pattern (Section 6) for a decision-making task and compare its output against a single-agent (non-debate) approach on the same question.
**Real-world:** build a multi-agent actuarial analysis system with a "data-retrieval" agent (using Lesson 1's RAG tools), an "analysis" agent (performing calculations, Lesson 5's tool calling), and a "reporting" agent (synthesizing a final structured report, Lesson 6), and compare its output quality and cost/latency against a single, well-prompted agent handling the entire task alone.

---

## 10. Real Industry Use Cases

- **Complex research/analysis tools**: multi-agent systems with researcher/analyst/writer role division are common in advanced document-analysis and report-generation products.
- **Software development agent teams**: some coding-agent products use multi-agent patterns (a "planner," a "coder," a "reviewer/tester") mirroring human software team roles.
- **Content moderation and quality-review pipelines**: debate/critique patterns are used specifically to catch errors or biased outputs that a single-pass agent might miss.
- **2025-2026 industry recalibration**: growing practitioner consensus that many tasks initially built as multi-agent systems can be simplified to single-agent or simpler Lesson 7 workflow patterns without meaningful quality loss, at substantial cost/latency/complexity savings — a genuinely important, hard-won lesson worth taking seriously rather than defaulting to multi-agent architectures by default.

---

## 11. Common Mistakes

- Defaulting to a multi-agent architecture for tasks that a single well-prompted agent (or a simpler Lesson 7 workflow) could handle equally well — adding coordination overhead, cost, and failure points (Section 3's math) without a commensurate quality benefit.
- Using a fully peer-to-peer topology for tasks that don't genuinely require it, incurring $O(n^2)$ coordination complexity and making the system's behavior much harder to predict/debug than a hierarchical alternative.
- Not accounting for compounding failure probability across multiple agent hand-offs (Section 3) when estimating overall system reliability for a genuinely multi-step, multi-agent task.
- Running "debate" or "voting" patterns with insufficiently diverse agents (same prompt, same model, same temperature=0 configuration) — providing no genuine benefit over a single agent call, per Section 3's diversity requirement.

---

## 12. Best Practices (2026)

- Before building a multi-agent system, seriously evaluate whether a single agent or a simpler Lesson 7 workflow pattern would suffice — the 2025-2026 practitioner consensus strongly favors simplicity by default, reserving genuine multi-agent complexity for tasks that clearly benefit from role specialization, parallelization, or adversarial critique.
- Prefer hierarchical (manager-worker) topologies over fully peer-to-peer ones for most applications, given their more tractable coordination complexity and easier debuggability.
- Ensure genuine diversity (different prompts, different tool access, or different underlying models) when using voting/debate patterns — homogeneous agents provide no benefit.
- Build full observability (logging every individual agent's inputs/outputs, not just the final synthesized result) into any multi-agent system from the start — essential for diagnosing which specific agent/hand-off is responsible when the overall system produces a poor result.

---

## 13. Exercises

**Easy:** Implement the `ManagerAgent`/`SpecializedAgent` pattern (Section 5) with 3 mocked worker roles and verify correct task decomposition and dispatch.
**Medium:** Implement retry-with-feedback for a failing worker (Section 5) and verify the manager correctly re-dispatches with additional context on failure.
**Hard:** Implement the debate pattern (Section 6) using a real LLM API for both positions and the judge, on a genuinely debatable question, and qualitatively assess whether the debate surfaces considerations a single-pass answer would have missed.
**Mathematical:** Using Section 3's compounding-failure formula, compute overall system success probability for a 4-agent hierarchical system with per-agent success probability 0.9 and per-hand-off coordination success probability 0.95, and compare against a hypothetical single-agent system with an 0.85 success probability on the same overall task.
**Coding:** Build a simple cost/latency comparison: implement the same task (e.g., document summarization) as both a single-agent solution and a 3-agent hierarchical solution, and measure total tokens used and wall-clock time for both approaches on the same set of test documents.

---

## 14. Mini Project

Build **and critically evaluate** a multi-agent actuarial reporting system: implement a hierarchical manager-worker architecture with at least 3 specialized agents (e.g., data retrieval via RAG, Lesson 1; calculation via tool calling, Lesson 5; and structured report synthesis, Lesson 6), fully instrumented with logging for observability; then build an equivalent single-agent solution to the same task; and produce a genuinely honest comparison report covering output quality, total cost (tokens/API calls), latency, and failure modes for both approaches — directly practicing the "is multi-agent complexity actually justified here" evaluation this lesson emphasizes, rather than assuming the more sophisticated-sounding architecture is automatically better.

---

## 15. Interview Preparation

- Explain the difference between hierarchical and peer-to-peer multi-agent topologies, and the coordination-complexity tradeoff between them.
- When does a multi-agent architecture genuinely improve on a single well-designed agent, and when does it just add unnecessary overhead?
- How does failure probability compound across a multi-agent pipeline with multiple hand-offs, and what does this imply for reliability engineering?
- What's required for a voting/debate pattern among multiple agents to provide genuine benefit, versus adding cost without improving quality?

---

## 16. Summary

Multi-agent systems extend Lesson 7's orchestrator-workers pattern to full, potentially autonomous specialized agents (Lesson 3) collaborating via hierarchical, peer-to-peer, or debate-style topologies — offering genuine benefits (role specialization, parallelization, adversarial error-catching) at the real, quantifiable cost of increased coordination complexity ($O(n)$ to $O(n^2)$ communication pathways depending on topology) and compounding failure probability across agent hand-offs. The most important, hard-won practical lesson from 2023-2026's rapid multi-agent experimentation is restraint: many tasks are better served by a single well-prompted agent or a simpler Lesson 7 workflow pattern, and genuine multi-agent architecture should be a deliberate engineering decision justified by the specific task's characteristics, not a default reached for because it sounds more sophisticated — a discipline this lesson's mini project directly practices.

---

## 17. References

- Wu et al. — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)
- CrewAI official documentation
- Anthropic — "Building Effective Agents" (2024, directly relevant to when simpler patterns suffice)
- Du et al. — "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023, the debate-pattern research foundation)
