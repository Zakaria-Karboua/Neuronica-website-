# Phase 9 · Project 8 — Multi-Agent Insurance Report Generator

> Difficulty: ★★★★☆ | Primary phases exercised: 7 (Lessons 7–10), 8 (Lessons 1, 4, 8, 11)

---

## Overview

A hierarchical multi-agent system producing complete underwriting/risk reports: a manager agent decomposes an incoming case, dispatches to specialized researcher/analyst/writer sub-agents (each wrapping Projects 2, 6, and 7's capabilities), and synthesizes a final structured report — with a mandatory single-agent baseline built *first* and kept running in parallel, so this project directly practices Phase 7 Lesson 8's central discipline: proving multi-agent complexity is actually justified rather than assuming it.

---

## System Architecture

```
Case intake ──▶ Manager Agent (decomposes, dispatches, synthesizes)
                        │
        ┌───────────────┼────────────────┐
        ▼                ▼                 ▼
  Researcher Agent   Analyst Agent    Writer Agent
  (Project 6's RAG)  (Project 2's     (structured
                       scoring +       report synthesis,
                       Project 7's      Phase 7 Lesson 6)
                       calculations)
        │                │                 │
        └────────────────┴─────────────────┘
                        ▼
              Evaluator agent (quality/completeness check, Phase 7 Lesson 7)
                        │
                        ▼
              Final report (or feedback loop back to Manager if inadequate)

  [In parallel, for EVERY request: a single well-prompted agent baseline runs
   the same task, for the mandatory comparison — Section: Evaluation]
```

---

## Folder Structure

```
multi-agent-report-generator/
├── src/
│   ├── agents/
│   │   ├── manager.py
│   │   ├── researcher.py
│   │   ├── analyst.py
│   │   ├── writer.py
│   │   └── evaluator.py            # Phase 7 Lesson 7's evaluator-optimizer pattern
│   ├── single_agent_baseline/
│   │   └── baseline_agent.py       # the mandatory comparison point
│   ├── orchestration/
│   │   ├── hierarchical_dispatch.py
│   │   └── observability_logging.py  # per-agent transcript logging
│   ├── api/main.py
│   └── comparison/
│       ├── quality_comparison.py
│       └── cost_latency_comparison.py
├── tests/ (per-agent unit tests, full-pipeline integration tests)
├── k8s/ (one Deployment per agent role + the baseline)
└── docs/
    ├── architecture_decision_record.md   # Phase 8 Lesson 4's ADR practice
    └── multi_agent_vs_single_agent_report.md
```

---

## Documentation

`docs/multi_agent_vs_single_agent_report.md` is the project's centerpiece deliverable: a rigorous, honest comparison of the multi-agent architecture against the single-agent baseline on quality, cost, and latency — following exactly Phase 7 Lesson 8's mini-project discipline. If the multi-agent system doesn't clearly win, that's a legitimate, valuable finding to report, not a failure.

---

## Testing

Per-agent unit tests (each specialized agent tested in isolation with mocked upstream dependencies); full hierarchical-pipeline integration tests; a coordination-failure test (Phase 7 Lesson 8's Section 3 math made concrete — deliberately fail one sub-agent and verify the manager's retry-with-feedback logic recovers or fails gracefully); a diversity check for the evaluator agent (verifying it isn't simply the same model/prompt as the writer, which would provide no genuine independent review value, Phase 7 Lesson 8's diversity requirement).

---

## Dockerization

One container image per agent role (enabling independent scaling — the researcher agent, hitting the RAG index, may need different resource allocation than the writer agent) plus the baseline agent's own container, all sharing a common base image layer (Phase 8 Lesson 2's caching efficiency) to avoid redundant dependency installation across five near-identical images.

---

## CI/CD

Quality gate evaluates the *multi-agent system's* output against the single-agent baseline's output on a fixed test-case set — deployment of a multi-agent architecture change is blocked unless it maintains or improves its margin over the baseline, directly operationalizing the "is the added complexity still justified" question as an ongoing, automated check rather than a one-time decision.

---

## Deployment

Kubernetes: five agent Deployments (manager, researcher, analyst, writer, evaluator) plus the baseline, each independently scalable via their own HPA configuration (Phase 8 Lesson 3) — reflecting that different agent roles may have genuinely different load profiles (the manager is called once per request; sub-agents may be invoked variably depending on task decomposition).

---

## Monitoring

Per-agent-role latency/cost breakdown (via distributed tracing, Phase 8 Lesson 8, showing exactly which agent dominates total pipeline time/cost); coordination-failure rate (how often a sub-agent's result is rejected/retried by the manager); the ongoing multi-agent-vs-baseline quality/cost delta, tracked continuously in production, not just at initial launch.

---

## Evaluation

The full three-axis comparison (quality via LLM-as-judge validated against human ratings, Phase 7 Lesson 10; cost via Phase 8 Lesson 11's tracking; latency via Phase 8 Lesson 8's tracing) between the multi-agent pipeline and the single-agent baseline — with bootstrap confidence intervals on every reported difference, not just point estimates.

---

## Scalability

Independent per-role autoscaling (above) plus request-level parallelization of independent sub-agent calls (Phase 7 Lesson 7) within the manager's dispatch logic wherever the case decomposition permits it.

---

## Security Considerations

Coordination-complexity risk (Phase 7 Lesson 8's $O(n)$ vs $O(n^2)$ topology argument) is itself a security-relevant concern here: the hierarchical (not peer-to-peer) topology is a deliberate choice partly *because* it's more tractable to audit and monitor for anomalous behavior (Phase 8 Lesson 9's Layer 5) than a fully-connected alternative would be.

---

## Definition of Done

- [ ] Single-agent baseline fully functional and evaluated first, before multi-agent work begins
- [ ] Multi-agent pipeline implemented with genuine role diversity (verified, not assumed)
- [ ] Rigorous quality/cost/latency comparison report completed with confidence intervals
- [ ] Coordination-failure recovery tested and working
- [ ] Per-role independent scaling verified under simulated load
