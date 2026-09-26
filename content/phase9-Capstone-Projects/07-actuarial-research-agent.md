# Phase 9 · Project 7 — Actuarial Research Agent

> Difficulty: ★★★★☆ | Primary phases exercised: 7 (Lessons 3–6, 9–10), 8 (Lessons 1, 8–10)

---

## Overview

An agentic system that answers multi-step actuarial questions by autonomously deciding which tools to invoke — searching the Project 6 RAG index, performing calculations, converting units, and querying the Project 2 risk-scoring service — using native tool calling and MCP-standardized tool discovery, with full behavioral guardrails given its ability to take real actions. This project's difficulty comes from the compounding-failure-probability challenge (Phase 7 Lesson 3): reliability, not raw capability, is the central engineering problem.

---

## System Architecture

```
User query ──▶ FastAPI ──▶ ReAct agent loop (Phase 7 Lesson 3)
                                    │
                    ┌───────────────┼──────────────────┐
                    ▼                ▼                  ▼
            MCP: RAG search   MCP: Calculator    MCP: Risk-scoring API
            (Project 6)        tool               (Project 2, read-only)
                    │                │                  │
                    └────────────────┴──────────────────┘
                                    │
                                    ▼
                    Structured final answer + full transcript log
```

Tools are exposed via a real MCP server (Phase 7 Lesson 4), not hardcoded into the agent — meaning the same tool server could, in principle, be reused by an entirely different client application without modification, directly demonstrating that lesson's central architectural benefit.

---

## Folder Structure

```
actuarial-research-agent/
├── src/
│   ├── mcp_servers/
│   │   ├── rag_search_server.py       # wraps Project 6's retrieval
│   │   ├── calculator_server.py
│   │   └── risk_scoring_server.py     # wraps Project 2, READ-ONLY scope
│   ├── agent/
│   │   ├── react_loop.py              # Phase 7 Lesson 3, with max_iterations safeguard
│   │   ├── tool_schemas.py            # Phase 7 Lesson 5's schema design
│   │   └── structured_response.py     # Phase 7 Lesson 6
│   ├── guardrails/
│   │   ├── permission_scoping.py      # least-privilege tool access, Phase 8 Lesson 9
│   │   └── injection_detection.py     # defense-in-depth Layer 1
│   ├── api/main.py
│   └── evaluation/
│       ├── task_success_rate.py       # Phase 7 Lesson 3's compounding-failure metric
│       └── golden_task_set.json
├── tests/ (unit, integration, red-team/adversarial)
├── k8s/
└── docs/threat_model.md
```

---

## Documentation

`docs/threat_model.md` (Phase 8 Lesson 9's mini-project format): every tool's permission level and justification, which actions require human approval, and an honest accounting of residual prompt-injection risk given the RAG corpus and any external content the agent might process.

---

## Testing

Beyond standard unit/integration tests: a **golden task set** of realistic multi-step actuarial questions with known correct final answers, measuring end-to-end task success rate (directly Phase 7 Lesson 3's $p^n$ compounding-failure concern, made empirical); adversarial/red-team tests attempting prompt injection via a deliberately-poisoned RAG document, verifying Layer 1–5 defenses (Phase 8 Lesson 9) hold; an iteration-limit test confirming the agent gracefully times out rather than looping indefinitely.

---

## Dockerization

Each MCP server runs as its own lightweight container (a genuine microservices decomposition — the calculator server needs no GPU or heavy dependencies, while the RAG server needs vector-DB connectivity), orchestrated via `docker-compose` locally and separate Kubernetes Deployments in production, directly extending Project 6's multi-container pattern.

---

## CI/CD

Quality gate includes the golden-task-set success rate (must not regress) *and* the red-team injection-resistance test suite (must not regress) — both gating deployment, since this is the first project where a "safety" regression is as disqualifying as an "accuracy" regression.

---

## Deployment

Kubernetes: agent orchestrator Deployment + one Deployment per MCP server, with network policies restricting which services can reach the risk-scoring API (defense in depth at the network layer, not just the application layer) — the risk-scoring tool is scoped read-only at both the MCP schema level (Phase 7 Lesson 5) and the underlying service's actual API permissions (belt and suspenders).

---

## Monitoring

Per-tool invocation rate and latency (identifying which tool dominates task latency, Phase 8 Lesson 8's tracing); anomalous tool-call-sequence detection (Phase 8 Lesson 9's Layer 5); average iterations-to-completion per task (a direct efficiency/cost signal, Phase 8 Lesson 11); human-approval request rate and approval/denial ratio for high-stakes actions.

---

## Evaluation

Task success rate on the golden set with confidence intervals; average cost per completed task (Phase 8 Lesson 11); a dedicated safety evaluation (Phase 8 Lesson 10) checking the agent never takes an unapproved high-stakes action across the red-team suite.

---

## Scalability

Parallel tool execution where sub-tasks are independent (Phase 7 Lesson 7's parallelization pattern) rather than the agent's default strictly-sequential ReAct loop, reducing average task latency for tasks decomposable into independent lookups.

---

## Security Considerations

The full Phase 8 Lesson 9 defense-in-depth stack: input separation for any retrieved/untrusted content, least-privilege scoping per tool (verified by the read-only enforcement test), human approval required for any tool with write access, and continuous anomaly monitoring — documented transparently in the threat model rather than assumed solved.

---

## Definition of Done

- [ ] All three tools exposed via real MCP servers, independently deployable and testable
- [ ] Golden task set success rate measured and documented with confidence intervals
- [ ] Red-team injection tests pass against the full defense-in-depth stack
- [ ] Threat model document completed, including honest residual-risk accounting
- [ ] Per-tool monitoring dashboard operational
