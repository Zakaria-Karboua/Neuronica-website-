# Phase 9 · Project 10 — Complete Production-Grade Insurance AI Platform

> Difficulty: ★★★★★ (capstone of capstones) | Integrates all 9 prior projects and every phase of this curriculum

---

## Overview

The final synthesis: a single, coherent platform unifying every prior project into one product — an insurer's internal AI system handling risk scoring (Project 2), document classification (Project 5), policy Q&A (Project 6), multi-step research (Project 7), and full report generation (Project 8), all continuously operated by the MLOps platform (Project 9), served through one API gateway, secured, monitored, cost-managed, and certified production-ready via Phase 8 Lesson 12's complete checklist. This project has no new ML techniques to learn — its difficulty is entirely *integration, operational discipline, and honest self-assessment*.

---

## System Architecture

```
                              ┌─────────────────────┐
                              │   API Gateway         │
                              │ (auth, rate limiting, │
                              │  routing, Lesson 9)   │
                              └──────────┬────────────┘
        ┌──────────────┬─────────────────┼─────────────────┬──────────────┐
        ▼              ▼                 ▼                 ▼              ▼
  Risk Scoring    Doc Classifier    Policy Q&A RAG    Research Agent   Report Generator
  (Project 2)      (Project 5)       (Project 6)       (Project 7)     (Project 8)
        │              │                 │                 │              │
        └──────────────┴─────────────────┼─────────────────┴──────────────┘
                                          ▼
                         Continuous MLOps Platform (Project 9)
                                          │
                                          ▼
                    Unified Observability + Cost + Security layer (Lessons 8–11)
```

---

## Folder Structure

```
insurance-ai-platform/
├── services/                    # each prior project as a subdirectory, minimally adapted
│   ├── risk-scoring/             # Project 2
│   ├── document-classifier/      # Project 5
│   ├── policy-qa/                # Project 6
│   ├── research-agent/           # Project 7
│   └── report-generator/         # Project 8
├── platform/
│   ├── gateway/                  # unified auth, routing, rate limiting
│   ├── mlops/                    # Project 9, generalized to all 5 model-bearing services
│   ├── observability/            # unified metrics/logs/traces across ALL services
│   └── cost-management/          # cross-service cost attribution and optimization
├── infra/
│   ├── terraform/                # Phase 8 Lesson 5, full platform infrastructure-as-code
│   └── k8s/                      # one manifest set per service + shared platform components
├── .github/workflows/            # per-service CI/CD + platform-level integration CI/CD
├── docs/
│   ├── system_architecture.md
│   ├── production_readiness_audit.md   # Phase 8 Lesson 12's full checklist, completed
│   ├── incident_response_runbook.md
│   └── capacity_plan.md
└── README.md
```

---

## Documentation

The complete documentation suite Phase 8 Lesson 12 describes: a system architecture document covering every service's role and interactions; a production-readiness audit (literally running that lesson's checklist tool against the *entire platform*, not just one service); an incident-response runbook covering cross-service failure scenarios (what happens if the vector DB backing Policy Q&A goes down while the Research Agent depends on it?); and a capacity plan projecting infrastructure needs 6–12 months out using Phase 4 Lesson 6's proper time-series forecasting on the platform's own historical usage data.

---

## Testing

Beyond each service's own test suite (inherited from Projects 2/5/6/7/8): **cross-service integration tests** (does the Research Agent correctly degrade if the Policy Q&A service it depends on is temporarily unavailable — Phase 8 Lesson 12's reliability-composition math made concrete); a full "game day" chaos exercise deliberately killing one service and verifying the platform's monitoring detects it (MTTD) and either auto-recovers or pages the right on-call path (MTTR) within documented targets.

---

## Dockerization

Every service retains its own container image (from its originating project); the platform adds a lightweight gateway container and shared base images for common dependencies, with a `docker-compose.yml` capable of running the *entire platform* locally for development — a genuine integration-testing capability, not just per-service isolation.

---

## CI/CD

A two-tier pipeline: each service's own CI/CD (inherited, unchanged) plus a platform-level pipeline that, on any service's deployment, runs the cross-service integration test suite before allowing the change to reach production — directly preventing a locally-passing service-level change from breaking a cross-service dependency.

---

## Deployment

Full Terraform-provisioned infrastructure (Phase 8 Lesson 5): a managed Kubernetes cluster hosting all five services plus platform components, with per-service resource allocation informed by each service's actual observed load profile (risk-scoring: high-volume, low-latency, CPU-only; document-classifier and research-agent: GPU-backed, bursty), and a documented, justified pricing-model choice per workload type (Lesson 5's decision procedure, applied five times).

---

## Monitoring

The unified observability layer (Project 9's dashboard, extended platform-wide) surfaces: per-service RED metrics, per-service AI-specific metrics (drift, cost, quality — whichever apply), cross-service request traces (a single user report-generation request touching 3+ services, fully traced end to end, Phase 8 Lesson 8), and platform-wide cost attribution (which service drives the most spend, and why — Phase 8 Lesson 11).

---

## Evaluation

Every prior project's own evaluation suite, still independently gating that service's deployments — plus a platform-level "does the whole system deliver value" evaluation: a set of realistic end-to-end user journeys (e.g., "classify this document, then answer a question about the resulting policy category, then generate a report") evaluated holistically for total latency, total cost, and final output quality.

---

## Scalability

Independent per-service autoscaling (each service's own HPA, inherited); platform-level capacity planning (the new contribution here) projecting aggregate infrastructure needs as usage grows, informed by proper time-series forecasting on historical cross-service usage data rather than per-service extrapolation alone (interaction effects — e.ks., report generation load driving both research-agent and RAG load simultaneously — matter at the platform level).

---

## Security Considerations

A unified API gateway enforcing authentication/authorization once, consistently, rather than each service reimplementing it independently (directly avoiding the entangled, inconsistent-security-per-service anti-pattern); network policies restricting inter-service communication to genuinely required paths only; a platform-wide secrets-management strategy (one vault, not five inconsistent ad hoc approaches); and a consolidated threat model document covering the full platform's attack surface, referencing and building on Project 7's per-agent threat model rather than duplicating it.

---

## Definition of Done — The Complete Curriculum Capstone Checklist

- [ ] All five services deployed, individually functional, and passing their own test suites
- [ ] Platform-level API gateway providing unified auth/routing across all services
- [ ] Cross-service integration tests passing, including a deliberate service-outage degradation test
- [ ] Full production-readiness audit (Phase 8 Lesson 12) completed and documented for the entire platform
- [ ] Unified observability dashboard operational across all services
- [ ] A "game day" chaos exercise conducted, with MTTD/MTTR measured and documented
- [ ] Capacity plan produced using proper time-series forecasting on real platform usage data
- [ ] Complete, honest documentation suite: architecture, readiness audit, runbook, capacity plan
- [ ] A final written retrospective: what worked, what you'd do differently, and which Phase 7 Lesson 8 "was multi-agent complexity actually justified" style questions you can now answer with real evidence from this platform, not assumption

---

## Closing Note

This project is the intended endpoint of the entire curriculum: every phase — programming foundations, data science, mathematics, classical ML, deep learning, LLMs, generative AI systems, and production AI engineering — contributes a load-bearing piece to this one platform. Completing it honestly (including documenting what *doesn't* work well, per Phase 7 Lesson 8's and Phase 8 Lesson 11's repeated emphasis on rigorous, unflattering self-assessment) is a more valuable outcome than a superficially complete but untested system. Treat the "Definition of Done" checklist above as a real bar, not a formality.
