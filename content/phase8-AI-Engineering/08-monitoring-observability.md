# Phase 8 · Lesson 8 — Monitoring & Observability

> Prerequisite: MLOps (Lesson 7), FastAPI/Kubernetes (Lessons 1, 3)

---

## 1. Introduction

### What is monitoring and observability?
**Monitoring** is the practice of collecting and tracking predefined metrics about a system's health and behavior (is it up? how fast is it responding? how many errors?). **Observability** is the broader, related capability of being able to ask *new*, previously-unanticipated questions about a system's internal state using the data it emits (logs, metrics, traces) — the distinction matters because production systems inevitably fail in ways nobody predicted in advance, and observability is what lets you diagnose those genuinely novel failure modes rather than only the ones you thought to build a specific dashboard for.

### Why does it exist?
Every system built across this phase — a FastAPI service (Lesson 1), a Kubernetes deployment (Lesson 3), an MLOps pipeline (Lesson 7) — will eventually behave unexpectedly in production. Monitoring and observability exist to make that unexpected behavior *visible and diagnosable* quickly, rather than discovered only when a user complains or a business metric visibly suffers.

### Historical background
Traditional monitoring (Nagios, and similar tools, 1990s-2000s) focused on simple up/down checks and threshold alerts. The "three pillars of observability" framing (metrics, logs, traces) matured through the 2010s alongside microservices architectures (where a single user request might traverse many services, making simple up/down monitoring insufficient), with modern tools (Prometheus, Grafana, OpenTelemetry) becoming the standard stack by the 2020s — now extended with LLM/AI-specific observability needs (tracking token usage, prompt/response pairs, retrieval quality, Phase 7 Lesson 10's evaluation metrics) as generative AI applications matured into production systems.

### Real-world motivation
Every RAG system, agent, and API built across Phases 7-8 needs monitoring not just for traditional software health (uptime, latency, error rate) but also AI-specific signals (is retrieval quality degrading, are hallucination rates increasing, is cost per request creeping up) — this lesson covers both.

---

## 2. Theory

### The three pillars of observability
- **Metrics**: numeric, aggregatable measurements over time (request count, latency percentiles, error rate, GPU utilization) — efficient to store/query at scale, ideal for dashboards and alerting on known conditions.
- **Logs**: discrete, timestamped event records (often unstructured or semi-structured text) — richer detail than metrics, essential for debugging a *specific* incident, but more expensive to store/query at scale.
- **Traces**: records of a single request's full path through a (potentially multi-service) system, showing exactly where time was spent at each step — essential for diagnosing latency issues in complex, multi-component pipelines (exactly Phase 7's RAG/agent architectures, where a slow response could stem from retrieval, generation, or a tool call).

### The RED and USE methods (established metric-selection frameworks)
- **RED** (for request-driven services, like your FastAPI API): **R**ate (requests/second), **E**rrors (error rate), **D**uration (latency, ideally as percentiles, Lesson 1's discussion) — a minimal, well-validated set of metrics sufficient to characterize a service's health for most purposes.
- **USE** (for resources, like a GPU or database): **U**tilization (percentage busy), **S**aturation (how much extra work is queued), **E**rrors (resource-level errors) — a complementary framework for infrastructure-level (rather than request-level) monitoring.

### AI/LLM-specific observability additions
Beyond RED/USE, generative AI applications need: **token usage** (Phase 6 Lesson 1's cost accounting, tracked per-request and aggregated), **latency-to-first-token** and **tokens-per-second** (distinct from total request latency, particularly relevant for streaming responses), **retrieval quality metrics** (Phase 7 Lesson 1's Recall@k, tracked over time), and **quality/safety signals** (Phase 7 Lesson 10's faithfulness scores, guardrail trigger rates) — none of which have a direct analog in traditional web-service monitoring.

### Alerting and alert fatigue
Alerts should fire only when human intervention is genuinely needed — poorly-tuned alerting (too sensitive, firing on normal statistical fluctuation) causes **alert fatigue**, where engineers begin ignoring or muting alerts entirely, defeating the entire purpose of the alerting system — directly connecting to Phase 3 Lesson 4's Type I/Type II error framework: an alert threshold is fundamentally a hypothesis test, and its sensitivity/specificity tradeoff must be tuned deliberately, not left at an arbitrary default.

---

## 3. Mathematical Foundations

### Percentile-based SLOs (Service Level Objectives)
Rather than a single "average latency" target, production services typically define SLOs on specific percentiles: "p99 latency < 500ms" — directly reusing Phase 3 Lesson 4's distributional thinking, since the tail (p99, p999) often matters more for user experience than the mean, which can look acceptable while a meaningful fraction of requests experience much worse performance (exactly Phase 8 Lesson 1's earlier point, now formalized as an explicit target).

### Statistical process control for alerting thresholds
Rather than a fixed, arbitrary alert threshold, a more principled approach models the metric's normal/expected variation (e.g., via a rolling mean and standard deviation, or more robust median/MAD statistics, Phase 2 Lesson 3) and alerts when a new observation falls outside an expected control band (e.g., more than $k$ standard deviations from the rolling mean) — directly extending Phase 3 Lesson 4's hypothesis-testing framework into a continuously-monitored, sequential setting, and Phase 8 Lesson 7's drift-detection statistical process control, now applied to operational metrics rather than model input distributions.

### Sampling for cost-effective tracing
Capturing a full trace for *every* request can be prohibitively expensive at high request volumes — production tracing systems commonly use **sampling** (recording only a fraction of requests in full detail, e.g., 1% under normal conditions, with elevated sampling for requests that error or exceed a latency threshold) — a direct precision/cost tradeoff, informed by the same reasoning as Phase 3 Lesson 4's sample-size/statistical-power considerations: you need enough sampled traces to reliably characterize typical behavior, without the overhead of capturing everything.

### Error budgets (directly extending Phase 8 Lesson 5's availability discussion)
If an SLO targets 99.9% availability, the remaining 0.1% is an explicit **error budget** — a quantified, intentional allowance for failures/downtime, which can be "spent" on deliberate risk (a risky deployment, an experimental feature) as long as the cumulative failure rate stays within budget — a genuinely useful reframing that converts "avoid all failures" (impossible and counterproductive) into a quantified, manageable resource allocation problem.

---

## 4. Algorithm — Setting Up Statistically-Sound Alerting (fully specified)

```
FOR each metric you want to alert on (e.g., p99 latency, error rate, token cost per request):
1. COLLECT a baseline period of normal operation data for this metric
2. COMPUTE the metric's typical central tendency and spread (median + MAD, Phase 2 Lesson 3's
   robust statistics -- preferred over mean/std for metrics that may have occasional real outliers)
3. DEFINE an alert threshold as a MULTIPLE of the spread from the central tendency
   (e.g., alert if the current value exceeds median + 4*MAD) -- NOT an arbitrary fixed number
4. IMPLEMENT the alert to fire only after SUSTAINED deviation (e.g., 3 consecutive monitoring
   intervals outside the band), NOT a single noisy data point -- directly reduces false-positive
   alert fatigue (Section 2) at the cost of slightly slower detection of GENUINE issues
5. PERIODICALLY re-establish the baseline (Section 4, step 1) as normal operating conditions evolve
   over time (e.g., after a legitimate, intentional change in typical traffic patterns)
6. ROUTE alerts appropriately: distinguish "needs immediate human attention" from "worth reviewing
   later" -- not every anomaly warrants paging someone at 3am
```

---

## 5. Python Implementation

```python
"""monitoring_core.py — RED metrics collection, statistical alerting, and AI-specific observability"""
import time
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class RequestMetric:
    timestamp: float
    duration_ms: float
    status_code: int
    tokens_used: int = 0
    endpoint: str = ""


class MetricsCollector:
    """A simplified RED-metrics collector (Section 2) with AI-specific extensions."""
    def __init__(self, window_size: int = 1000):
        self.requests: deque = deque(maxlen=window_size)

    def record(self, metric: RequestMetric) -> None:
        self.requests.append(metric)

    def compute_red_metrics(self, window_seconds: float = 60.0) -> dict:
        now = time.time()
        recent = [r for r in self.requests if now - r.timestamp <= window_seconds]
        if not recent:
            return {"rate": 0, "error_rate": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}

        durations = sorted(r.duration_ms for r in recent)
        errors = sum(1 for r in recent if r.status_code >= 500)

        return {
            "rate": len(recent) / window_seconds,
            "error_rate": errors / len(recent),
            "p50_ms": np.percentile(durations, 50),
            "p95_ms": np.percentile(durations, 95),
            "p99_ms": np.percentile(durations, 99),
            "total_tokens": sum(r.tokens_used for r in recent),           # AI-specific (Section 2)
        }


class RobustAlertThreshold:
    """Section 4's statistical alerting: median + k*MAD, requiring SUSTAINED deviation."""
    def __init__(self, k: float = 4.0, sustained_intervals: int = 3):
        self.k = k
        self.sustained_intervals = sustained_intervals
        self.baseline_values: list[float] = []
        self.consecutive_breaches = 0

    def set_baseline(self, values: list[float]) -> None:
        self.baseline_values = values

    def check(self, current_value: float) -> tuple[bool, str]:
        median = np.median(self.baseline_values)
        mad = np.median(np.abs(np.array(self.baseline_values) - median))
        threshold = median + self.k * mad

        if current_value > threshold:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0

        if self.consecutive_breaches >= self.sustained_intervals:
            return True, f"ALERT: {current_value:.2f} exceeds threshold {threshold:.2f} for {self.consecutive_breaches} consecutive checks"
        return False, f"OK: {current_value:.2f} (threshold: {threshold:.2f}, consecutive breaches: {self.consecutive_breaches})"


if __name__ == "__main__":
    collector = MetricsCollector()
    rng = np.random.default_rng(0)
    now = time.time()
    for i in range(200):
        collector.record(RequestMetric(
            timestamp=now - rng.uniform(0, 55), duration_ms=rng.gamma(2, 50),
            status_code=rng.choice([200, 200, 200, 200, 500], p=[0.94, 0.02, 0.02, 0.01, 0.01]),
            tokens_used=rng.integers(100, 2000),
        ))
    print(collector.compute_red_metrics())

    alerter = RobustAlertThreshold(k=4.0)
    alerter.set_baseline([rng.gamma(2, 50) for _ in range(100)])   # normal baseline latencies
    for latency in [95, 98, 400, 420, 410]:                        # a sustained latency SPIKE
        fired, message = alerter.check(latency)
        print(message)
```

---

## 6. Build From Scratch

**A minimal distributed-trace representation (making Section 2's "traces" pillar concrete for a RAG/agent pipeline):**
```python
import time
import uuid

class Span:
    """A single step within a larger trace -- e.g., ONE stage of a Phase 7 RAG pipeline."""
    def __init__(self, name: str, trace_id: str, parent_span_id: str | None = None):
        self.span_id = str(uuid.uuid4())[:8]
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time: float | None = None
        self.metadata: dict = {}

    def finish(self, **metadata) -> None:
        self.end_time = time.perf_counter()
        self.metadata.update(metadata)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0


class Tracer:
    def __init__(self):
        self.spans: list[Span] = []

    def start_trace(self, root_name: str) -> Span:
        trace_id = str(uuid.uuid4())[:8]
        span = Span(root_name, trace_id)
        self.spans.append(span)
        return span

    def start_child_span(self, name: str, parent: Span) -> Span:
        span = Span(name, parent.trace_id, parent_span_id=parent.span_id)
        self.spans.append(span)
        return span

    def print_trace_tree(self, trace_id: str) -> None:
        trace_spans = [s for s in self.spans if s.trace_id == trace_id]
        for span in trace_spans:
            indent = "  " if span.parent_span_id else ""
            print(f"{indent}{span.name}: {span.duration_ms:.1f}ms {span.metadata}")


# Simulating a Phase 7 RAG pipeline's trace
tracer = Tracer()
root = tracer.start_trace("rag_query")
time.sleep(0.01)

retrieval_span = tracer.start_child_span("vector_search", root)
time.sleep(0.05)                                    # simulating retrieval latency
retrieval_span.finish(chunks_retrieved=5)

generation_span = tracer.start_child_span("llm_generation", root)
time.sleep(0.3)                                       # simulating generation latency (usually dominant)
generation_span.finish(tokens_generated=150)

root.finish()
tracer.print_trace_tree(root.trace_id)
```
This directly demonstrates a trace's value for diagnosing latency: without it, you'd only know "the request took 360ms total"; with it, you can see exactly that generation (300ms) dominated over retrieval (50ms) — precisely the diagnostic information needed to know *where* to focus optimization effort (Lesson 10) in a multi-step pipeline.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `MetricsCollector` (in-memory) | Prometheus — production-grade metrics collection, storage, and querying (PromQL) at scale, with Grafana for dashboards |
| `RobustAlertThreshold` | Prometheus Alertmanager, or dedicated anomaly-detection alerting tools, supporting the same statistical-threshold logic at production scale with proper routing/deduplication |
| `Tracer`/`Span` (toy) | OpenTelemetry — the industry-standard distributed tracing framework, with exporters to Jaeger, Grafana Tempo, and other trace-visualization backends |
| No AI-specific tooling shown | LangSmith, Langfuse, Arize — dedicated LLM/RAG observability platforms tracking prompts, tokens, retrieval quality, and evaluation metrics (Phase 7 Lesson 10) specifically |

---

## 8. Visual Explanations

**The three pillars of observability:**
```
METRICS (aggregated, cheap):        LOGS (detailed, per-event):        TRACES (per-request, cross-service):
  request_rate: 150/s                 [ERROR] 14:32:01 timeout           rag_query (360ms total)
  error_rate: 0.02                     connecting to vector_db             ├─ vector_search (50ms)
  p99_latency: 450ms                   for request abc123                 └─ llm_generation (300ms)
  (good for DASHBOARDS/ALERTS)        (good for DEBUGGING one incident)  (good for LOCATING where time went)
```

**Error budget (Section 3) as a spendable resource:**
```
SLO: 99.9% availability -> 0.1% error budget per month (~43 minutes of allowed downtime)
Month so far: ████████░░  (80% of error budget consumed by a risky deployment + 2 minor incidents)
   -> remaining budget: 20% (~8.6 minutes) -> team should be MORE cautious about further risky changes
      this month, having nearly exhausted the intentionally-allotted failure allowance
```

---

## 9. Practical Examples

**Simple:** implement `MetricsCollector` (Section 5) and simulate a stream of requests, computing RED metrics over a rolling window.
**Medium:** implement the robust, sustained-deviation alerting logic (Section 5) and verify it correctly avoids firing on a single noisy spike while correctly firing on a genuine sustained latency increase.
**Real-world:** instrument your Phase 8 Lesson 1 FastAPI actuarial API with request logging capturing RED metrics plus token usage, set up percentile-based latency tracking, and implement statistically-sound alerting on both latency and error rate.

---

## 10. Real Industry Use Cases

- **Every production service at scale**: Prometheus + Grafana (or a cloud provider's managed equivalent) is the near-universal standard metrics/dashboarding stack.
- **OpenTelemetry**: increasingly the industry-standard tracing instrumentation format, vendor-neutral and supported across essentially every major observability backend.
- **LLM-specific observability platforms** (LangSmith, Langfuse, Arize): rapidly matured through 2023-2026 specifically to address the AI-specific monitoring needs (token cost, retrieval quality, hallucination rate) this lesson covers, now standard tooling for any serious production RAG/agent deployment.
- **SRE (Site Reliability Engineering) practices**: error budgets and percentile-based SLOs (Section 3) are foundational SRE concepts, pioneered publicly by Google's SRE book and now industry-standard practice.

---

## 11. Common Mistakes

- Monitoring only mean latency, missing tail-latency (p99) degradation that significantly affects user experience for a meaningful fraction of requests.
- Setting alert thresholds arbitrarily (a fixed number chosen without reference to actual baseline variation) — leads to either alert fatigue (too sensitive) or missed genuine incidents (too lenient).
- Alerting on single noisy data points rather than requiring sustained deviation — a frequent, avoidable source of alert fatigue.
- Not tracking AI-specific metrics (token cost, retrieval quality) alongside traditional RED metrics for LLM/RAG applications — missing genuinely important signals that traditional web-service monitoring alone won't surface.

---

## 12. Best Practices (2026)

- Track latency as percentiles (p50, p95, p99), not just the mean, and set SLOs on the percentile that matters for your application's user experience.
- Use robust statistics (median/MAD, Phase 2 Lesson 3) rather than mean/std for establishing alert baselines, especially for metrics prone to occasional genuine outliers.
- Require sustained deviation (multiple consecutive breaches) before firing an alert, to reduce false-positive alert fatigue.
- Implement distributed tracing for any multi-step pipeline (RAG, agentic workflows, Phase 7) to enable precise latency/failure diagnosis at the specific step responsible, not just the aggregate request.
- Track AI-specific metrics (token usage/cost, retrieval quality, guardrail trigger rates, Phase 7 Lesson 10's evaluation scores) as first-class monitored signals for any generative AI application.

---

## 13. Exercises

**Easy:** Implement `MetricsCollector` (Section 5) and compute RED metrics for a simulated stream of 500 requests with a realistic latency distribution.
**Medium:** Implement the robust alerting logic (Section 5) and test it against a synthetic metric stream containing both isolated noise spikes and a genuine sustained shift, verifying it correctly distinguishes between them.
**Hard:** Implement the tracing system (Section 6) for a simulated multi-step RAG pipeline (retrieval, re-ranking, generation) and use it to correctly identify which step is the latency bottleneck across several simulated requests with varying per-step latencies.
**Mathematical:** Given a baseline metric's median and MAD, derive what percentage of normally-distributed-equivalent data would fall within a $k=4$ MAD-based threshold, and discuss how this compares to a standard-deviation-based Gaussian threshold at a similar false-positive rate.
**Coding:** Integrate OpenTelemetry (or a simplified equivalent) into your Phase 8 Lesson 1 FastAPI application, instrumenting each major processing step, and visualize a captured trace to identify your application's actual latency bottleneck.

---

## 14. Mini Project

Build a **complete monitoring and observability stack for your actuarial RAG/agent application**: implement RED metrics collection plus AI-specific metrics (token cost, retrieval Recall@k tracked over time, Phase 7 Lesson 10's faithfulness scores), distributed tracing across your pipeline's major steps (retrieval, generation, any tool calls), statistically-sound alerting with sustained-deviation requirements on at least 3 distinct metrics, and a simple dashboard (even a basic script generating periodic reports, if a full Grafana setup isn't available) summarizing system health — directly preparing this application for genuine, ongoing production operation.

---

## 15. Interview Preparation

- Explain the three pillars of observability (metrics, logs, traces) and when each is most useful.
- Why should latency be monitored via percentiles rather than just the mean?
- What is an error budget, and how does it help balance reliability against the pace of change?
- What AI/LLM-specific metrics would you add to a standard RED-metrics monitoring setup for a RAG application?

---

## 16. Summary

Monitoring and observability make production systems' behavior visible and diagnosable — RED metrics (rate, errors, duration, tracked as percentiles) provide efficient dashboards and alerting for known conditions, logs provide rich detail for specific incident debugging, and distributed traces reveal exactly where time is spent within multi-step pipelines (Phase 7's RAG/agentic architectures especially). Statistically-sound alerting (robust baseline statistics, sustained-deviation requirements, directly reusing Phase 2 Lesson 3's robust statistics and Phase 3 Lesson 4's hypothesis-testing framework) prevents alert fatigue while still catching genuine issues, and AI-specific observability (token cost, retrieval quality, Phase 7 Lesson 10's evaluation metrics) extends this discipline to the unique signals generative AI applications require — completing the operational visibility needed to trust Lesson 7's MLOps lifecycle and Lesson 4's CI/CD pipeline are actually working as intended in production.

---

## 17. References

- Beyer, Jones, Petoff, Murphy (eds.) — *Site Reliability Engineering* (Google's SRE book, free online — the foundational source for SLOs/error budgets)
- OpenTelemetry official documentation
- Prometheus and Grafana official documentation
- Langfuse/LangSmith documentation (LLM-specific observability platforms)
