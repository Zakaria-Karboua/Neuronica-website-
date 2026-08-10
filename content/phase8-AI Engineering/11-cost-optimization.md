# Phase 8 · Lesson 11 — Cost Optimization

> Prerequisite: Cloud (Lesson 5), Monitoring & Observability (Lesson 8), Phase 6 Lessons 1, 8 (Tokenization, Quantization)

---

## 1. Introduction

### What does cost optimization cover?
Systematic techniques for reducing the financial cost of running AI/ML systems in production — spanning LLM API/inference costs (dominated by token usage, Phase 6 Lesson 1), infrastructure costs (compute/storage, Lesson 5), and training costs (Phase 6 Lesson 5's pretraining, or fine-tuning, Lessons 6-7) — while maintaining acceptable quality and performance, treating cost as a first-class engineering constraint alongside correctness and latency, not an afterthought addressed only when a bill arrives unexpectedly high.

### Why does it exist?
LLM-based applications can have genuinely surprising, rapidly-scaling costs — a seemingly reasonable per-request cost multiplied across a growing user base, or an inefficient RAG pipeline making unnecessarily large/frequent LLM calls, can produce a bill that undermines an otherwise successful application's viability. Cost optimization exists to make these costs visible, predictable, and deliberately controlled, rather than discovered as an unpleasant surprise.

### Historical background
Cost optimization has always mattered for cloud infrastructure (Lesson 5's pricing-model decisions), but LLM-specific cost optimization matured rapidly through 2023-2026 as organizations discovered that naive RAG/agentic implementations (Phase 7) could consume tokens far more liberally than initially anticipated — driving the development of specific techniques (caching, model routing, quantization) covered in this lesson.

### Real-world motivation
Every RAG/agentic system built in Phase 7, every fine-tuned model from Phase 6, has a real, ongoing operational cost — this lesson equips you to estimate, monitor, and deliberately reduce that cost without sacrificing the application's actual value.

---

## 2. Theory

### The token economics of LLM API costs (directly extending Phase 6 Lesson 1)
LLM API costs are typically billed per token (input and output tokens, often at different rates, with output tokens usually costing more since they require actual generation compute) — meaning cost scales directly with: how much context is included per request (system prompt, RAG-retrieved chunks, Phase 7 Lesson 1's chunking decisions, conversation history, Phase 7 Lesson 9's memory management), and how many requests are made (each agentic loop iteration, Phase 7 Lesson 3, is a separate billed request).

### Model routing / right-sizing
Not every request needs the most capable (and most expensive) available model — a simple classification or routing task (Phase 7 Lesson 7's routing pattern) can often be handled by a smaller, cheaper model, reserving larger models for genuinely complex sub-tasks requiring their additional capability — directly the same "match capability/cost to task difficulty" principle introduced in Phase 7 Lesson 7, now framed explicitly as a cost-optimization lever.

### Caching
Many LLM applications have genuine query/response overlap (common questions, repeated sub-tasks within an agentic workflow) — caching previously-computed responses (or intermediate results, like RAG retrieval results for a repeated query) avoids redundant, costly LLM calls entirely for repeated or highly similar requests, a direct, often substantial cost reduction with essentially zero quality tradeoff when implemented correctly (only caching genuinely deterministic or acceptably-stale-tolerant results).

### Quantization and efficient serving (directly extending Phase 6 Lesson 8 and Lesson 4 of this phase)
For self-hosted models, quantization reduces both memory footprint and often inference compute cost; efficient serving techniques (Grouped-Query Attention's KV-cache reduction, Phase 6 Lesson 4; batching multiple requests together to better utilize GPU compute) directly reduce the compute cost per served request — all techniques already covered elsewhere in this curriculum, now framed explicitly through a cost lens.

### Prompt and context optimization
Reducing unnecessary tokens in the system prompt, RAG context (Phase 7 Lesson 1's chunk size/count decisions), and conversation history (Phase 7 Lesson 9's summarization) directly reduces per-request cost — a genuinely available lever that doesn't require any infrastructure change, just more disciplined prompt/context engineering.

---

## 3. Mathematical Foundations

### Total cost decomposition for an LLM application
$$
\text{Total Cost} = \sum_{\text{requests}} \left( \text{input\_tokens} \times \text{price}_{\text{in}} + \text{output\_tokens} \times \text{price}_{\text{out}} \right)
$$
For an agentic system (Phase 7 Lesson 3) with $n$ average iterations per task, and each iteration's context growing due to the accumulating transcript, total cost per task grows **super-linearly** with $n$ (each subsequent iteration re-sends a longer accumulated context) — directly motivating both minimizing iteration count (Phase 7 Lesson 3's efficiency argument) and context/memory management (Phase 7 Lesson 9's summarization) as genuine cost levers, not just latency ones.

### Caching hit-rate economics
If a fraction $h$ (hit rate) of requests can be served from cache at near-zero marginal cost, and the remainder cost $c$ per request:
$$
\text{Average cost per request} = (1-h) \times c
$$
Even a modest cache hit rate provides proportional cost savings — e.g., $h=0.3$ (30% of requests served from cache) directly reduces average cost by 30%, a substantial, easily-achievable win for applications with genuine query repetition (common in customer-support-style applications with frequently-asked questions).

### Model routing's expected cost/quality tradeoff
If a cheap model correctly handles a fraction $p$ of requests at cost $c_{cheap}$, and the remaining $(1-p)$ genuinely require escalation to an expensive model at cost $c_{expensive}$ (plus, typically, the cheap model's cost already spent attempting it first):
$$
\text{Expected cost} = p \cdot c_{cheap} + (1-p)(c_{cheap} + c_{expensive})
$$
This is favorable whenever $c_{cheap} \ll c_{expensive}$ and $p$ is reasonably high — directly quantifying why routing/escalation architectures (Phase 7 Lesson 7) can provide substantial average cost savings even when a meaningful fraction of requests still require the expensive model.

### Quantization's compute-cost relationship (directly extending Phase 6 Lesson 8)
Lower-precision inference (Phase 6 Lesson 8's INT8/INT4 quantization) can reduce both memory bandwidth requirements and, on hardware with dedicated low-precision support, actual compute time — directly reducing the compute-hours (and thus dollar cost) needed to serve a given request volume, at the precision/quality tradeoff already characterized in that lesson.

---

## 4. Algorithm — A Cost-Optimization Audit Procedure (fully specified)

```
GIVEN a production LLM application with observed cost data (Lesson 8's monitoring):
1. BREAK DOWN total cost by component:
     which endpoints/features consume the most tokens?
     what fraction is input vs. output tokens? (informs where trimming effort matters most)
     what fraction comes from agentic loop iterations vs. single-shot requests?

2. FOR the highest-cost components, IDENTIFY applicable optimization levers:
     a. Can a cheaper/smaller model handle this task adequately? (Section 2's routing)
     b. Is there genuine request/response overlap enabling caching? (Section 2's caching)
     c. Is context/prompt size larger than genuinely necessary? (Section 2's context optimization)
     d. For agentic tasks: can iteration count be reduced (better tool design, Phase 7 Lesson 5;
        better task decomposition, Phase 7 Lesson 7)?
     e. For self-hosted models: would quantization (Phase 6 Lesson 8) meaningfully reduce cost
        without unacceptable quality loss?

3. IMPLEMENT the highest-expected-value optimizations FIRST (estimate savings via Section 3's formulas)

4. VALIDATE that quality hasn't meaningfully regressed (Phase 4 Lesson 3/Phase 7 Lesson 10's
   evaluation rigor) as a result of any cost optimization -- cost savings that silently
   degrade quality are not a genuine win

5. CONTINUOUSLY MONITOR (Lesson 8) cost metrics going forward, treating cost as a tracked,
   alertable metric alongside latency/error rate, not a quarterly afterthought
```

---

## 5. Python Implementation

```python
"""cost_optimization_core.py — cost tracking, model routing decisions, and caching"""
import hashlib
import time
from dataclasses import dataclass


# Illustrative per-token pricing (representative of real-world tiered pricing structures)
MODEL_PRICING = {
    "small_model": {"input": 0.0001, "output": 0.0003},    # $ per 1K tokens
    "large_model": {"input": 0.003, "output": 0.015},
}


@dataclass
class CostTracker:
    total_cost: float = 0.0
    request_count: int = 0
    cache_hits: int = 0

    def record_request(self, model: str, input_tokens: int, output_tokens: int, from_cache: bool = False) -> float:
        self.request_count += 1
        if from_cache:
            self.cache_hits += 1
            return 0.0
        pricing = MODEL_PRICING[model]
        cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
        self.total_cost += cost
        return cost

    def report(self) -> dict:
        hit_rate = self.cache_hits / self.request_count if self.request_count else 0
        return {
            "total_cost": round(self.total_cost, 4),
            "request_count": self.request_count,
            "cache_hit_rate": round(hit_rate, 3),
            "avg_cost_per_request": round(self.total_cost / max(1, self.request_count - self.cache_hits), 6),
        }


class ResponseCache:
    """A simple exact-match cache -- Section 2's caching lever, applied to deterministic-enough queries."""
    def __init__(self, ttl_seconds: float = 3600):
        self.cache: dict[str, tuple[str, float]] = {}
        self.ttl_seconds = ttl_seconds

    def _key(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        key = self._key(query)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return response
            del self.cache[key]   # expired
        return None

    def set(self, query: str, response: str) -> None:
        self.cache[self._key(query)] = (response, time.time())


def route_by_complexity(query: str, complexity_classifier_fn) -> str:
    """Section 2's model-routing lever: cheap classification decides which model handles the FULL request."""
    is_complex = complexity_classifier_fn(query)
    return "large_model" if is_complex else "small_model"


if __name__ == "__main__":
    tracker = CostTracker()
    cache = ResponseCache()

    queries = ["What is the mortality rate at age 65?", "What is the mortality rate at age 65?", "Complex multi-step actuarial analysis..."]
    mock_classifier = lambda q: "complex" in q.lower()

    for query in queries:
        cached_response = cache.get(query)
        if cached_response:
            tracker.record_request("small_model", 0, 0, from_cache=True)
            print(f"CACHE HIT: {query[:40]}...")
            continue

        model = route_by_complexity(query, mock_classifier)
        input_tokens, output_tokens = 150, 80   # illustrative -- real values from actual tokenization
        cost = tracker.record_request(model, input_tokens, output_tokens)
        cache.set(query, f"[response to: {query[:20]}]")
        print(f"{model} call: ${cost:.6f} for {query[:40]}...")

    print(tracker.report())
```

---

## 6. Build From Scratch

**A minimal cost-projection tool (extending Section 3's formulas to forecast at scale):**
```python
def project_monthly_cost(avg_input_tokens: int, avg_output_tokens: int, requests_per_day: int,
                            model_pricing: dict, cache_hit_rate: float = 0.0) -> dict:
    effective_requests_per_day = requests_per_day * (1 - cache_hit_rate)   # Section 3's caching formula
    daily_cost = effective_requests_per_day * (
        (avg_input_tokens / 1000) * model_pricing["input"] + (avg_output_tokens / 1000) * model_pricing["output"]
    )
    return {
        "daily_cost": round(daily_cost, 2),
        "monthly_cost": round(daily_cost * 30, 2),
        "annual_cost": round(daily_cost * 365, 2),
        "cache_savings_pct": round(cache_hit_rate * 100, 1),
    }

no_cache = project_monthly_cost(2000, 500, 10_000, MODEL_PRICING["large_model"], cache_hit_rate=0.0)
with_cache = project_monthly_cost(2000, 500, 10_000, MODEL_PRICING["large_model"], cache_hit_rate=0.3)
print("Without caching:", no_cache)
print("With 30% cache hit rate:", with_cache)
print(f"Monthly savings from caching: ${no_cache['monthly_cost'] - with_cache['monthly_cost']:.2f}")
```
This directly demonstrates the cost-forecasting discipline every production LLM application should apply *before* scaling up — projecting realistic monthly/annual costs from average per-request token counts and expected volume, and quantifying the concrete savings available from even a modest caching implementation.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `CostTracker`/`ResponseCache` | LLM provider dashboards (usage/cost tracking built into API consoles); dedicated LLM cost-monitoring platforms (often bundled with Phase 8 Lesson 8's observability tools, e.g., Langfuse tracking cost alongside quality metrics) |
| Manual model routing | LLM gateway/proxy tools (e.g., LiteLLM) providing built-in routing, caching, and fallback logic across multiple model providers |
| `project_monthly_cost` (manual) | Cloud provider cost calculators (Lesson 5) for infrastructure costs; LLM-specific cost calculators published by API providers |

---

## 8. Visual Explanations

**Cost decomposition and optimization levers:**
```
Total LLM API cost
       │
   ┌───┴───┬────────┬─────────────┐
   ▼       ▼        ▼             ▼
Input    Output   Number of    Model choice
tokens   tokens   requests     (per-token price)
   │       │        │             │
   ▼       ▼        ▼             ▼
Trim     Trim     Cache/       Route to
context  verbosity reduce      cheaper model
(Ph.7    (prompt   iterations  when adequate
Les.1/9) design)   (Ph.7 Les.3/7)
```

**Model routing's expected-cost tradeoff (Section 3):**
```
100 requests, 80% handled adequately by CHEAP model, 20% need ESCALATION to expensive model:

ALL requests to expensive model:        Routed (cheap-first, escalate when needed):
  100 × $0.05 = $5.00                     80 × $0.005 (cheap, sufficient)
                                          + 20 × ($0.005 + $0.05) (cheap attempt + expensive escalation)
                                          = $0.40 + $1.10 = $1.50   <- substantial savings
```

---

## 9. Practical Examples

**Simple:** implement `CostTracker` (Section 5) and simulate 100 requests across two model tiers, computing total cost and average cost per request.
**Medium:** implement the response cache (Section 5) and measure the cost reduction from caching on a query stream with realistic repetition patterns (e.g., 20-30% repeated queries).
**Real-world:** audit your Phase 7 actuarial RAG/agent application's actual token usage (via Lesson 8's monitoring instrumentation), identify the highest-cost components, and implement at least one optimization (routing, caching, or context trimming), measuring the before/after cost difference.

---

## 10. Real Industry Use Cases

- **Every production LLM application at meaningful scale**: cost optimization is now standard, expected engineering practice, not an afterthought — many organizations report LLM API costs as a top-line operational expense requiring active management.
- **LLM gateway/routing products** (LiteLLM, and similar): directly commercialize the routing/caching patterns covered in this lesson, letting applications automatically route requests across multiple providers/models based on cost and capability.
- **Self-hosted model serving cost optimization**: organizations serving their own fine-tuned models (Phase 6 Lessons 6-7) apply quantization (Phase 6 Lesson 8) and efficient serving techniques (Phase 6 Lesson 4's GQA/KV-caching, batching) specifically to reduce the GPU-hours needed per served request.
- **FinOps practices for AI**: the broader "FinOps" (financial operations) discipline, originally developed for general cloud cost management (Lesson 5), has extended specifically to address LLM/AI cost governance as a distinct sub-discipline by 2026.

---

## 11. Common Mistakes

- Using the most capable (and most expensive) available model for every request regardless of actual task complexity — leaving substantial, easily-captured cost savings from routing unused.
- Not implementing caching for applications with genuine query repetition — repeatedly paying for identical or near-identical LLM calls.
- Allowing agentic loops (Phase 7 Lesson 3) to run with excessive, unnecessary iterations due to poor tool design or task decomposition — directly inflating cost super-linearly per Section 3's accumulating-context argument.
- Treating cost optimization as a one-time audit rather than an ongoing, monitored practice (Lesson 8) — costs can silently creep up as usage patterns or context sizes evolve over time.

---

## 12. Best Practices (2026)

- Implement model routing by default for any application with genuinely variable task complexity, reserving expensive models for sub-tasks that actually require their capability.
- Cache aggressively wherever genuine query/response overlap exists and staleness tolerance permits it.
- Actively manage context size (RAG chunk count/size, conversation history summarization, Phase 7 Lesson 9) rather than defaulting to maximal context inclusion "just in case."
- Track cost as a first-class, continuously monitored metric (Lesson 8) alongside latency and quality, with the same alerting discipline applied to unexpected cost spikes as to performance regressions.

---

## 13. Exercises

**Easy:** Implement `CostTracker` (Section 5) and compute the cost difference between routing all requests to a large model versus routing appropriately between small and large models for a simulated query stream.
**Medium:** Implement the response cache (Section 5) and measure cost savings across query streams with varying repetition rates (10%, 30%, 50%), plotting the resulting cost-reduction curve.
**Hard:** Build a complete cost-projection and optimization-recommendation tool: given observed usage patterns (average tokens per request, request volume, complexity distribution), recommend a specific combination of routing/caching optimizations and project the resulting cost savings.
**Mathematical:** Derive, using Section 3's model-routing formula, the break-even complexity-classification accuracy $p$ at which a routing architecture becomes cost-effective compared to always using the expensive model, for a given ratio of $c_{cheap}$ to $c_{expensive}$.
**Coding:** Instrument a real (or realistically simulated) agentic pipeline (Phase 7 Lesson 3) with per-iteration cost tracking, and identify whether cost grows linearly or super-linearly with iteration count for your specific implementation, directly testing Section 3's theoretical claim.

---

## 14. Mini Project

Conduct a **complete cost audit and optimization of your Phase 7 actuarial RAG/agent application**: instrument comprehensive cost tracking (per-endpoint, per-component, input vs. output token breakdown); identify the highest-cost components and apply at least two optimization levers (model routing for appropriate sub-tasks, caching for repeated queries, context/prompt trimming); validate via Phase 7 Lesson 10's evaluation framework that quality hasn't meaningfully regressed; and produce a before/after cost report projecting monthly/annual savings at your application's expected usage scale.

---

## 15. Interview Preparation

- Explain the main cost drivers for an LLM-based application and the corresponding optimization levers for each.
- How does model routing reduce average cost, and what determines whether it's a favorable tradeoff for a given application?
- Why does an agentic loop's cost tend to grow super-linearly with iteration count, and what mitigates this?
- How would you validate that a cost-optimization change hasn't silently degraded application quality?

---

## 16. Summary

Cost optimization treats LLM/AI application cost as a first-class, continuously monitored engineering constraint — token economics (directly extending Phase 6 Lesson 1) determine per-request cost, with model routing (matching task complexity to model capability/cost, extending Phase 7 Lesson 7), caching (exploiting genuine query repetition), context/prompt optimization (Phase 7 Lessons 1 and 9), and quantization/efficient serving (Phase 6 Lessons 4 and 8, for self-hosted models) providing the primary, quantifiable levers for reducing cost without sacrificing quality — validated, critically, via the same evaluation rigor (Phase 4 Lesson 3, Phase 7 Lesson 10) applied to any other engineering change, ensuring cost savings never come at an unacknowledged quality cost. This discipline, combined with Lesson 8's ongoing monitoring, is what keeps every system built across Phases 6-7 economically sustainable at real production scale.

---

## 17. References

- LLM provider pricing documentation and cost-calculator tools (OpenAI, Anthropic, and others)
- LiteLLM official documentation (multi-provider routing and cost-management gateway)
- FinOps Foundation resources (finops.org) — the broader cloud/AI cost-governance discipline
- Langfuse/LangSmith documentation (cost-tracking features alongside quality observability, Lesson 8)
