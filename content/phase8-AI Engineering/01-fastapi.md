# Phase 8 · Lesson 1 — FastAPI

> Prerequisite: Phase 1 (Advanced Python's `asyncio`, OOP), Phase 6-7 (something worth serving)

---

## 1. Introduction

### What is FastAPI?
A modern Python web framework (Ramírez, 2018) for building APIs, built on top of Starlette (ASGI web toolkit) and Pydantic (Phase 7 Lesson 6's schema validation library) — designed specifically around Python type hints, automatic request/response validation, and native `async`/`await` support (Phase 1 Lesson 2), making it the dominant choice for serving ML/LLM models and building the backend of essentially every AI application covered in Phases 6-7.

### Why does it exist?
Serving a trained model (or an LLM-based pipeline) to real users requires an HTTP API — accepting requests, validating input, running inference, returning structured responses, and handling many concurrent requests efficiently. FastAPI exists to make this fast to build (automatic validation/documentation from type hints, minimal boilerplate) and fast to run (native async support, critical for I/O-bound LLM API calls and RAG pipeline steps).

### Historical background
FastAPI emerged specifically to combine Flask's simplicity with modern Python's type-hint ecosystem and native async capabilities, which older frameworks (Flask, Django) either lacked or added later/less natively. Its automatic OpenAPI/Swagger documentation generation (directly from your code's type hints and Pydantic models) and built-in data validation made it rapidly popular for ML/API serving specifically, where request/response schema correctness matters enormously.

### Real-world motivation
Every model you've built across Phases 4-7 — an XGBoost classifier, a fine-tuned LLM, a full RAG pipeline — eventually needs to be wrapped in an API for real applications to consume; FastAPI is very likely the tool you'll use to do that.

---

## 2. Theory

### ASGI and async request handling
FastAPI is built on ASGI (Asynchronous Server Gateway Interface), the async-native successor to WSGI (the older, synchronous standard Flask/Django originally used) — allowing a single worker process to handle many concurrent requests efficiently when those requests spend most of their time waiting on I/O (exactly the profile of an LLM API call or a RAG pipeline's retrieval step, Phase 1 Lesson 2's `asyncio` concepts directly applying here).

### Pydantic models as the request/response contract
Every endpoint's expected input and output shape is defined via Pydantic models (Phase 7 Lesson 6) — FastAPI automatically validates incoming request bodies against these models (rejecting malformed requests with a clear error *before* your endpoint logic even runs) and serializes your endpoint's return value according to the declared response model, giving you Phase 7 Lesson 6's structured-output reliability guarantees on both sides of your API.

### Dependency injection
FastAPI's `Depends()` system lets you declare reusable, composable pieces of logic (authentication checks, database connections, loaded ML models) that are automatically resolved and injected into your endpoint functions — a direct, practical application of the Dependency Inversion principle (Phase 1 Lesson 8) and the Strategy/Factory patterns (Phase 1 Lesson 10), letting you swap implementations (e.g., a mock model for testing vs. a real one for production) without changing endpoint code.

### Automatic API documentation
FastAPI generates interactive OpenAPI (Swagger) documentation automatically from your type hints and Pydantic models — genuinely free documentation that's always in sync with your actual code (since it's derived from the code itself, not maintained separately), directly addressing Phase 1 Lesson 8's documentation-maintenance concerns.

---

## 3. Mathematical Foundations

FastAPI is primarily a software-engineering topic, but a few quantitative framings matter for production serving:

### Throughput under async I/O-bound workloads
For $n$ concurrent requests each spending $t_{io}$ time waiting on I/O (an LLM API call) and $t_{cpu}$ time on actual computation, a single-threaded async server can achieve throughput approaching $1/t_{cpu}$ requests/second (I/O wait time overlaps across requests, Phase 1 Lesson 2's event-loop concurrency) rather than $1/(t_{io}+t_{cpu})$ for a naive synchronous server processing one request fully before starting the next — a direct, quantifiable justification for using FastAPI's async endpoints for any I/O-bound (especially LLM-API-calling) workload.

### Latency percentiles as the right way to characterize API performance
Rather than reporting only mean latency, production API monitoring (Lesson 8, this phase) reports percentiles (p50, p95, p99) — directly reusing Phase 3 Lesson 4's distributional thinking: mean latency can look acceptable while a meaningful fraction of requests (the tail) experience much worse latency, which is often what actually determines user-perceived quality for a latency-sensitive application.

### Queueing theory intuition (Little's Law, revisited from Phase 1 Lesson 2)
$$
L = \lambda W
$$
For an API server, average number of in-flight requests $L$ equals arrival rate $\lambda$ times average time-in-system $W$ — directly informing capacity planning: if you know your expected request rate and target latency, you can estimate how many concurrent requests your server must be able to handle, informing worker-count and infrastructure sizing decisions (Lessons 2-3, this phase).

---

## 4. Algorithm — Request Lifecycle Through a FastAPI Endpoint (fully specified)

```
1. REQUEST arrives at the ASGI server (Uvicorn, typically)
2. ROUTING: FastAPI matches the request path/method to the correct endpoint function
3. DEPENDENCY RESOLUTION: any Depends() declarations are resolved (auth checks, DB connections,
   loaded model instances) -- potentially recursively, if a dependency itself has dependencies
4. REQUEST VALIDATION: the request body/query params/path params are validated against the
   endpoint's declared Pydantic models -- INVALID requests are rejected here with a 422 error,
   BEFORE your endpoint function's own code ever executes
5. ENDPOINT EXECUTION: your function runs (awaited, if async def) with validated, typed inputs
6. RESPONSE SERIALIZATION: your return value is validated/serialized against the declared response_model
7. RESPONSE sent back to the client, with automatically-generated OpenAPI docs reflecting this
   entire contract available at /docs
```

---

## 5. Python Implementation

```python
"""fastapi_core.py — a production-shaped ML/LLM model-serving API"""
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time


class MortalityPredictionRequest(BaseModel):
    age: int = Field(ge=0, le=120, description="Age in years")
    smoker: bool
    region_risk_score: float = Field(ge=-3, le=3)


class MortalityPredictionResponse(BaseModel):
    mortality_probability: float
    risk_tier: str
    model_version: str


class ModelState:
    """Holds the loaded model, initialized ONCE at startup (not per-request -- expensive to reload)."""
    model = None
    version = "v1.2.0"


ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: load the model ONCE (directly relevant given Phase 6's model-loading cost)
    ml_models["mortality_model"] = lambda age, smoker, risk: (
        1 / (1 + 2.71828 ** -(-4 + 0.05 * age + 1.2 * smoker + 0.3 * risk))
    )
    yield
    # SHUTDOWN: release resources here if needed
    ml_models.clear()


app = FastAPI(title="Actuarial Model API", version="1.0.0", lifespan=lifespan)


def get_model():
    """A DEPENDENCY (Section 2) -- swappable for a mock model in tests via dependency_overrides."""
    if "mortality_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ml_models["mortality_model"]


def verify_api_key(x_api_key: str = "") -> bool:
    """A simple auth DEPENDENCY -- in production, use a real auth scheme (OAuth2, Lesson 9's security territory)."""
    if x_api_key != "expected-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@app.post("/predict/mortality", response_model=MortalityPredictionResponse)
async def predict_mortality(
    request: MortalityPredictionRequest,
    model=Depends(get_model),
    _authorized: bool = Depends(verify_api_key),
) -> MortalityPredictionResponse:
    start = time.perf_counter()
    probability = model(request.age, request.smoker, request.region_risk_score)
    tier = "high" if probability > 0.05 else ("medium" if probability > 0.01 else "low")

    elapsed = time.perf_counter() - start
    if elapsed > 1.0:   # a real system would log this to a monitoring pipeline (Lesson 8)
        print(f"WARNING: slow prediction took {elapsed:.2f}s")

    return MortalityPredictionResponse(
        mortality_probability=round(probability, 5), risk_tier=tier, model_version=ModelState.version
    )


@app.get("/health")
async def health_check() -> dict:
    """Essential for Kubernetes liveness/readiness probes (Lesson 3)."""
    return {"status": "healthy", "model_loaded": "mortality_model" in ml_models}
```

**Run with:** `uvicorn fastapi_core:app --reload` (development) or a production ASGI server configuration (Lesson 2-3's containerization/orchestration).

---

## 6. Build From Scratch

**A minimal WSGI-vs-ASGI concurrency demonstration (to make Section 3's async throughput claim concrete):**
```python
import asyncio
import time

async def simulate_io_bound_request(request_id: int, io_delay: float = 0.5) -> str:
    await asyncio.sleep(io_delay)   # simulates an LLM API call or DB query (Phase 1 Lesson 2's asyncio)
    return f"Request {request_id} completed"

async def handle_requests_concurrently(n_requests: int, io_delay: float = 0.5):
    start = time.perf_counter()
    tasks = [simulate_io_bound_request(i, io_delay) for i in range(n_requests)]
    results = await asyncio.gather(*tasks)                 # ASYNC: all requests' I/O waits OVERLAP
    elapsed = time.perf_counter() - start
    print(f"Async (concurrent): {n_requests} requests in {elapsed:.2f}s")
    return results

def handle_requests_sequentially(n_requests: int, io_delay: float = 0.5):
    start = time.perf_counter()
    for i in range(n_requests):
        time.sleep(io_delay)                                # SYNC: each request's I/O wait BLOCKS the next
    elapsed = time.perf_counter() - start
    print(f"Sync (sequential): {n_requests} requests in {elapsed:.2f}s")

asyncio.run(handle_requests_concurrently(10, io_delay=0.3))   # ~0.3s total (all overlap)
handle_requests_sequentially(10, io_delay=0.3)                 # ~3.0s total (fully sequential)
```
This directly, empirically demonstrates Section 3's throughput claim: 10 requests each with 0.3s of I/O wait complete in roughly 0.3s total when handled concurrently (async), versus roughly 3.0s when handled sequentially — precisely why FastAPI's async endpoint support matters enormously for any LLM-API-calling or RAG-pipeline-serving workload.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| Manual request validation | FastAPI's automatic Pydantic-based validation, with zero extra code beyond type hints |
| Manual API documentation | FastAPI's automatic OpenAPI/Swagger docs at `/docs`, generated directly from code |
| `simulate_io_bound_request` illustration | Real Uvicorn/Gunicorn ASGI server running actual FastAPI async endpoints at production scale |
| Flask (older, WSGI-based alternative) | FastAPI — native async, automatic validation/docs; Flask requires additional libraries (Flask-RESTX, etc.) to approach equivalent functionality |

---

## 8. Visual Explanations

**Synchronous vs. asynchronous request handling (I/O-bound workload):**
```
SYNCHRONOUS (WSGI-style):              ASYNCHRONOUS (ASGI/FastAPI):
Req1: [wait I/O.......][process]        Req1: [wait I/O.......][process]
                        Req2: [wait...]  Req2:  [wait I/O.......][process]
                                 Req3:.. Req3:   [wait I/O.......][process]
   (fully sequential -- SLOW)              (I/O waits OVERLAP -- much faster overall throughput)
```

**FastAPI's request lifecycle (Section 4):**
```
Request ──▶ Routing ──▶ Dependency Resolution ──▶ Pydantic Validation ──▶ Endpoint Logic
                                                         │ (rejects invalid input HERE,
                                                            before your code even runs)
                                                         ▼
                                          Response Serialization ──▶ Client
```

---

## 9. Practical Examples

**Simple:** build a single FastAPI endpoint accepting a Pydantic-validated request and returning a Pydantic-validated response, and inspect the automatically-generated `/docs` page.
**Medium:** add a dependency-injected "model" (Section 5) and an authentication dependency, testing that requests with invalid API keys are correctly rejected.
**Real-world:** wrap your Phase 7 RAG pipeline or actuarial agent (Lesson 3 of that phase) in a FastAPI application with proper async endpoints, a `/health` endpoint, and Pydantic-validated request/response schemas for each interaction type — a genuine, deployable API for an application you've already built.

---

## 10. Real Industry Use Cases

- **Every ML/LLM model-serving API you'll encounter in production**: FastAPI is the dominant choice for wrapping Python-based models (classical ML, Phase 4; deep learning, Phase 5; LLM pipelines, Phases 6-7) in a real HTTP API.
- **Hugging Face's own tooling and many open-source model-serving projects**: frequently built on or compatible with FastAPI as the serving layer.
- **RAG and agentic application backends** (Phase 7): the async-native design is specifically well-suited to the I/O-heavy nature of these pipelines (vector DB queries, LLM API calls, tool executions).
- **Internal microservices at companies of every size**: FastAPI's rapid development speed (automatic validation/docs) makes it a common default for internal API development generally, not just ML-specific use cases.

---

## 11. Common Mistakes

- Using blocking (synchronous) code inside an `async def` endpoint — freezes the entire event loop for all concurrent requests, exactly Phase 1 Lesson 2's warning about blocking calls inside coroutines, now at production-API stakes.
- Loading a large model (Phase 6) inside the endpoint function itself (reloading it on every single request) rather than once at application startup (Section 5's `lifespan` pattern) — catastrophically slow and wasteful.
- Not setting appropriate Pydantic field constraints (`ge`, `le`, custom validators, Phase 7 Lesson 6) — allowing invalid input (negative ages, out-of-range probabilities) to reach model inference code.
- Forgetting a `/health` endpoint — essential for container orchestration (Lesson 3's Kubernetes liveness/readiness probes) to know whether your service is actually functioning correctly.

---

## 12. Best Practices (2026)

- Load models/expensive resources once at application startup (via `lifespan`), never per-request.
- Use `async def` endpoints for any I/O-bound logic (LLM API calls, database queries, external HTTP requests), and ensure any library calls within them are genuinely async-compatible (use `httpx.AsyncClient`, not synchronous `requests`, for outbound HTTP calls).
- Define explicit Pydantic request/response models with appropriate field constraints for every endpoint — both for automatic validation and for the resulting automatic documentation's clarity.
- Include a `/health` (and often a separate `/ready`) endpoint from the start, anticipating Lesson 3's container-orchestration health-check requirements.

---

## 13. Exercises

**Easy:** Build a FastAPI endpoint accepting a Pydantic model with at least 2 field constraints, and verify invalid requests are correctly rejected with a 422 error.
**Medium:** Implement dependency injection for a mock "database connection" and write a test that overrides this dependency with an in-memory mock for testing purposes.
**Hard:** Build the full async I/O-bound throughput demonstration (Section 6) and empirically verify the concurrency speedup scales as expected across different numbers of simulated concurrent requests.
**Mathematical:** Given an expected request rate $\lambda$ and target average latency $W$, use Little's Law (Section 3) to estimate the expected number of in-flight concurrent requests $L$, and discuss how this informs server capacity planning.
**Coding:** Add structured logging (recording request latency, model version, and prediction result for every request) to the Section 5 API, in preparation for Lesson 8's monitoring/observability content.

---

## 14. Mini Project

Build a **complete, production-shaped FastAPI application wrapping your Phase 7 actuarial RAG/agent system**: implement proper Pydantic request/response schemas for every interaction type, dependency-injected model/resource loading via `lifespan`, authentication, a `/health` endpoint, structured request logging, and comprehensive error handling (returning clear, appropriate HTTP status codes for different failure modes) — a genuinely deployable API, ready for Lesson 2's containerization and Lesson 3's orchestration.

---

## 15. Interview Preparation

- Explain the difference between ASGI and WSGI, and why it matters for serving I/O-bound ML/LLM workloads.
- How does FastAPI's dependency injection system work, and what design pattern does it directly implement?
- Why is it important to load a model once at startup rather than per-request?
- Explain Little's Law and how it would inform capacity planning for an API server.

---

## 16. Summary

FastAPI provides the standard, production-grade way to wrap Python-based ML/LLM systems (Phases 4-7) in a real HTTP API — Pydantic-based automatic validation gives request/response reliability guarantees (directly extending Phase 7 Lesson 6's structured-output discipline), native async support delivers substantial throughput benefits for I/O-bound workloads (Phase 1 Lesson 2's concurrency concepts, now at production-API scale), and dependency injection provides a clean, testable way to manage models and other expensive resources. This lesson's FastAPI application is the direct, necessary foundation for Lesson 2's containerization and Lesson 3's orchestration — the next steps in taking a working application to genuine production deployment.

---

## 17. References

- Official FastAPI documentation (fastapi.tiangolo.com) — exceptionally clear and comprehensive
- Ramírez, S. — FastAPI's creator's various talks/writings on the framework's design philosophy
- ASGI specification (asgi.readthedocs.io)
- Uvicorn documentation (the standard ASGI server used to run FastAPI applications)
