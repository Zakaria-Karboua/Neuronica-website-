# Phase 1 · Lesson 2 — Advanced Python

> Prerequisite: Python Fundamentals (Lesson 1)

---

## 1. Introduction

### What is this topic?
"Advanced Python" here means the language features that separate script-writers from engineers who build libraries and frameworks: decorators, generators/coroutines, context managers, descriptors, metaclasses, the `functools`/`itertools` toolkits, closures, and the concurrency model (threading, multiprocessing, `asyncio`).

### Why does it exist?
Every major AI library you will use leans on these mechanisms. PyTorch's `nn.Module.__call__` uses `__call__` overloading; Hugging Face's `Trainer` uses decorators for hooks; FastAPI is built almost entirely on decorators, type hints, and `asyncio`; `contextlib` context managers appear everywhere for resource management (opening model checkpoints, database connections, GPU memory contexts).

### Historical background
Decorators arrived in Python 2.4 (2004) to solve the "wrapping a function to add behavior" problem elegantly. `asyncio` arrived in Python 3.4 (2014) and matured hugely through 3.5's `async`/`await` syntax (2015), driven by the industry's need for high-throughput I/O-bound servers — directly relevant to LLM API serving today.

### Real-world motivation
When you later read FastAPI's `@app.post(...)` or PyTorch's `@torch.no_grad()`, you should immediately know what's happening under the hood rather than treating it as magic.

---

## 2. Theory

- **Closures**: a function that "remembers" variables from its enclosing scope even after that scope has finished executing.
- **Decorators**: syntactic sugar (`@deco`) for `func = deco(func)` — a higher-order function that wraps another function/class to extend its behavior without modifying its source.
- **Generators**: functions using `yield` that produce a *lazy iterator* — pausing/resuming execution state between values, saving memory.
- **Coroutines (`async def`)**: functions that can suspend at `await` points, allowing an event loop to run other coroutines while waiting on I/O.
- **Context managers**: objects implementing `__enter__`/`__exit__` (or `@contextmanager`), guaranteeing setup/teardown code always runs, even on exceptions.
- **Descriptors**: objects implementing `__get__`/`__set__`, the mechanism behind `property`, methods themselves, and ORMs.
- **Metaclasses**: "classes of classes" — control how classes themselves are constructed (`type` is the default metaclass).

---

## 3. Mathematical Foundations

Concurrency correctness is a discrete-systems reasoning problem more than a numerical one, but two formalisms matter:

### Amdahl's Law (parallel speedup ceiling)
$$
S(n) = \frac{1}{(1-p) + \frac{p}{n}}
$$
where $p$ is the parallelizable fraction of the program and $n$ the number of processors. This explains why multiprocessing gives real speedups for CPU-bound work but threading does not (in standard CPython) for CPU-bound work due to the GIL — $p \to 0$ for GIL-bound sections.

### Little's Law (queueing, relevant to async servers)
$$
L = \lambda W
$$
Average number of in-flight requests $L$ equals arrival rate $\lambda$ times average time-in-system $W$. This underlies why async I/O (handling thousands of concurrent slow requests, e.g., waiting on an LLM API call) scales throughput without needing thousands of OS threads — it minimizes $W$ spent idly blocked.

### Amortized complexity of `functools.lru_cache`
With a cache of capacity $k$, worst-case per-call cost is $O(1)$ hash lookup; the *value* comes from converting an $O(2^n)$ naive-recursive computation (e.g., Fibonacci) into $O(n)$ via memoization — this is dynamic programming, and it's the exact same principle behind KV-caching in transformer inference, which you'll meet again in Phase 5–6.

---

## 4. Algorithm — Generator State Machine

```
CALL generator_function() 
   -> does NOT execute body; returns a generator object (frame suspended before first line)
FIRST next(gen)
   -> executes until first `yield`, returns value, freezes frame state (locals, instruction pointer)
SECOND next(gen)
   -> resumes exactly where frozen, continues to next `yield`
...
StopIteration raised when function returns / falls off the end
```
Complexity: generating $n$ items lazily is $O(1)$ additional memory (vs. $O(n)$ for a materialized list), at the cost of only supporting single forward iteration.

---

## 5. Python Implementation

```python
"""advanced_patterns.py — decorators, generators, async, context managers"""
import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


def retry(times: int = 3, delay: float = 0.5):
    """Decorator factory: retries a flaky function (e.g. a model API call)."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)          # preserves func.__name__, __doc__
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    time.sleep(delay * attempt)   # exponential-ish backoff
            raise last_exc
        return wrapper
    return decorator


@retry(times=3, delay=0.2)
def call_flaky_api() -> str:
    import random
    if random.random() < 0.7:
        raise ConnectionError("simulated network blip")
    return "success"


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    """Generator: yields fixed-size batches — core pattern for model inference."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f}s")


async def fetch_prediction(model_id: str, delay: float) -> dict:
    """Simulated async I/O-bound call (e.g. a remote inference endpoint)."""
    await asyncio.sleep(delay)
    return {"model": model_id, "result": "ok"}


async def fetch_all(model_ids: list[str]) -> list[dict]:
    tasks = [fetch_prediction(m, delay=0.3) for m in model_ids]
    return await asyncio.gather(*tasks)   # concurrent, not sequential


if __name__ == "__main__":
    with timer("batching demo"):
        for batch in batched(list(range(10)), 3):
            print(batch)

    results = asyncio.run(fetch_all(["gpt-x", "claude-y", "llama-z"]))
    print(results)
```

**Notes:** `asyncio.gather` runs all three simulated 0.3s calls *concurrently*, so total wall time ≈ 0.3s, not 0.9s — exactly the mechanism you rely on when calling multiple LLM/tool APIs in an agentic pipeline (Phase 7).

---

## 6. Build From Scratch

**A minimal decorator-based memoizer (mirrors `functools.lru_cache`):**
```python
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```
Turns the naive $O(2^n)$ recursion into $O(n)$ by caching each unique argument tuple's result — args must be hashable, which is *why* `functools.lru_cache` requires hashable arguments.

**A minimal `@contextmanager` implementation (mirrors what `contextlib` does):**
```python
class my_contextmanager:
    def __init__(self, gen_func):
        self.gen_func = gen_func
    def __call__(self, *args, **kwargs):
        return _GeneratorCM(self.gen_func(*args, **kwargs))

class _GeneratorCM:
    def __init__(self, gen):
        self.gen = gen
    def __enter__(self):
        return next(self.gen)
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            next(self.gen)
        except StopIteration:
            return False
```

---

## 7. Library Implementation (Comparison)

| From-scratch | Standard library | Notes |
|---|---|---|
| `memoize` | `functools.lru_cache(maxsize=...)` | stdlib adds bounded size + thread-safety + stats (`cache_info()`) |
| `my_contextmanager` | `contextlib.contextmanager` | stdlib handles exception re-raising, generator cleanup edge cases correctly |
| Hand-rolled thread pool | `concurrent.futures.ThreadPoolExecutor` | stdlib handles worker lifecycle, exception propagation, graceful shutdown |

---

## 8. Visual Explanations

**Async event loop (single thread, cooperative multitasking):**
```
Time ─────────────────────────────────────────────▶
Task A: [run]--(await I/O)---------------[resume, run]
Task B:        [run]--(await I/O)--[resume, run]
Task C:               [run]--(await I/O)--[resume, run]
                ▲ event loop switches tasks only at `await` points
```

**GIL vs multiprocessing:**
```
Threading (CPU-bound):        Multiprocessing (CPU-bound):
 CPU: [T1][T2][T1][T2]  (GIL    CPU1: [P1][P1][P1][P1]
       serializes CPU work)     CPU2: [P2][P2][P2][P2]  <- true parallel
```

---

## 9. Practical Examples

**Simple:** a `@log_calls` decorator that prints function name + args each call.
**Medium:** an infinite generator `primes()` yielding primes lazily using a sieve dict.
**Real-world:** an async batching client that collects incoming requests for 50ms then sends them as one batch to a model server — the exact pattern behind dynamic batching in production LLM inference servers (vLLM, TGI).

---

## 10. Real Industry Use Cases

- **FastAPI / Uvicorn** (used to serve nearly every internal model API at AI labs): built entirely on `async def` endpoints and an event loop.
- **vLLM / TGI**: use async request queues and continuous batching — conceptually generators/coroutines at massive scale.
- **PyTorch `DataLoader`**: uses multiprocessing workers to prefetch batches in parallel with GPU compute, hiding I/O latency behind Amdahl's-law-style overlap.
- **LangGraph/LangChain**: agent loops are essentially generator-driven state machines.

---

## 11. Common Mistakes

- Forgetting `functools.wraps` in a decorator — silently breaks introspection (`help()`, `pytest` fixture detection, FastAPI's dependency injection).
- Blocking calls (`time.sleep`, sync HTTP requests) inside `async def` — freezes the entire event loop, killing concurrency for *every* task.
- Using threads for CPU-bound numeric loops in pure Python, expecting speedup — GIL prevents it (use multiprocessing or vectorized NumPy instead).
- Mutable default arguments in decorated functions — the same trap as Lesson 1, easy to reintroduce inside wrapper closures.

---

## 12. Best Practices (2026)

- Prefer `asyncio` + `httpx.AsyncClient` (not `requests`) for concurrent API calls to LLM providers.
- Use `TaskGroup` (3.11+) over raw `asyncio.gather` for structured concurrency with better error propagation.
- For CPU-bound parallelism, evaluate Python 3.13's free-threaded build if your dependencies (NumPy/PyTorch) support it — a genuine 2026 shift away from always reaching for multiprocessing.
- Use `functools.cache` (simpler unbounded variant of `lru_cache`, 3.9+) for pure functions with small state spaces.

---

## 13. Exercises

**Easy:** Write a decorator `@deprecated` that prints a warning when a function is called.
**Medium:** Write a generator that reads a huge log file line-by-line and yields only lines matching a regex, without loading the whole file into memory.
**Hard:** Implement an async rate limiter (token bucket) that throttles concurrent calls to an external API to N requests/second.
**Mathematical:** Derive the expected wall-clock time for `asyncio.gather` over $n$ tasks each with i.i.d. latency $L_i$, assuming unlimited concurrency (answer: $\max_i L_i$, not $\sum L_i$).
**Coding:** Implement your own bounded LRU cache decorator (size-limited, evicts least-recently-used).

---

## 14. Mini Project

Build an **async batch-inference simulator**: a queue accepts "requests" (simulated with `asyncio.sleep`), a background coroutine collects them for up to 50ms or until 8 accumulate (whichever first), then "processes" them as one batch and returns results to each caller via `asyncio.Future`. This is a simplified but structurally accurate model of how real LLM inference servers batch requests.

---

## 15. Interview Preparation

- Explain what the GIL is and why `asyncio` still helps despite it.
- What happens if you call a blocking function inside an `async def` coroutine?
- Difference between a generator and a coroutine.
- Design question: how would you build a decorator that caches results of an expensive model call, but expires entries after 60 seconds (TTL cache)?

---

## 16. Summary

Advanced Python is fundamentally about controlling **when and how code executes**: decorators inject behavior around a call, generators defer computation across `yield` boundaries, coroutines defer across `await` boundaries under a single-threaded event loop, and context managers guarantee cleanup regardless of control flow. Every one of these mechanisms reappears, renamed but structurally identical, inside PyTorch, FastAPI, and every agentic framework in Phases 5–8.

---

## 17. References

- Ramalho, L. — *Fluent Python*, chapters on decorators, generators, concurrency
- David Beazley — "Generators: The Final Frontier" (PyCon talk)
- Official docs: `asyncio`, `contextlib`, `functools`, `itertools`
- PEP 492 (async/await), PEP 525 (async generators), PEP 703 (no-GIL)
