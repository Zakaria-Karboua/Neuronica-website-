# Phase 1 · Lesson 1 — Python Fundamentals

> Prerequisite for everything else in this curriculum. No prior programming assumed.

---

## 1. Introduction

### What is Python?
Python is a high-level, dynamically typed, interpreted, garbage-collected general-purpose programming language created by Guido van Rossum, first released in 1991. In 2026 it is the dominant language of AI/ML engineering: PyTorch, Hugging Face Transformers, LangChain, FastAPI, and essentially every major model-training and model-serving stack expose Python as their primary interface, even when the underlying compute kernels are written in C++/CUDA/Rust.

### Why does it exist?
Van Rossum designed Python as a reaction to ABC, a teaching language he found elegant but limited. He wanted a language that:
- read close to pseudocode (executable pseudocode),
- enforced readability through significant whitespace,
- had a small, orthogonal core with a "batteries included" standard library,
- could glue together components written in faster languages (C extensions).

### Historical background
- **1991** — Python 0.9.0 released.
- **2000** — Python 2.0, introduced list comprehensions, garbage collection.
- **2008** — Python 3.0, breaking backward compatibility to fix design mistakes (`print` as function, unified string/unicode model, true division).
- **2020** — Python 2 EOL. The ecosystem is 3.x-only.
- **2023–2026** — Performance-focused releases (3.11 "faster CPython" project, 3.12/3.13 continuing speedups, experimental free-threaded/no-GIL builds in 3.13+) driven largely by the needs of AI workloads that spend enormous compute inside Python-orchestrated pipelines.

### Real-world motivation
Every lesson after this one — NumPy, PyTorch, Transformers, FastAPI, LangGraph — is a Python library. If your fundamentals here are shaky, every subsequent lesson becomes harder than it needs to be. This lesson is the foundation the entire curriculum stands on.

---

## 2. Theory

### Definitions
- **Interpreter**: a program that executes source code directly (compiles to bytecode, then runs it on a virtual machine — the CPython VM).
- **Dynamic typing**: variable types are resolved at runtime, not compile time. A name (`x`) is just a reference/label; the *object* it points to carries the type.
- **Duck typing**: "if it walks like a duck and quacks like a duck, it's a duck" — Python cares about an object's behavior (does it support `.append()`?) rather than its declared type.
- **Everything is an object**: integers, functions, classes, modules — all are objects with an identity (`id()`), a type (`type()`), and a value.

### Intuition
Think of a Python variable as a sticky note (the name) that you attach to a box (the object) sitting in memory. `x = 5` doesn't put `5` "into" `x`; it creates the integer object `5` and sticks the note `x` onto it. `x = "hello"` peels the note off the integer and sticks it onto a new string object. The integer object doesn't disappear immediately — it disappears only when no notes point to it (reference counting + garbage collection).

### Important terminology
| Term | Meaning |
|---|---|
| Mutable | Object's internal state can change after creation (`list`, `dict`, `set`) |
| Immutable | Object's state cannot change after creation (`int`, `str`, `tuple`, `frozenset`) |
| Namespace | A mapping from names to objects (module namespace, function-local namespace, class namespace) |
| Scope | The region of code where a namespace is directly accessible (LEGB: Local, Enclosing, Global, Built-in) |
| Truthy/Falsy | Non-boolean values used in a boolean context (`0`, `""`, `[]`, `None` are falsy; most else is truthy) |

---

## 3. Mathematical Foundations

Python fundamentals are not math-heavy, but three areas matter for correctness and performance:

### Numerical representation
Python's `int` is arbitrary precision (bignum), implemented as an array of "digits" in base $2^{30}$ internally. This means `2**1000` never overflows — unlike C's fixed-width integers. `float` follows IEEE-754 double precision: 1 sign bit, 11 exponent bits, 52 mantissa bits, giving roughly 15–17 significant decimal digits.

$$
\text{float value} = (-1)^{s} \times 1.m \times 2^{e - 1023}
$$

This is why `0.1 + 0.2 != 0.3` in Python (and in almost every language using IEEE-754): $0.1$ and $0.2$ have no exact finite binary representation, so their sum accumulates rounding error at the bit level.

### Complexity notation
We use Big-O notation to describe how an algorithm's running time or memory grows with input size $n$:

$$
f(n) = O(g(n)) \iff \exists\, c > 0, n_0 \text{ such that } f(n) \le c \cdot g(n) \; \forall n \ge n_0
$$

For fundamentals, the two you must internalize immediately:
- `list.append(x)` → amortized $O(1)$ (Python over-allocates capacity, doubling geometrically, so the total cost of $n$ appends is $O(n)$, giving $O(1)$ per operation on average).
- `x in list` → $O(n)$ (linear scan) vs `x in set`/`x in dict` → average $O(1)$ (hash table lookup).

### Hashing
`dict` and `set` are hash tables. A hash function $h: \text{Key} \to \mathbb{Z}$ maps a key to a bucket index via `h(key) mod table_size`. Only immutable (hashable) objects can be dict keys/set members, because if a key mutated after insertion its hash would change and it would become unfindable in its original bucket — this is *why* lists can't be dict keys but tuples can.

---

## 4. Algorithm — "How Python Executes Your Code"

**Pseudocode of the execution pipeline:**
```
1. Source code (.py) is read
2. Tokenizer splits it into tokens (keywords, identifiers, literals, operators)
3. Parser builds an Abstract Syntax Tree (AST)
4. Compiler converts AST into bytecode (.pyc, stored in __pycache__)
5. CPython Virtual Machine executes bytecode instruction by instruction
   - maintains a stack per frame
   - dispatches each bytecode op (LOAD_FAST, BINARY_ADD, CALL_FUNCTION, ...) 
6. Reference counts are updated as objects are created/destroyed
7. Garbage collector periodically scans for reference cycles
```

You can inspect bytecode yourself:
```python
import dis
def add(a, b):
    return a + b
dis.dis(add)
```

**Complexity of variable lookup (LEGB):** Local lookup is $O(1)$ (frame's local array, indexed at compile time). Global/builtin lookups are dict lookups, $O(1)$ average but slower in constant factor than local — this is why hot loops that reference global names benefit from being copied into locals first.

---

## 5. Python Implementation

```python
"""
production_example.py
A small, clean module demonstrating idiomatic modern (2026) Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True, frozen=True)
class Patient:
    """Immutable, memory-efficient record (slots=True avoids per-instance __dict__)."""
    patient_id: str
    age: int
    systolic_bp: float

    def risk_flag(self) -> bool:
        return self.age > 65 and self.systolic_bp > 140


def high_risk_patients(patients: list[Patient]) -> Iterator[Patient]:
    """Generator: lazily yields high-risk patients without building an
    intermediate list — O(1) extra memory instead of O(n)."""
    for p in patients:
        if p.risk_flag():
            yield p


if __name__ == "__main__":
    cohort = [
        Patient("P001", 72, 148.0),
        Patient("P002", 45, 118.0),
        Patient("P003", 80, 155.0),
    ]
    for patient in high_risk_patients(cohort):
        print(f"{patient.patient_id} flagged: age={patient.age}, "
              f"SBP={patient.systolic_bp}")
```

**Line-by-line notes:**
- `from __future__ import annotations` — postpones evaluation of type hints, letting you use `list[Patient]` and forward references cleanly on any 3.x version.
- `@dataclass(slots=True, frozen=True)` — auto-generates `__init__`, `__repr__`, `__eq__`; `slots=True` removes the per-instance `__dict__` (real memory savings at scale — important when you're holding millions of records, e.g. an actuarial mortality table); `frozen=True` makes instances immutable and hashable.
- Generator function (`yield`) — critical pattern for large datasets: never materialize what you don't need to.

---

## 6. Build From Scratch — Reimplementing Core Behaviors

To truly understand Python, implement small pieces of "built-in" behavior yourself.

**A minimal hash table (to understand what `dict` does under the hood):**
```python
class SimpleHashMap:
    def __init__(self, capacity: int = 8):
        self._capacity = capacity
        self._buckets: list[list[tuple]] = [[] for _ in range(capacity)]
        self._size = 0

    def _index(self, key) -> int:
        return hash(key) % self._capacity

    def put(self, key, value) -> None:
        bucket = self._buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > 0.75:   # load-factor resize
            self._resize()

    def get(self, key):
        bucket = self._buckets[self._index(key)]
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def _resize(self) -> None:
        old_items = [kv for bucket in self._buckets for kv in bucket]
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        for k, v in old_items:
            self.put(k, v)
```
This mirrors real `dict` mechanics: hashing, bucket chaining for collisions, and resizing when the load factor crosses a threshold — average $O(1)$ get/put, worst case $O(n)$ under pathological hash collisions.

---

## 7. Library Implementation (Comparison)

| From-scratch `SimpleHashMap` | Built-in `dict` |
|---|---|
| Pure Python loops, ~30 lines | Implemented in C, open-addressing (not chaining) since CPython 3.6+ |
| $O(1)$ average, high constant factor | $O(1)$ average, far lower constant factor (compiled C) |
| No ordering guarantee logic | Insertion-order preserving since 3.7 (implementation detail → language guarantee) |
| Educational only | Production-grade, use this always |

**Lesson:** understand the mechanism by building it once; use the standard library forever after, because CPython's C implementation will always outperform hand-rolled Python by 10–100x for this kind of primitive.

---

## 8. Visual Explanations

**Reference counting + garbage collection (ASCII):**
```
x = [1, 2, 3]          x ──▶ [list object]  refcount = 1
y = x                  x,y ──▶ [list object]  refcount = 2
del x                       y ──▶ [list object]  refcount = 1
del y                       (refcount 0) ──▶ memory freed immediately
```

**LEGB scope resolution order:**
```
 ┌─────────────────────────────┐
 │  Built-in   (len, print...) │
 │  ┌─────────────────────────┐│
 │  │  Global (module level)  ││
 │  │ ┌───────────────────────┤│
 │  │ │ Enclosing (closures)  ││
 │  │ │ ┌─────────────────────┤│
 │  │ │ │  Local (function)   ││
 │  │ │ │                     ││
 │  │ │ └─────────────────────┘│
 │  │ └───────────────────────┘│
 │  └─────────────────────────┘│
 └─────────────────────────────┘
   lookup order: Local → Enclosing → Global → Built-in
```

---

## 9. Practical Examples

**Simple:**
```python
temperatures_c = [22.5, 19.0, 30.1]
temperatures_f = [c * 9 / 5 + 32 for c in temperatures_c]
```

**Medium — word frequency counter:**
```python
from collections import Counter

text = "the cat sat on the mat the cat ran"
counts = Counter(text.split())
print(counts.most_common(2))   # [('the', 3), ('cat', 2)]
```

**Real-world — parsing a CSV of insurance claims without pandas (pure stdlib):**
```python
import csv
from statistics import mean

with open("claims.csv", newline="") as f:
    reader = csv.DictReader(f)
    claim_amounts = [float(row["amount"]) for row in reader]

print(f"Average claim: {mean(claim_amounts):.2f}")
```

---

## 10. Real Industry Use Cases

- **OpenAI / Anthropic**: model training orchestration, data pipelines, and eval harnesses are Python-first; performance-critical kernels drop to C++/CUDA/Triton but are called *from* Python.
- **Netflix**: Python powers much of its internal tooling and offline ML pipelines (feature computation, A/B test analysis).
- **Google DeepMind**: JAX and Haiku — Python APIs over XLA-compiled numerical kernels.
- **Meta**: PyTorch itself — the Python layer is the ergonomic front-end over a C++/CUDA core (`libtorch`).
- **NVIDIA**: cuDF, cuML, RAPIDS — Python APIs that mimic pandas/scikit-learn while running on GPU.

The consistent industry pattern: **Python is the control plane; compiled languages are the data plane.** As an AI engineer you live mostly in the control plane and must know exactly where the boundary is.

---

## 11. Common Mistakes

**Beginner:**
```python
# Mutable default argument — classic trap
def add_item(item, bucket=[]):   # BUG: bucket is created ONCE at def time
    bucket.append(item)
    return bucket

add_item(1)   # [1]
add_item(2)   # [1, 2]  <- surprising! Same list reused across calls.
```
Fix: `def add_item(item, bucket=None): bucket = bucket or []`

**Production:**
- Catching bare `except:` and swallowing errors silently — hides real bugs in production pipelines.
- Comparing floats with `==` instead of `math.isclose()` — causes flaky tests in numerical code (very relevant to actuarial/statistical work).
- Using `import *` in library code — pollutes namespaces and breaks static analysis tools.

**Debugging techniques:** `pdb`/`breakpoint()`, `logging` instead of `print` in anything beyond a notebook, `assert` for invariants during development (stripped in `-O` mode, so never for security checks).

---

## 12. Best Practices (2026)

- Target **Python 3.12/3.13** — free-threaded (no-GIL) builds are now usable for CPU-bound parallel workloads, a major shift from the historical GIL bottleneck.
- Use `uv` or `pip-tools`/`poetry` for reproducible dependency management (raw `pip install` without a lockfile is now considered a rookie mistake in 2026 pipelines).
- Type hints + `mypy`/`pyright` are standard even in research code, not just production — LLM-assisted coding tools rely heavily on type signatures for correctness.
- Prefer `pathlib.Path` over string path manipulation.
- **Deprecated**: Python 2 syntax (obviously), `%`-string formatting in new code (use f-strings), `distutils` (removed from stdlib in 3.12 — use `packaging`/`setuptools`).

---

## 13. Exercises

**Easy**
1. Write a function `is_palindrome(s: str) -> bool` ignoring case and spaces.
2. Explain, without running it, what `[1,2,3] * 2` evaluates to and why.

**Medium**
3. Implement a generator `sliding_window(iterable, size)` that yields tuples of consecutive elements.
4. Given a list of dicts representing transactions, write pure-Python code (no pandas) to compute total amount per category.

**Hard**
5. Implement your own `LRUCache` class (used constantly in caching layers for inference servers) using only `dict` + a doubly linked list, achieving $O(1)$ get/put.

**Mathematical**
6. Prove that Python's amortized `list.append` is $O(1)$ given a doubling growth strategy (geometric series argument).

**Coding**
7. Write a decorator `@timeit` that prints the execution time of any function it wraps.

---

## 14. Mini Project

**Build a CLI "Claims Triage Tool":** reads a CSV of insurance claims (id, amount, date, flagged_reason), uses generators to stream-process large files without loading everything into memory, classifies claims into risk buckets using pure Python logic (no libraries yet — those come in later lessons), and prints a summary report using only `argparse`, `csv`, `dataclasses`, and `collections`.

---

## 15. Interview Preparation

**Theory**
- Explain the difference between `is` and `==`.
- What is the GIL and how does it affect CPU-bound vs I/O-bound code?
- Why are tuples hashable but lists aren't?

**Coding**
- Reverse a linked list implemented from scratch.
- Flatten an arbitrarily nested list without using recursion (use an explicit stack).

**System design (early flavor)**
- How would you process a 50 GB CSV file that doesn't fit in memory using pure Python?

---

## 16. Summary

Python's power for AI engineering comes from: dynamic typing + duck typing for fast iteration, a hashing-based `dict`/`set` core that underlies almost every fast lookup structure you'll use, generators for memory-efficient streaming, and a "thin Python, thick C" architecture that every major ML library (NumPy, PyTorch, pandas) follows. Master reference semantics, mutability, and scope now — every subtle bug in later, more complex ML code traces back to one of these three fundamentals.

---

## 17. References

- Ramalho, L. — *Fluent Python* (2nd ed., O'Reilly)
- Official CPython docs: https://docs.python.org/3/
- Real Python — practical tutorials: https://realpython.com/
- PEP 8 (style guide), PEP 484 (type hints), PEP 703 (making the GIL optional)
- "CPython Internals" — Anthony Shaw
