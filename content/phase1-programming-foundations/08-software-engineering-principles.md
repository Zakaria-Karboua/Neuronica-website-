# Phase 1 · Lesson 8 — Software Engineering Principles

> Prerequisite: Lessons 1–7

---

## 1. Introduction

### What is this topic?
The set of practices for writing code that remains correct, understandable, and changeable as it grows and as teams around it grow — as opposed to code that merely "works once." This covers: SOLID principles, DRY/KISS/YAGNI, clean code practices, code review discipline, technical debt management, and architectural thinking (modularity, separation of concerns).

### Why does it exist?
Research (and decades of painful industry experience) consistently shows that the majority of a codebase's total cost is incurred *after* initial writing — in reading, debugging, and modifying it. Software engineering principles exist to minimize that long-tail cost, not to make the first version look elegant.

### Historical background
Many of these ideas (structured programming, information hiding) trace to the 1970s software crisis (large projects failing due to unmanaged complexity — Dijkstra's "Go To Statement Considered Harmful," Parnas's work on modularity). SOLID was named/popularized by Robert C. Martin in the early 2000s, synthesizing earlier OOP design wisdom.

### Real-world motivation
ML/AI code has a well-earned reputation for being especially messy — notebooks full of global state, copy-pasted preprocessing, no tests, "it worked on my machine." Applying real software engineering discipline to ML code is precisely what separates a fragile research prototype from a system a team can build production infrastructure on top of (which is the entire point of Phases 6-9 of this curriculum).

---

## 2. Theory

### SOLID (object-oriented design principles)
| Principle | Meaning |
|---|---|
| **S**ingle Responsibility | a class/module should have one reason to change |
| **O**pen/Closed | open for extension, closed for modification (add behavior via new code, not editing existing tested code) |
| **L**iskov Substitution | subclasses must be usable anywhere their base class is expected, without breaking correctness |
| **I**nterface Segregation | prefer many small, specific interfaces over one large, general one |
| **D**ependency Inversion | depend on abstractions, not concrete implementations |

### Other core principles
- **DRY** (Don't Repeat Yourself): every piece of knowledge should have one authoritative representation.
- **KISS** (Keep It Simple): prefer the simplest solution that correctly solves the problem.
- **YAGNI** (You Aren't Gonna Need It): don't build speculative flexibility for requirements that don't yet exist.
- **Separation of Concerns**: distinct aspects of a program (data access, business logic, presentation) should live in distinct, loosely coupled modules.
- **Technical debt**: the implied future cost of choosing an easy-now solution over a better-but-slower one — a real, quantifiable tradeoff, not simply "bad code."

---

## 3. Mathematical Foundations

### Cyclomatic complexity (McCabe, 1976)
A quantitative measure of a function's structural complexity, computed from its control-flow graph:
$$
M = E - N + 2P
$$
where $E$ = edges, $N$ = nodes, $P$ = connected components (usually 1 per function) in the control-flow graph. Equivalently, $M = (\text{number of decision points}) + 1$. Empirically, $M > 10$ correlates with sharply increasing defect rates — a real, measurable justification for "break this function up," not just aesthetic preference.

### Coupling and cohesion (graph-theoretic view)
Model a codebase as a dependency graph $G = (V, E)$ where $V$ = modules and $E$ = "depends on" edges. **Coupling** relates to edge density between modules; **cohesion** relates to how tightly related the elements *within* a module are. Well-engineered systems minimize inter-module edges (low coupling) while maximizing intra-module relatedness (high cohesion) — directly analogous to graph partitioning / community-detection objectives you'll see again in Phase 4 (clustering).

### The economics of technical debt (a simplified model)
If $C_0$ is the cost of doing something properly now and $C_1 > C_0$ is the cost of doing it hastily now plus fixing it later, technical debt is rational exactly when:
$$
C_0 > P(\text{feature survives to need fixing}) \times C_1
$$
i.e., when the probability-weighted future cost is *less* than the upfront proper-engineering cost — which is why deliberately-incurred technical debt (e.g., in a throwaway research prototype) is a legitimate engineering decision, not always a mistake.

---

## 4. Algorithm — A Refactoring Decision Procedure

```
GIVEN a function/module flagged as "hard to change":
1. Measure: is cyclomatic complexity high (>10)? Are responsibilities mixed (SRP violation)?
2. Write characterization tests FIRST (tests capturing current behavior, even if imperfect)
   -> this gives you a safety net before touching anything
3. Extract smaller functions/classes along natural seams (one responsibility each)
4. Re-run tests after EACH small step -> confirm behavior unchanged
5. Only THEN consider larger structural changes (e.g., introducing an abstraction/interface)
STOP when: complexity is reduced AND all tests still pass AND the change is independently reviewable
```
This is Kent Beck's "make the change easy, then make the easy change" refactoring discipline — always test-protected, always incremental.

---

## 5. Python Implementation — SOLID Applied to an ML Pipeline

```python
"""solid_ml_pipeline.py — before/after applying SOLID to a common ML anti-pattern"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


# --- VIOLATES SRP + DIP: one class does loading, cleaning, training, AND saving ---
class MonolithicPipeline:
    def run(self, path: str) -> None:
        # loading
        import pandas as pd
        df = pd.read_csv(path)
        # cleaning (mixed in)
        df = df.dropna()
        # feature engineering (mixed in)
        df["age_band"] = df["age"] // 10
        # training (mixed in)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(df[["age_band"]], df["target"])
        # saving (mixed in) — and hard-coded path, hard to test
        import joblib
        joblib.dump(model, "model.pkl")


# --- SOLID-respecting version: each responsibility isolated behind an abstraction ---
class DataLoader(ABC):
    @abstractmethod
    def load(self, source: str): ...

class Cleaner(ABC):
    @abstractmethod
    def clean(self, df): ...

class FeatureEngineer(ABC):
    @abstractmethod
    def transform(self, df): ...

class ModelTrainer(ABC):
    @abstractmethod
    def fit(self, X, y): ...

class ModelSaver(ABC):
    @abstractmethod
    def save(self, model, destination: str) -> None: ...


@dataclass
class Pipeline:
    """Depends only on ABSTRACTIONS (Dependency Inversion), each with ONE job (SRP)."""
    loader: DataLoader
    cleaner: Cleaner
    engineer: FeatureEngineer
    trainer: ModelTrainer
    saver: ModelSaver

    def run(self, source: str, destination: str) -> None:
        df = self.loader.load(source)
        df = self.cleaner.clean(df)
        df = self.engineer.transform(df)
        model = self.trainer.fit(df.drop(columns=["target"]), df["target"])
        self.saver.save(model, destination)
```

**Why this matters concretely:** `MonolithicPipeline` cannot be unit-tested piece by piece, cannot swap CSV loading for a database loader without editing tested code (violates Open/Closed), and mixes five reasons-to-change into one class (violates SRP). The `Pipeline` version can be tested with mock implementations of each abstraction, and swapping `CSVLoader` for `SQLLoader` requires zero changes to `Pipeline` itself.

---

## 6. Build From Scratch

**A tiny cyclomatic complexity counter (to make Section 3's formula concrete):**
```python
import ast

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1   # base path
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1   # each `and`/`or` adds a branch
        self.generic_visit(node)

def cyclomatic_complexity(source: str) -> int:
    tree = ast.parse(source)
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity
```
Run this against `MonolithicPipeline.run` vs. `Pipeline.run` above — the monolithic version scores meaningfully higher, giving an objective number behind the subjective feeling of "this function does too much."

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `ComplexityVisitor` (AST walker) | `radon`, `flake8-cognitive-complexity`, SonarQube — full static analysis suites |
| Manual SOLID review | `ruff`/`pylint` static checks + architecture linters (e.g., `import-linter` enforcing layered dependency rules) |
| Manual refactoring | IDE-assisted refactoring (rename, extract method) with AST-aware safety |

---

## 8. Visual Explanations

```
Monolithic (high coupling, low cohesion):        Layered (SOLID-respecting):

  ┌─────────────────────┐                        Loader -> Cleaner -> Engineer -> Trainer -> Saver
  │ MonolithicPipeline    │                          │         │          │           │        │
  │  load+clean+feature+  │                       (each swappable independently,
  │  train+save, all mixed│                        each testable in isolation,
  └─────────────────────┘                        each with exactly ONE reason to change)
```

**Cyclomatic complexity as a control-flow graph:**
```
def f(x):
    if x > 0:        # decision point 1
        if x > 10:    # decision point 2
            return "big"
        return "small"
    return "neg"

Nodes: 4 (entry, if-x>0, if-x>10, exits)   Edges: 5   -> M = E - N + 2 = 5 - 4 + 2 = 3
(3 independent paths through the function — matches intuition: neg / small / big)
```

---

## 9. Practical Examples

**Simple:** refactor a function with 3 nested `if` statements into early returns (guard clauses), reducing nesting depth without changing logic.
**Medium:** extract a data-validation block duplicated across 3 functions into a single shared validator function (DRY).
**Real-world:** refactor a Jupyter-notebook-style "one giant cell" mortality-model script into a `DataLoader`/`FeatureEngineer`/`ModelTrainer` module structure exactly like Section 5, each independently unit-testable (feeding directly into Lesson 9, Testing).

---

## 10. Real Industry Use Cases

- **Google's engineering practices** (documented publicly): heavy emphasis on code review, small CLs (changelists), and readability standards enforced by dedicated reviewers — directly SOLID/DRY-driven culture at scale.
- **scikit-learn's API design**: `fit`/`transform`/`predict` is a textbook application of the Liskov Substitution Principle — any estimator can be swapped into a `Pipeline` because they all honor the same contract.
- **Hugging Face Transformers**: despite handling hundreds of model architectures, a consistent `PreTrainedModel`/`PreTrainedTokenizer` interface (Interface Segregation + Dependency Inversion) lets downstream code (fine-tuning scripts, `Trainer`) stay architecture-agnostic.
- **Netflix/Meta ML platforms**: heavy investment in shared, well-tested feature pipeline abstractions specifically to stop every team from re-implementing (and subtly breaking) the same data-cleaning logic — DRY at organizational scale.

---

## 11. Common Mistakes

- **Premature abstraction** (violating YAGNI): building a plugin system for "future model types" before you have even two real model types — adds complexity that pays off only if the speculative future actually arrives.
- Copy-pasting a preprocessing function across 5 notebooks instead of importing a shared module — the single most common source of train/serve skew bugs in real ML systems.
- Confusing "clever" with "clean" — a one-liner using three chained comprehensions that no one (including future-you) can parse in under a minute is a maintenance liability, not a flex.
- Refactoring without tests as a safety net — "cleaning up" code with no tests is how silent regressions get introduced.

---

## 12. Best Practices (2026)

- Treat notebooks as *exploration* tools only; promote anything reused more than once into a proper tested module (`src/`) — a near-universal 2026 ML engineering norm.
- Use static analysis (`ruff`, `mypy`) in CI to catch SOLID/DRY violations and type errors before human review time is spent on them.
- Adopt "small PRs, fast reviews" — large PRs correlate strongly with worse review quality and more escaped bugs (well-documented in engineering research).
- Use architecture decision records (ADRs) — short Markdown docs capturing *why* a significant design choice was made — increasingly standard for AI systems where model/infra choices need to be justified and revisited later.

---

## 13. Exercises

**Easy:** Take a function violating SRP (does validation + computation + logging) and split it into three functions.
**Medium:** Compute the cyclomatic complexity (by hand, then verify with the Section 6 tool) of a function with nested conditionals and a loop.
**Hard:** Refactor `MonolithicPipeline` (Section 5) into the SOLID version yourself from scratch, writing at least one concrete implementation of each abstract interface (e.g., `CSVLoader`, `DropNaCleaner`).
**Mathematical:** Given a module dependency graph with $n$ modules and $e$ edges, propose a numeric "coupling score" and explain what graph property (e.g., average degree) it should track.
**Coding:** Add a static-analysis check (using `ast`) that flags any function exceeding a cyclomatic complexity threshold, suitable for wiring into CI.

---

## 14. Mini Project

Take a deliberately messy, monolithic script (write one yourself: a 100+ line function mixing data loading, cleaning, feature engineering, model training, and saving with hard-coded paths) and refactor it fully into a SOLID-respecting package structure with clear module boundaries, each piece unit-testable in isolation — write the before/after cyclomatic complexity numbers into your README as evidence of the improvement.

---

## 15. Interview Preparation

- Explain each SOLID principle with a concrete example from a codebase you've worked on (or from this lesson's ML pipeline example).
- What is technical debt, and when is it a *rational* engineering choice rather than simply "bad code"?
- How would you approach refactoring a large, untested legacy function without breaking existing behavior?
- System design: how would you structure a codebase shared by 3 ML teams to minimize coupling between their independently-evolving model pipelines?

---

## 16. Summary

Software engineering principles exist to control the *cost curve* of a codebase over time, not to satisfy aesthetic preferences. SOLID, DRY, and cyclomatic-complexity-aware refactoring are concrete, partly-measurable tools for keeping ML code — which has a strong cultural tendency toward notebook sprawl and copy-paste — maintainable enough to actually support production systems. Every later phase of this curriculum (especially Phase 8, AI Engineering) assumes you apply these principles by default, not as an afterthought.

---

## 17. References

- Martin, Robert C. — *Clean Code*, *Clean Architecture*
- McCabe, T.J. — "A Complexity Measure" (1976, original cyclomatic complexity paper)
- Fowler, Martin — *Refactoring: Improving the Design of Existing Code*
- Google Engineering Practices documentation (publicly available code review guidelines)
