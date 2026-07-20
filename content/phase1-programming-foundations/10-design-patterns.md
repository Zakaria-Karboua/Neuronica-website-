# Phase 1 · Lesson 10 — Design Patterns

> Prerequisite: Lessons 1–9 (this lesson closes out Phase 1 by synthesizing OOP, SOLID, and testing into reusable design vocabulary)

---

## 1. Introduction

### What is this topic?
Design patterns are named, reusable solutions to recurring software design problems — a shared vocabulary that lets engineers say "just use a Strategy pattern here" instead of re-explaining the same structural idea from scratch every time. They are not code to copy-paste; they are *shapes* of solutions, adapted to context.

### Why does it exist?
The "Gang of Four" (Gamma, Helm, Johnson, Vlissides) catalogued 23 patterns in their 1994 book *Design Patterns: Elements of Reusable Object-Oriented Software*, observing that expert object-oriented designers kept independently reinventing the same handful of structures. Naming them turned tacit expert intuition into explicit, teachable knowledge.

### Historical background
The patterns movement drew on architect Christopher Alexander's concept of "pattern languages" in building architecture (1970s), imported into software by the GoF in the early 1990s. Some GoF patterns have since become language features themselves (e.g., Python's iterators/generators built the Iterator pattern into the language; decorators built the Decorator pattern into syntax) — a sign of how foundational they are.

### Real-world motivation
Every major ML framework you'll use in this curriculum leans on these patterns explicitly: PyTorch's `nn.Module` composition is Composite; `torch.optim` optimizers are Strategy; Hugging Face's `AutoModel.from_pretrained()` is Factory; callback systems in training loops (`on_epoch_end`, etc.) are Observer. Recognizing the pattern means the library's API design stops feeling arbitrary.

---

## 2. Theory — The Three GoF Categories

| Category | Purpose | Key patterns covered here |
|---|---|---|
| **Creational** | control *how* objects are created | Factory Method, Builder, Singleton |
| **Structural** | control how objects/classes are *composed* | Composite, Adapter, Decorator |
| **Behavioral** | control how objects *communicate/interact* | Strategy, Observer, Template Method |

### Definitions
- **Factory Method**: defer object instantiation to a subclass/function so calling code doesn't need to know the concrete class.
- **Builder**: separate the construction of a complex object from its representation, building it step by step.
- **Singleton**: ensure a class has exactly one instance, globally accessible (use sparingly — often overused; frequently better replaced by dependency injection).
- **Composite**: compose objects into tree structures, letting clients treat individual objects and compositions uniformly.
- **Adapter**: convert one interface into another that client code expects, without modifying either side.
- **Decorator** (structural pattern, distinct from Python's `@decorator` syntax though closely related): attach additional responsibilities to an object dynamically, as an alternative to subclassing.
- **Strategy**: define a family of interchangeable algorithms, encapsulate each, and select one at runtime.
- **Observer**: define a one-to-many dependency so that when one object changes state, all its dependents are notified automatically.
- **Template Method**: define the skeleton of an algorithm in a base class, letting subclasses override specific steps without changing the overall structure.

---

## 3. Mathematical Foundations

Design patterns are primarily a structural/architectural topic rather than a numerically derivable one, but two formal connections matter:

### Patterns as category-theoretic morphisms (conceptual)
Several patterns can be viewed through composition: the **Strategy** pattern is literally a first-class function (a morphism $f: \text{Input} \to \text{Output}$) selected at runtime — the OOP encoding of what Lesson 2's higher-order functions do more directly. In languages/paradigms with first-class functions (Python included), Strategy is often *just* passing a function/callable rather than instantiating a class hierarchy — a useful simplification insight, not a criticism.

### Composite pattern and recursive structures
The Composite pattern formalizes a recursive tree: if $T$ is a `Component`, then a `Composite` is defined as
$$
T ::= \text{Leaf} \;\mid\; \text{Composite}(T_1, T_2, \dots, T_k)
$$
This is exactly the recursive grammar behind a parse tree/AST (Lesson 8's `ast` module) or a neural network's module tree (`nn.Sequential` containing other `nn.Module`s) — operations like `forward()` or `total_parameters()` are computed via recursive tree traversal, $O(n)$ in the number of nodes.

### Observer pattern and event complexity
With $n$ observers registered to one subject, notifying all observers on a state change is $O(n)$ per event; if observers themselves trigger further events (cascading), total complexity can blow up — a real design risk (uncontrolled observer cascades) that must be reasoned about explicitly in event-driven ML pipelines (e.g., training callbacks that themselves log to systems that trigger further callbacks).

---

## 4. Algorithm — Choosing a Pattern (a decision procedure)

```
IS the problem "I need to create objects without hardcoding the concrete class"?
   -> Factory Method (simple) or Builder (complex, multi-step construction)
IS the problem "I need a tree of part-whole objects treated uniformly"?
   -> Composite
IS the problem "I have an existing interface that doesn't match what client code expects"?
   -> Adapter
IS the problem "I need to swap an algorithm/behavior at runtime, without an if/elif chain"?
   -> Strategy
IS the problem "many parts of the system need to react to one object's state changes, without tight coupling"?
   -> Observer
IS the problem "several variants of an algorithm share the same overall steps but differ in specific sub-steps"?
   -> Template Method
```
Anti-pattern warning baked into the procedure: if none of these questions clearly match, you probably don't need a named pattern — reach for the simplest direct solution (YAGNI, Lesson 8) rather than forcing a pattern where a plain function would do.

---

## 5. Python Implementation

```python
"""ml_design_patterns.py — patterns as they actually appear in ML codebases"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable


# --- STRATEGY: interchangeable preprocessing algorithms, selected at runtime ---
class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, values: list[float]) -> list[float]: ...

class MinMaxScaling(ScalingStrategy):
    def scale(self, values):
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) for v in values]

class ZScoreScaling(ScalingStrategy):
    def scale(self, values):
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return [(v - mean) / std for v in values]

class FeaturePipeline:
    def __init__(self, strategy: ScalingStrategy):
        self.strategy = strategy   # swap strategies without touching this class (Open/Closed)
    def run(self, values: list[float]) -> list[float]:
        return self.strategy.scale(values)


# --- FACTORY METHOD: instantiate models by name without callers knowing concrete classes ---
class ModelFactory:
    _registry: dict[str, Callable[[], "BaseModel"]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def create(cls, name: str) -> "BaseModel":
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name]()

class BaseModel(ABC):
    @abstractmethod
    def predict(self, x): ...

@ModelFactory.register("linear")
class LinearModel(BaseModel):
    def predict(self, x): return 2 * x + 1

@ModelFactory.register("constant")
class ConstantModel(BaseModel):
    def predict(self, x): return 42


# --- OBSERVER: training callbacks reacting to epoch-end events ---
class TrainingSubject:
    def __init__(self):
        self._observers: list[Callable[[int, float], None]] = []
    def subscribe(self, observer: Callable[[int, float], None]) -> None:
        self._observers.append(observer)
    def notify_epoch_end(self, epoch: int, loss: float) -> None:
        for obs in self._observers:
            obs(epoch, loss)

def log_to_console(epoch: int, loss: float) -> None:
    print(f"[epoch {epoch}] loss={loss:.4f}")

def early_stopping_check(best_loss: list[float]):
    def observer(epoch: int, loss: float) -> None:
        if loss < best_loss[0]:
            best_loss[0] = loss
    return observer


if __name__ == "__main__":
    pipeline = FeaturePipeline(ZScoreScaling())
    print(pipeline.run([10, 20, 30, 40]))

    model = ModelFactory.create("linear")
    print(model.predict(5))

    subject = TrainingSubject()
    best = [float("inf")]
    subject.subscribe(log_to_console)
    subject.subscribe(early_stopping_check(best))
    subject.notify_epoch_end(1, 0.85)
    subject.notify_epoch_end(2, 0.62)
```

---

## 6. Build From Scratch

**Composite pattern for a model-architecture tree (mirrors `nn.Module` nesting):**
```python
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def param_count(self) -> int: ...

class Dense(Layer):
    def __init__(self, in_f: int, out_f: int):
        self.in_f, self.out_f = in_f, out_f
    def param_count(self) -> int:
        return self.in_f * self.out_f + self.out_f   # weights + bias

class Sequential(Layer):
    """Composite: a container of Layers that is ITSELF a Layer."""
    def __init__(self, *layers: Layer):
        self.layers = layers
    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)   # recursive tree sum

model = Sequential(Dense(784, 256), Dense(256, 64), Dense(64, 10))
print(model.param_count())   # recursively sums across the whole "network"
```
This is a deliberately simplified but structurally faithful mirror of how `torch.nn.Sequential` (containing arbitrary nested `nn.Module`s) computes things like total parameter count — recursive tree traversal over a Composite structure.

---

## 7. Library Implementation (Comparison)

| From-scratch pattern | Real ML library usage |
|---|---|
| `ScalingStrategy` (Strategy) | `sklearn.preprocessing` scalers, all sharing `fit`/`transform`; `torch.optim` optimizers (SGD, Adam) selected interchangeably |
| `ModelFactory` (Factory Method) | `transformers.AutoModel.from_pretrained(name)` — returns the correct concrete architecture class from a string |
| `TrainingSubject` (Observer) | PyTorch Lightning / Keras callback systems (`on_epoch_end`, `on_batch_end` hooks) |
| `Sequential` (Composite) | `torch.nn.Sequential`, `torch.nn.ModuleList` — real nested module trees |

---

## 8. Visual Explanations

**Strategy pattern (swap algorithm without changing the caller):**
```
FeaturePipeline ---uses---> ScalingStrategy (interface)
                                 ▲        ▲
                          MinMaxScaling  ZScoreScaling
   (FeaturePipeline code never changes when you add a new strategy)
```

**Composite pattern (uniform treatment of leaf vs. container):**
```
                Sequential (composite)
               /       |        \
           Dense     Dense     Sequential (nested!)
                                 /      \
                             Dense     Dense
   param_count() called on the root recurses down through the WHOLE tree uniformly
```

**Observer pattern (one-to-many notification):**
```
TrainingSubject.notify_epoch_end()
        │
        ├──▶ log_to_console(epoch, loss)
        ├──▶ early_stopping_check(...)
        └──▶ [any other subscribed observer]
```

---

## 9. Practical Examples

**Simple:** implement a `PaymentStrategy` interface with `CreditCard`/`PayPal` implementations selected at checkout time.
**Medium:** implement a `Builder` for constructing a complex `ModelConfig` object (many optional hyperparameters) step by step instead of a constructor with 15 keyword arguments.
**Real-world:** a `FeatureEngineeringStrategy` interface letting you swap between different actuarial risk-scoring formulas (e.g., different regulatory jurisdictions) without touching the surrounding pipeline code — directly applicable to multi-market actuarial pricing work.

---

## 10. Real Industry Use Cases

- **PyTorch**: `nn.Module` composition = Composite; `torch.optim` = Strategy; hooks (`register_forward_hook`) = Observer.
- **Hugging Face Transformers**: `AutoModel`/`AutoTokenizer` = Factory Method at massive scale (hundreds of registered architectures behind one factory interface).
- **scikit-learn**: `Pipeline` = Composite (each step, including nested pipelines, honors the same `fit`/`transform` interface); `GridSearchCV` internally uses Strategy to swap estimators.
- **Keras/PyTorch Lightning**: callback systems = Observer, letting logging, checkpointing, and early stopping all react to the same training loop events without the training loop knowing about any of them specifically (Dependency Inversion, tying back to Lesson 8).

---

## 11. Common Mistakes

- **Pattern-itis**: forcing a GoF pattern where a plain function or dict lookup would be simpler and clearer — patterns solve *recurring structural problems*, not "make code look more sophisticated."
- Overusing Singleton for global state — creates hidden coupling and makes testing hard (can't easily substitute a test double); dependency injection is usually preferable.
- Implementing Strategy with a full class hierarchy in Python when a first-class function would do the same job with less ceremony (Python's functions being objects makes many GoF patterns lighter-weight than in Java/C++).
- Deep, unnecessary Composite nesting "for future flexibility" that violates YAGNI (Lesson 8) when the tree will realistically never be more than 2 levels deep.

---

## 12. Best Practices (2026)

- In Python specifically, prefer functions/closures over full class-based Strategy implementations when the "family of algorithms" has no additional state to encapsulate — idiomatic modern Python leans lighter-weight than classic GoF Java-style code.
- Use `Protocol` (structural typing) rather than `ABC` when you want Strategy-style interchangeability without forcing inheritance — increasingly common in typed 2026 Python.
- Reserve Singleton for genuinely process-wide, stateless resources (e.g., a config loader) and prefer explicit dependency passing everywhere else.
- Recognize patterns in libraries you use daily (this lesson's whole point) — it turns "why does this API look like this" into instantly transferable understanding across new libraries.

---

## 13. Exercises

**Easy:** Implement a `NotificationStrategy` interface (Email/SMS) selected at runtime.
**Medium:** Implement a `Builder` for a `TrainingConfig` object with a fluent interface (`.with_lr(0.01).with_epochs(50).build()`).
**Hard:** Implement a Composite-based expression tree evaluator (numbers and binary operators as tree nodes), computing the result via recursive traversal — a direct precursor to how computation graphs work in Phase 5 (autograd).
**Mathematical:** Given an Observer subject with $n$ observers, each of which may itself trigger $m$ further downstream notifications, derive the worst-case total notification count for one initial event (geometric growth risk) and explain why real systems cap or avoid such cascades.
**Coding:** Refactor the `MonolithicPipeline` mistake from Lesson 8 using the Template Method pattern instead of full dependency injection — define the pipeline skeleton in a base class, letting subclasses override just the cleaning/feature-engineering steps.

---

## 14. Mini Project

Build a **pluggable feature-engineering framework**: a `Strategy`-based system supporting multiple scaling/encoding strategies selectable by config string via a `Factory`, a `Composite`-based pipeline allowing pipelines to nest other pipelines, and an `Observer`-based logging system that reports pipeline progress/statistics without the pipeline code needing to know anything about logging — write tests (Lesson 9) confirming each strategy is swappable without modifying the pipeline's own code (a direct, testable demonstration of the Open/Closed Principle from Lesson 8).

---

## 15. Interview Preparation

- Explain the Strategy pattern and give a real example from a Python ML library you've used.
- Why is Singleton often considered an anti-pattern in modern software design, and what would you use instead?
- Explain how the Composite pattern relates to recursive tree structures like ASTs or neural network module trees.
- Design question: how would you design a config-driven system letting a data pipeline swap its cleaning/feature-engineering strategy per client/market without duplicating pipeline code?

---

## 16. Summary

Design patterns are Phase 1's capstone: they synthesize OOP (Lesson 3), SOLID (Lesson 8), and testable modular design (Lesson 9) into a shared vocabulary for recurring structural problems. Recognizing Strategy, Factory, Composite, and Observer in PyTorch, scikit-learn, and Hugging Face turns their APIs from memorized incantations into predictable, learnable structure — the mental model that will make every subsequent phase of this curriculum (especially Phase 5 onward, where you'll read and extend real framework source code) dramatically easier to internalize.

---

## 17. References

- Gamma, Helm, Johnson, Vlissides — *Design Patterns: Elements of Reusable Object-Oriented Software* (Gang of Four, 1994)
- Refactoring.Guru — modern, example-rich pattern reference (refactoring.guru/design-patterns)
- Alex Martelli — "Python Design Patterns" talks (on idiomatic, lightweight Pythonic pattern implementations)
- PyTorch / Hugging Face / scikit-learn source code — the best "real examples" available, now readable with this lesson's vocabulary
