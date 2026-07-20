# Phase 1 · Lesson 3 — Object-Oriented Programming

> Prerequisite: Lessons 1–2

---

## 1. Introduction

### What is OOP?
A programming paradigm organizing code around **objects** — bundles of state (attributes) and behavior (methods) — rather than around functions acting on passive data. Python is multi-paradigm, but nearly every serious library (PyTorch's `nn.Module`, scikit-learn's `Estimator`, Hugging Face's `PreTrainedModel`) is built as a class hierarchy.

### Why does it exist?
OOP emerged (Simula 1967, Smalltalk 1970s) to manage growing software complexity by encapsulating state so that only well-defined interfaces touch it, and by allowing new behavior to be added via inheritance/composition instead of rewriting existing code.

### Historical background
Python's OOP model deliberately keeps encapsulation *convention-based* (no true `private` keyword — just `_leading_underscore` and `__name_mangling`), reflecting the language's "we're all consenting adults" philosophy versus Java/C++'s enforced access modifiers.

### Real-world motivation
Every neural network you build in PyTorch is a Python class inheriting from `nn.Module`. If you don't understand `__init__`, inheritance, and `super()`, you cannot read or write a single PyTorch model definition correctly.

---

## 2. Theory

- **Class**: a blueprint defining attributes/methods; **instance**: a concrete object built from that blueprint.
- **Encapsulation**: bundling data with the methods that operate on it, hiding internal representation.
- **Inheritance**: a class (child/subclass) derives attributes/methods from another (parent/superclass), modeling an "is-a" relationship.
- **Polymorphism**: different classes can be used interchangeably if they implement a common interface (duck typing in Python — no explicit interface keyword needed).
- **Composition**: building complex objects by combining simpler ones as attributes ("has-a" relationship) — often preferred over inheritance for flexibility.
- **MRO (Method Resolution Order)**: the algorithm (C3 linearization) Python uses to decide which class's method runs when multiple inheritance is involved.

---

## 3. Mathematical Foundations

### C3 Linearization (Method Resolution Order)
For multiple inheritance `class D(B, C)` where `B(A)`, `C(A)`, Python computes MRO via:
$$
L[D] = D + \text{merge}(L[B], L[C], [B, C])
$$
where `merge` takes the head of the first list that does not appear in the tail of any other list, repeating until all lists are consumed. This guarantees a *monotonic, consistent* linearization — avoiding the classic "diamond problem" ambiguity of C++.

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass
print(D.__mro__)
# (D, B, C, A, object)  — computed by C3, not naive depth-first
```

### Complexity of attribute lookup
Instance attribute lookup: check instance `__dict__` → $O(1)$ hash lookup; if absent, walk the MRO checking each class's `__dict__` → $O(k)$ where $k$ = MRO depth (usually small, effectively constant in practice).

---

## 4. Algorithm — Object Construction Sequence

```
CALL ClassName(args)
 1. type.__call__(ClassName, args) invoked
 2. ClassName.__new__(ClassName, args) → allocates raw instance (rarely overridden)
 3. instance.__init__(args) → initializes attributes on the allocated instance
 4. constructed instance returned to caller
```
`__new__` vs `__init__`: `__new__` creates the object (needed for immutable types like subclassing `tuple`/`str`, or singleton patterns); `__init__` configures an already-created object. 99% of the time you only override `__init__`.

---

## 5. Python Implementation

```python
"""oop_patterns.py — inheritance, composition, dunder methods, ABCs"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Model(ABC):
    """Abstract base class — cannot be instantiated directly."""

    @abstractmethod
    def predict(self, x: float) -> float:
        ...

    def __call__(self, x: float) -> float:
        # Mirrors PyTorch's nn.Module.__call__ -> forward() pattern
        return self.predict(x)


class LinearModel(Model):
    def __init__(self, weight: float, bias: float):
        self.weight = weight
        self.bias = bias

    def predict(self, x: float) -> float:
        return self.weight * x + self.bias

    def __repr__(self) -> str:
        return f"LinearModel(weight={self.weight}, bias={self.bias})"

    def __add__(self, other: "LinearModel") -> "LinearModel":
        # Operator overloading: combine two linear models (ensembling)
        return LinearModel(self.weight + other.weight, self.bias + other.bias)


@dataclass
class Ensemble:
    """Composition ('has-a' models) instead of inheritance."""
    models: list[Model] = field(default_factory=list)

    def predict(self, x: float) -> float:
        return sum(m(x) for m in self.models) / len(self.models)


if __name__ == "__main__":
    m1 = LinearModel(2.0, 1.0)
    m2 = LinearModel(1.5, -0.5)
    print(m1(5))            # __call__ -> predict(5) = 11.0
    print(m1 + m2)           # __add__ overload
    ens = Ensemble([m1, m2])
    print(ens.predict(5))    # averaged prediction
```

**Notes:** `Model.__call__` delegating to `self.predict` is *exactly* the mechanism `torch.nn.Module.__call__` uses to delegate to `forward()` (plus hooks) — recognizing this pattern here means PyTorch's model API will feel familiar rather than magical in Phase 5.

---

## 6. Build From Scratch

**A minimal `property`-like descriptor (mirrors what `@property` does):**
```python
class ValidatedAttribute:
    """Descriptor enforcing a value constraint — mirrors sklearn param validation."""
    def __init__(self, min_value=0):
        self.min_value = min_value

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if value < self.min_value:
            raise ValueError(f"{value} below minimum {self.min_value}")
        setattr(instance, self.name, value)


class Account:
    balance = ValidatedAttribute(min_value=0)
    def __init__(self, balance):
        self.balance = balance   # triggers ValidatedAttribute.__set__
```

---

## 7. Library Implementation (Comparison)

| Concept from scratch | Real library usage |
|---|---|
| `ValidatedAttribute` descriptor | `pydantic.BaseModel` field validators (FastAPI request models, Phase 8) |
| `Model.__call__` → `predict` | `torch.nn.Module.__call__` → `forward` (Phase 5) |
| `ABC` + `@abstractmethod` | `sklearn.base.BaseEstimator`/`TransformerMixin` interfaces |
| `Ensemble` composition | `sklearn.ensemble.VotingClassifier` internals |

---

## 8. Visual Explanations

```
        Model (ABC)
       /            \
 LinearModel     TreeModel   <- inheritance ("is-a")

 Ensemble --- has-a ---> [Model, Model, Model]   <- composition ("has-a")
```

**Attribute lookup chain (instance -> class -> MRO -> object):**
```
instance.attr
   │
   ├─ found in instance.__dict__? -> return
   └─ not found -> walk type(instance).__mro__
                    -> found in some class __dict__? -> return (bound method/descriptor)
                    -> not found anywhere -> AttributeError
```

---

## 9. Practical Examples

**Simple:** a `Shape` base class with `Circle`/`Rectangle` subclasses implementing `area()`.
**Medium:** an abstract `DataLoader` base class with `CSVLoader`/`JSONLoader` subclasses sharing a common `validate()` method.
**Real-world:** a `BaseRiskModel` ABC with `MortalityModel`/`LapseModel` subclasses (directly relevant to actuarial engineering work) sharing common serialization (`to_dict`/`from_dict`) and validation logic via inheritance, while composing a shared `FeaturePipeline` object.

---

## 10. Real Industry Use Cases

- **PyTorch**: `nn.Module` is the universal base class for every layer and model — inheritance + `__call__` + hooks.
- **scikit-learn**: `BaseEstimator`, `TransformerMixin`, `ClassifierMixin` — mixin-based composition of `fit`/`predict`/`transform` interfaces, enabling `Pipeline` to treat wildly different models uniformly (polymorphism).
- **Hugging Face Transformers**: `PreTrainedModel` base class standardizes `.from_pretrained()`, `.save_pretrained()`, `.generate()` across hundreds of architectures.
- **FastAPI/Pydantic**: request/response schemas are classes using descriptor-based validation.

---

## 11. Common Mistakes

- Overusing deep inheritance hierarchies ("inheritance for code reuse") when composition is more flexible and testable — the classic "favor composition over inheritance" principle.
- Forgetting `super().__init__()` in a subclass, silently skipping parent initialization.
- Mutable class-level attributes (`class Foo: items = []`) shared unintentionally across *all* instances — the class-level analog of the mutable-default-argument bug.
- Overriding `__eq__` without also overriding `__hash__`, breaking dict/set usage of instances.

---

## 12. Best Practices (2026)

- Use `dataclasses` (or `attrs`/`pydantic` for validation-heavy cases) instead of hand-written boilerplate `__init__`/`__repr__`/`__eq__`.
- Prefer `ABC` + `@abstractmethod` to document required interfaces explicitly rather than relying purely on duck typing for library code.
- Use `Protocol` (structural typing, `typing.Protocol`) when you want duck-typed interfaces *with* static type checking — increasingly standard in 2026 typed Python codebases.
- Favor composition + small, focused mixins over deep multi-level inheritance trees.

---

## 13. Exercises

**Easy:** Implement a `Animal` base class with `Dog`/`Cat` subclasses overriding a `speak()` method; call polymorphically from a list.
**Medium:** Implement `__eq__`, `__hash__`, and `__lt__` on a `Version` class so instances can be sorted and deduplicated in a set.
**Hard:** Implement a custom metaclass that automatically registers every subclass of a `Plugin` base class into a global registry dict (the "plugin pattern" used by many ML frameworks for registering model architectures).
**Mathematical:** Compute the MRO by hand (C3 linearization) for a diamond inheritance case with 4 classes, then verify with `ClassName.__mro__`.
**Coding:** Build a `Serializable` mixin providing generic `to_json`/`from_json` via `dataclasses.asdict`, usable by any dataclass.

---

## 14. Mini Project

Build a small **model registry system**: an abstract `BaseModel` class defining `fit`, `predict`, `save`, `load`; concrete subclasses `MeanBaselineModel` and `LinearRegressionModel` (from-scratch, no sklearn yet); a metaclass-based registry so any new subclass is automatically discoverable by name string (`ModelRegistry.get("linear")`) — this pattern is exactly how you'll later plug in new model architectures into larger systems.

---

## 15. Interview Preparation

- Explain the difference between `@staticmethod`, `@classmethod`, and an instance method.
- What problem does `super()` solve in multiple inheritance, and how does MRO relate to it?
- Why does Python not have true private attributes, and how does `__name_mangling` work?
- Design question: how would you design a class hierarchy for multiple interchangeable ML model backends (sklearn, XGBoost, PyTorch) behind one unified `predict()` interface?

---

## 16. Summary

OOP in Python gives you encapsulation, inheritance, and polymorphism through a deliberately flexible, convention-driven object model (no enforced access control, duck typing over strict interfaces, but `ABC`/`Protocol` available when you want explicit contracts). Every major ML library's public API is a class hierarchy; understanding `__init__`/`__call__`/inheritance/MRO here directly unlocks reading PyTorch, scikit-learn, and Transformers source code without confusion later.

---

## 17. References

- Ramalho, L. — *Fluent Python*, Part on Object-Oriented Idioms
- Raymond Hettinger — "Python's Class Development Toolkit" (PyCon talk)
- Official docs: `dataclasses`, `abc`, `typing.Protocol`
- Gang of Four — *Design Patterns* (for OOP design vocabulary used in Lesson 10)
