# Phase 1 · Lesson 9 — Testing (pytest)

> Prerequisite: Lessons 1–8

---

## 1. Introduction

### What is this topic?
Automated testing: writing code that verifies other code behaves correctly, executed by a test runner. `pytest` is the dominant Python testing framework in 2026 — simpler syntax than the standard library's `unittest`, powerful fixture system, and a massive plugin ecosystem (`pytest-cov`, `pytest-asyncio`, `pytest-mock`, `hypothesis` for property-based testing).

### Why does it exist?
Manual testing ("run the script, eyeball the output") does not scale and does not protect against regressions when code changes six months later. Automated tests encode expected behavior *permanently*, letting you refactor fearlessly (directly enabling Lesson 8's refactoring discipline) and catch breakage the moment it's introduced, not weeks later in production.

### Historical background
xUnit-style testing (JUnit, 1997) established the assert-based test pattern now universal. `pytest` (2004, still actively developed) succeeded because it eliminated JUnit-style boilerplate (`assertEqual(a, b)` → just `assert a == b`, with pytest's assertion rewriting giving readable failure diffs automatically).

### Real-world motivation
ML code has unique testing challenges beyond typical software: you must test not just "does this function run" but "does this data pipeline preserve invariants," "does this model produce reproducible outputs given a fixed seed," and "does a retrained model's performance stay within an acceptable band" — testing philosophy extends directly into Phase 8's evaluation/monitoring topics.

---

## 2. Theory

### Test types (the testing pyramid)
| Level | Scope | Speed | Count (typical ratio) |
|---|---|---|---|
| Unit test | one function/class in isolation (dependencies mocked) | milliseconds | many (base of the pyramid) |
| Integration test | multiple components together (e.g., real DB) | seconds | fewer |
| End-to-end test | full system, real-like environment | slow (seconds–minutes) | fewest (top of pyramid) |

### Key concepts
- **Assertion**: a boolean check that must hold; failure raises `AssertionError` with a diagnostic message.
- **Fixture**: reusable setup/teardown logic injected into test functions (pytest's signature feature — dependency injection for tests).
- **Mock/stub**: a fake stand-in for a real dependency (e.g., a fake API client) so a unit test doesn't depend on network/external state.
- **Test coverage**: the percentage of code lines/branches executed by the test suite — a *necessary but not sufficient* quality signal (100% coverage with weak assertions still catches nothing).
- **Property-based testing**: instead of hand-picking example inputs, generate many random inputs satisfying a specification and check invariants hold for all of them (`hypothesis` library).

---

## 3. Mathematical Foundations

### Coverage as a set-cover problem
If $P$ = the set of all executable paths through a program, and each test $t_i$ exercises a subset $S_i \subseteq P$, then achieving "$k\%$ path coverage" is a set-cover problem:
$$
\bigcup_i S_i \supseteq k\% \text{ of } P
$$
Full path coverage is generally intractable for realistic programs (paths grow exponentially with branching — Lesson 8's cyclomatic complexity directly predicts this), which is *why* line/branch coverage (weaker, tractable proxies) are used in practice instead of full path coverage.

### Statistical testing of ML models
Unlike deterministic software, ML model outputs are often stochastic or approximate. Testing a model's *quality* (not just "does it run") requires statistical reasoning:
$$
H_0: \text{new model performance} \le \text{baseline performance}
$$
Rejecting $H_0$ (e.g., via a paired t-test or bootstrap confidence interval on a held-out metric) with a pre-registered significance threshold is the rigorous way to test "did this change actually improve the model" — a direct bridge to Phase 3 (Statistics) and Phase 4 (Model Evaluation).

### Flakiness and probability
A flaky test that fails randomly with probability $p$ per run will, over $n$ CI runs, produce at least one false failure with probability $1 - (1-p)^n$ — which approaches 1 quickly even for small $p$. This is the mathematical reason flaky tests are treated as a critical engineering problem, not a minor annoyance, in any CI pipeline running hundreds of times a day.

---

## 4. Algorithm — Test Discovery and Execution (how pytest works)

```
1. pytest scans directories for files matching test_*.py or *_test.py
2. Within each file, collects functions named test_* and classes named Test* (methods test_*)
3. For each test function:
     a. Resolve and construct any requested fixtures (recursively, respecting scope: function/class/module/session)
     b. Run the test function body
     c. Catch AssertionError -> report FAILED with a rich diff
        Catch any other Exception -> report ERROR
        No exception -> report PASSED
4. Teardown fixtures in reverse order of setup
5. Aggregate results: X passed, Y failed, Z errors, report summary + exit code (0 = all passed)
```
Fixture scope resolution is itself a dependency-graph problem (session-scoped fixtures built once, reused across many tests; function-scoped rebuilt every test) — directly reusing DAG concepts from Lesson 5 (Git) and Lesson 4 (graphs).

---

## 5. Python Implementation

```python
"""test_mortality_features.py — realistic pytest suite for an ML feature pipeline"""
import pytest
import pandas as pd
from mortality_features import clean_data, compute_age_band, MortalityFeaturePipeline


# --- Fixtures: reusable, composable test setup ---
@pytest.fixture
def raw_patient_df() -> pd.DataFrame:
    """Function-scoped by default: fresh copy for every test that uses it."""
    return pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "age": [45, None, 72, 30],
        "systolic_bp": [120, 135, 150, None],
    })


@pytest.fixture(scope="module")
def trained_pipeline(raw_patient_df) -> MortalityFeaturePipeline:
    """Module-scoped: expensive setup built ONCE and reused across all tests in this file."""
    pipeline = MortalityFeaturePipeline()
    pipeline.fit(clean_data(raw_patient_df))
    return pipeline


# --- Basic unit tests ---
def test_clean_data_drops_rows_with_missing_age(raw_patient_df):
    cleaned = clean_data(raw_patient_df)
    assert cleaned["age"].isna().sum() == 0
    assert len(cleaned) == 3   # P2 dropped (age was None)


def test_compute_age_band_boundaries():
    assert compute_age_band(0) == "0-9"
    assert compute_age_band(9) == "0-9"
    assert compute_age_band(10) == "10-19"   # boundary condition — easy to get off-by-one wrong


# --- Parametrized test: run the same logic across many inputs efficiently ---
@pytest.mark.parametrize("age,expected_band", [
    (0, "0-9"), (5, "0-9"), (10, "10-19"), (99, "90-99"), (100, "100+"),
])
def test_age_band_parametrized(age, expected_band):
    assert compute_age_band(age) == expected_band


# --- Exception testing ---
def test_compute_age_band_rejects_negative():
    with pytest.raises(ValueError, match="age must be non-negative"):
        compute_age_band(-5)


# --- Testing against invariants, not just exact outputs (important for ML pipelines) ---
def test_pipeline_output_invariants(trained_pipeline, raw_patient_df):
    features = trained_pipeline.transform(clean_data(raw_patient_df))
    assert not features.isna().any().any(), "no NaNs should survive the pipeline"
    assert (features["age"] >= 0).all(), "age feature must never be negative"


# --- Using a mock to isolate from an external dependency (e.g. a model registry API) ---
def test_pipeline_save_calls_registry_once(mocker, trained_pipeline):
    mock_registry = mocker.patch("mortality_features.ModelRegistry.upload")
    trained_pipeline.publish("v1.2.0")
    mock_registry.assert_called_once_with("v1.2.0")
```

**Run with:** `pytest -v --cov=mortality_features --cov-report=term-missing`

---

## 6. Build From Scratch

**A minimal test runner (to demystify what pytest does under the hood):**
```python
import traceback

def run_tests(module):
    test_funcs = [getattr(module, name) for name in dir(module) if name.startswith("test_")]
    passed, failed = 0, 0
    for func in test_funcs:
        try:
            func()
            print(f"PASSED: {func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {func.__name__} -> {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {func.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0
```
This captures pytest's core loop: discover `test_*` functions, run each in isolation, catch `AssertionError` specifically (vs. other exceptions), aggregate results. Missing (deliberately, for simplicity): fixture injection, parametrization, plugin hooks, rich assertion rewriting — all of which are why you use real `pytest` in production, not this toy version.

---

## 7. Library/Tool Comparison

| From scratch | pytest |
|---|---|
| `run_tests` loop | full CLI (`pytest -k`, `-m`, `-x`, `--lf`), parallel execution (`pytest-xdist`) |
| No fixture support | rich fixture system with scopes, autouse, dependency injection |
| Plain `assert` | assertion rewriting (shows *why* `a == b` failed, e.g., diffing lists/dicts) |
| No mocking | `pytest-mock`/`unittest.mock` integration |
| No coverage | `pytest-cov` (wraps `coverage.py`) |

---

## 8. Visual Explanations

**Testing pyramid:**
```
        ▲
       / \        E2E tests (few, slow, high confidence, brittle)
      /---\
     /     \      Integration tests (moderate count)
    /-------\
   /         \    Unit tests (many, fast, isolated)
  /___________\
```

**Fixture scope lifecycle:**
```
session scope:  [ built once ]───────────────────────[ torn down at end of run ]
module scope:      [ built once per file ]────[ torn down ]
function scope:        [built][test][torn down] [built][test][torn down] ...
```

---

## 9. Practical Examples

**Simple:** test a `celsius_to_fahrenheit` function against 3 known conversion values.
**Medium:** test a CSV-loading function raises a clear error on malformed input (missing required column).
**Real-world:** a `test_no_data_leakage` test verifying that a train/test split function never lets the same `patient_id` appear in both splits — a real, subtle, high-stakes ML bug class this kind of test specifically exists to catch.

```python
def test_no_patient_overlap_in_split(raw_patient_df):
    train_df, test_df = time_based_split(raw_patient_df)
    overlap = set(train_df["patient_id"]) & set(test_df["patient_id"])
    assert overlap == set(), f"Data leakage: {overlap} in both train and test"
```

---

## 10. Real Industry Use Cases

- **Every AI lab's CI pipeline**: unit tests on tokenizers, data loaders, and model architecture code run on every commit before merge (Phase 8, CI/CD).
- **Hugging Face Transformers**: thousands of model-specific tests ensure every architecture produces expected output shapes/values against fixed reference tensors — critical given hundreds of contributors.
- **Google/Meta ML infra**: golden-file / snapshot testing for model outputs (does this model's output on a fixed input match a saved reference within tolerance) is standard practice to catch silent regressions from library upgrades.
- **Insurance/actuarial software (directly relevant to your domain)**: regulatory-grade pricing/reserving models are typically required to have extensive test suites validating formula correctness against known reference calculations — testing discipline here has compliance, not just engineering, weight.

---

## 11. Common Mistakes

- Testing implementation details instead of behavior (e.g., asserting a private helper was called a specific way) — makes tests brittle, breaking on harmless refactors.
- Writing tests with no assertions ("smoke tests" that just check "it didn't crash") and calling it sufficient coverage.
- Chasing 100% coverage as a goal in itself, while leaving critical edge cases (empty input, all-NaN column, single-row DataFrame) completely untested.
- Non-deterministic tests (relying on real randomness, real time, or real network) without fixing seeds/mocking — the primary cause of CI flakiness (tie back to Section 3's flakiness math).

---

## 12. Best Practices (2026)

- Fix random seeds explicitly in any test touching stochastic code (`np.random.seed`, `torch.manual_seed`) — non-negotiable for ML test determinism.
- Use `pytest.mark.parametrize` liberally instead of copy-pasting near-identical test functions (DRY, from Lesson 8).
- Separate fast unit tests (run on every save/commit) from slow integration tests (run in CI only, or nightly) using `pytest.mark.slow` + `pytest -m "not slow"` for local dev speed.
- For ML-specific correctness, combine classic unit tests (deterministic code paths) with statistical tests (Section 3) for model-quality regressions, and consider `hypothesis` for property-based testing of data-cleaning invariants (e.g., "cleaning is idempotent — cleaning twice equals cleaning once").

---

## 13. Exercises

**Easy:** Write unit tests for the `binary_search` function from Lesson 4, including the "not found" case and empty-array edge case.
**Medium:** Write a parametrized test suite for the `compute_age_band` function covering every documented boundary.
**Hard:** Write a `hypothesis`-based property test asserting that `merge_sort` (Lesson 4) always returns a list that is (a) sorted and (b) a permutation of the input, for arbitrary randomly generated lists.
**Mathematical:** Given a flaky test with 2% per-run failure probability, compute the probability of at least one false failure across 100 CI runs/day over a month (30 days) — and discuss why this makes even "small" flakiness rates unacceptable at scale.
**Coding:** Extend Section 6's toy test runner to support a simple `@fixture`-like decorator providing function-scoped setup/teardown.

---

## 14. Mini Project

Take the **Model Registry mini project from Lesson 3** (OOP) and write a complete `pytest` suite for it: unit tests for each concrete model class's `fit`/`predict`, a parametrized test verifying every registered model type can be instantiated by name from the registry, a mocked test for the save/load functionality (no real disk I/O), and a coverage report (`pytest --cov`) — aim for meaningful coverage of both the happy path and edge cases (empty input, unregistered model name).

---

## 15. Interview Preparation

- What's the difference between a unit test, an integration test, and an end-to-end test, and what ratio would you aim for?
- How do you test code that depends on an external API or database?
- What is test coverage, and why isn't 100% coverage sufficient evidence of correctness?
- System design/ML-specific: how would you design a test suite to catch data leakage and train/serve skew in a production ML pipeline before it reaches deployment?

---

## 16. Summary

Automated testing converts "I believe this code works" into "I can prove this code works, repeatably, forever, at the cost of one command." `pytest`'s fixture system and assertion rewriting remove the historical boilerplate excuse for skipping tests, and ML-specific testing (data invariants, leakage checks, statistical model-quality tests) extends the same discipline into the genuinely different failure modes of ML systems — a foundation every later phase (especially Phase 8's CI/CD and Phase 9's capstones) assumes is already in place.

---

## 17. References

- Official pytest documentation: https://docs.pytest.org/
- Percival, H. — *Test-Driven Development with Python*
- `hypothesis` documentation (property-based testing)
- Google Testing Blog — practical testing philosophy at scale
