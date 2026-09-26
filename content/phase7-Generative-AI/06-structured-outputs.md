# Phase 7 · Lesson 6 — Structured Outputs

> Prerequisite: Tool Calling (Lesson 5)

---

## 1. Introduction

### What are structured outputs?
Techniques for guaranteeing (or strongly encouraging) an LLM's response conforms to a specific, predefined format — JSON matching a schema, a specific XML structure, or another machine-parseable format — rather than the free-form natural language text an LLM produces by default. This is the same underlying mechanism as Lesson 5's tool-calling (constrained decoding), applied more generally to any situation where a program downstream needs to reliably parse an LLM's output, not just when the output represents a tool invocation.

### Why does it exist?
Any LLM output that feeds into further programmatic processing (populating a database, driving a UI, triggering a downstream API call) requires reliable parsing — free-form text responses, even when *usually* well-formatted, will eventually produce a variation that breaks a naive parser (extra commentary before the JSON, inconsistent field names, malformed syntax). Structured output techniques exist to make this reliability a guaranteed property of the system rather than a hopeful expectation.

### Historical background
Early practitioners (2020-2022) relied entirely on prompt engineering ("respond only in JSON matching this format...") — better than nothing, but fundamentally unreliable, since a language model's raw next-token sampling process has no inherent guarantee of producing syntactically valid output. The maturation of constrained decoding techniques (directly building on Lesson 5's mechanism) and providers' native "structured output" / "JSON mode" API features (2023-2024 onward) converted this from a best-effort prompting technique into a genuinely reliable, guaranteed capability.

### Real-world motivation
Every RAG citation format (Lesson 1), every tool-call argument (Lesson 5), and every data-extraction pipeline built on an LLM depends on structured output reliability — this lesson consolidates the specific techniques and tradeoffs involved.

---

## 2. Theory

### The reliability spectrum: prompting alone vs. constrained decoding vs. native structured-output modes
- **Prompting alone** ("please respond in JSON"): no guarantee; the model can and occasionally will violate the requested format, especially on longer or more complex outputs.
- **Constrained decoding** (Lesson 5's mechanism, e.g., via the `outlines`/`guidance` libraries or providers' native support): the token sampling process itself is restricted to only produce schema-valid output — a structural guarantee, not a hopeful request.
- **Provider-native structured-output modes**: many LLM APIs now offer a direct parameter (e.g., specifying a JSON schema or a Pydantic-style model) that internally applies constrained decoding, giving you the reliability guarantee without needing to implement the constraint-filtering logic yourself.

### Schema definition approaches
- **JSON Schema**: the general-purpose, widely-supported standard (also used for Lesson 5's tool parameter schemas) for describing valid JSON structure, types, and constraints.
- **Pydantic models** (Python): a popular, more ergonomic way to define schemas as Python classes, which libraries (Instructor, and native SDK integrations) can automatically convert into the underlying JSON Schema needed for constrained generation, while also providing automatic parsing/validation of the returned output back into a typed Python object.

### Nested and complex structures
Structured output techniques extend to arbitrarily nested schemas (objects containing lists of objects, optional fields, enums with a fixed set of allowed values) — genuinely useful for real extraction tasks (e.g., extracting a list of line items from an invoice, each with its own sub-fields), though very deep nesting or very large schemas can, in practice, degrade the *semantic* quality of the extracted content even when syntactic validity remains guaranteed (constrained decoding guarantees the *shape* is correct, not that the *content* correctly reflects the source material).

### Structured output vs. tool calling — a genuinely blurry, overlapping distinction
Lesson 5's tool calling is, technically, a specific application of structured output (the "tool call" itself is a structured object). Many practitioners use "structured output" to specifically mean requesting a final structured *answer* (not an intermediate tool invocation) — e.g., asking a model to extract data into a JSON object as its final response, rather than using that JSON to trigger a subsequent function call — a useful distinction for reasoning about *when* in an application's flow the structuring is happening, even though the underlying mechanism is often identical.

---

## 3. Mathematical Foundations

### Constrained decoding as logit masking (making Lesson 5's mechanism fully precise here)
At each generation step, given a partial output so far, define the set of tokens $V_{valid} \subseteq V$ (the full vocabulary, Phase 6 Lesson 1) that would keep the output on a path toward eventual schema validity. The model's softmax output (Phase 5 Lesson 1) is modified:

$$
P_{constrained}(\text{token}_i) = \begin{cases} \dfrac{\exp(z_i/T)}{\sum_{j \in V_{valid}} \exp(z_j/T)} & i \in V_{valid} \\ 0 & i \notin V_{valid} \end{cases}
$$

— renormalizing probability mass entirely over the valid subset, guaranteeing the sampled token (however sampled — greedy, temperature-based, Phase 5 Lesson 7) is always schema-consistent. Computing $V_{valid}$ efficiently at every generation step (potentially thousands of times per response) requires an efficient incremental grammar/schema parser — a genuine engineering challenge underneath libraries like `outlines`, not merely a conceptual footnote.

### The precision-vs-content-quality tradeoff
Constrained decoding guarantees **syntactic** validity (the output *shape* always matches the schema) but says nothing about **semantic** correctness (whether the *content* filled into that shape accurately reflects the source information) — these are formally independent properties. A model can produce a perfectly schema-valid JSON object containing entirely fabricated field values — structured output solves a parsing-reliability problem, not a hallucination problem (Lesson 1's concern), and both must be addressed, typically through different techniques (RAG grounding, evaluation, Lesson 10).

### Schema complexity and generation latency
Very large or deeply nested schemas can meaningfully increase the computational overhead of constrained decoding (more complex valid-token-set computation at each step) and can also increase the number of output tokens needed (more verbose JSON structure) — a genuine, quantifiable latency/cost tradeoff (Phase 6 Lesson 1's tokenization cost accounting, resurfacing here) worth considering when designing extraction schemas for production use.

---

## 4. Algorithm — Structured Extraction with Validation and Retry (fully specified)

```
GIVEN a target schema, an LLM with structured-output support, and a source text to extract from:
1. CONSTRUCT the request: source text + schema (as a JSON Schema or Pydantic model) + extraction instructions
2. GENERATE: the LLM produces output constrained to match the schema (Section 3's mechanism)
3. PARSE: deserialize the output into the target structure (e.g., a Pydantic object)
   -- with a properly schema-constrained API, this step should NEVER fail on syntax/shape grounds
4. VALIDATE semantic correctness (beyond mere shape):
     check required business-logic constraints the schema alone can't express
     (e.g., "end_date must be after start_date", a cross-field constraint)
5. IF semantic validation fails:
     RE-PROMPT the model with the validation error as additional context, requesting a correction
     (a "retry with feedback" loop, bounded by a max-retry count -- Lesson 3's iteration-safeguard principle)
6. RETURN the validated, parsed structured object
```

---

## 5. Python Implementation

```python
"""structured_outputs_core.py — Pydantic-based schema definition, validation, and retry logic"""
from pydantic import BaseModel, Field, field_validator
from datetime import date


class PolicyExtraction(BaseModel):
    """A schema for extracting structured data from an insurance policy document."""
    policy_number: str = Field(description="The unique policy identifier.")
    coverage_amount: float = Field(gt=0, description="Total coverage amount in the policy's currency.")
    start_date: date = Field(description="Policy start date.")
    end_date: date = Field(description="Policy end date.")
    exclusions: list[str] = Field(default_factory=list, description="List of explicitly stated exclusions.")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        # Cross-field validation -- exactly Section 4's "semantic validation beyond mere shape" step
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


def extract_with_retry(source_text: str, llm_structured_call, max_retries: int = 2) -> PolicyExtraction | None:
    """llm_structured_call: a function that takes (text, schema, error_feedback) and returns raw JSON/dict,
    representing a real structured-output-mode LLM API call."""
    error_feedback = None
    for attempt in range(max_retries + 1):
        raw_output = llm_structured_call(source_text, PolicyExtraction.model_json_schema(), error_feedback)
        try:
            return PolicyExtraction.model_validate(raw_output)   # parses AND validates in one step
        except Exception as e:
            error_feedback = str(e)
            if attempt == max_retries:
                return None    # give up gracefully after exhausting retries (Lesson 3's safeguard principle)
    return None


# Illustrative mock (a real implementation calls an actual LLM API with structured-output mode enabled)
def mock_llm_structured_call(text: str, schema: dict, error_feedback: str | None) -> dict:
    # Simulating a first attempt with a validation-breaking error, corrected on retry
    if error_feedback is None:
        return {
            "policy_number": "POL-48291", "coverage_amount": 250000.0,
            "start_date": "2024-01-01", "end_date": "2023-12-31",   # INVALID: end before start
            "exclusions": ["flood damage", "acts of war"],
        }
    return {   # corrected on the retry, informed by the error_feedback
        "policy_number": "POL-48291", "coverage_amount": 250000.0,
        "start_date": "2024-01-01", "end_date": "2024-12-31",
        "exclusions": ["flood damage", "acts of war"],
    }


result = extract_with_retry("(a policy document's text would go here)", mock_llm_structured_call)
print(result)
```

---

## 6. Build From Scratch

**A minimal incremental JSON-validity checker (illustrating the "is this partial output still on a valid path" check underlying Section 3's constrained decoding, at a simplified level):**
```python
def incremental_json_validity_state(partial: str) -> str:
    """Returns 'valid_complete', 'valid_partial', or 'invalid' for a partial JSON string --
    a SIMPLIFIED illustration; real implementations use a proper streaming JSON/grammar parser."""
    depth = 0
    in_string = False
    escape_next = False
    for char in partial:
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth < 0:
                return "invalid"       # more closes than opens -- structurally broken
    if in_string:
        return "valid_partial"          # mid-string, could still become valid
    if depth == 0 and partial.strip():
        import json
        try:
            json.loads(partial)
            return "valid_complete"
        except json.JSONDecodeError:
            return "invalid"
    return "valid_partial"

for test in ['{"a": 1, "b":', '{"a": 1, "b": 2}', '{"a": 1}}', '{"a": "unterminated']:
    print(f"{test!r:35} -> {incremental_json_validity_state(test)}")
```
This is a genuine (if simplified relative to production-grade implementations) working sketch of the bracket-depth and string-state tracking that any incremental JSON-validity checker needs — real constrained-decoding libraries extend this exact idea into a full grammar parser integrated directly with the model's token-by-token sampling loop.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `incremental_json_validity_state` (simplified) | `outlines`, `guidance` — full grammar-aware constrained decoding libraries operating directly on model logits |
| Manual Pydantic + retry loop (Section 5) | `Instructor` library — purpose-built wrapper adding Pydantic-based structured extraction with automatic retry/validation directly on top of major LLM provider APIs |
| Manual JSON Schema construction | Provider-native structured-output modes (passing a Pydantic model or JSON Schema directly to the API) — increasingly the simplest, most reliable production approach |

---

## 8. Visual Explanations

**Structured output reliability spectrum:**
```
LEAST reliable                                                        MOST reliable
Prompting alone  ──▶  Regex/manual parsing of free text  ──▶  Provider-native structured-output mode
("please use JSON")   (Lesson 3's simplified illustration)     (constrained decoding, GUARANTEED validity)
```

**Syntactic validity vs. semantic correctness (two independent axes):**
```
                    Semantically CORRECT     Semantically WRONG
Syntactically       ┌──────────────────┐    ┌──────────────────┐
VALID                │   ideal outcome   │    │  valid JSON, but  │
                     │                   │    │  FABRICATED data  │
                     └──────────────────┘    └──────────────────┘
Syntactically       ┌──────────────────┐    ┌──────────────────┐
INVALID              │  (rare, with real │    │    worst case:    │
                     │   constrained      │    │  broken AND wrong │
                     │   decoding)         │    │                   │
                     └──────────────────┘    └──────────────────┘
   Constrained decoding ELIMINATES the bottom row entirely -- but the RIGHT column remains
   a real risk requiring separate mitigation (RAG grounding, Lesson 1; evaluation, Lesson 10)
```

---

## 9. Practical Examples

**Simple:** define a Pydantic schema for a simple structured extraction task and use a real LLM API's native structured-output mode to reliably extract matching data from a short text.
**Medium:** implement the retry-with-feedback loop (Section 5) and test it against a schema with a cross-field validation rule, verifying the model successfully self-corrects when given the validation error as feedback.
**Real-world:** build a structured extraction pipeline for actuarial policy documents (extending this lesson's `PolicyExtraction` example with real, more complete fields relevant to your domain), testing it across a diverse set of real or synthetic policy texts and measuring both syntactic success rate (should be ~100% with native structured-output mode) and semantic accuracy (does the extracted data actually match the source document).

---

## 10. Real Industry Use Cases

- **Every LLM-based data-extraction pipeline** (invoice processing, resume parsing, medical record structuring, and directly relevant, insurance policy/claims document processing): relies on exactly this lesson's techniques.
- **Structured output as the backbone of Lesson 5's tool calling and Lesson 4's MCP protocol**: both are, at the implementation level, specific applications of the same constrained-decoding mechanism.
- **API/UI-driving LLM responses**: any application where an LLM's output directly populates a user interface or triggers programmatic logic (not just displaying free text to a human) depends on structured output reliability.
- **Multi-agent systems** (Lesson 8): agents often communicate with each other via structured messages rather than free text, for the same parsing-reliability reasons.

---

## 11. Common Mistakes

- Relying on prompt engineering alone ("please respond in valid JSON") for any application where parsing failures have real consequences — use native structured-output modes or constrained-decoding libraries instead.
- Conflating syntactic validity with semantic correctness — a perfectly schema-valid extraction can still contain fabricated or misread content; these require separate mitigation strategies.
- Designing overly complex, deeply nested schemas without considering the latency/cost and content-quality tradeoffs (Section 3) — simpler, flatter schemas are often both faster and more reliably filled with correct content.
- Not implementing cross-field/business-logic validation beyond what the schema format itself can express (e.g., JSON Schema alone can't easily express "end_date > start_date") — this requires additional validation logic (Section 5's `field_validator` pattern) layered on top of basic schema conformance.

---

## 12. Best Practices (2026)

- Use provider-native structured-output modes (or well-established libraries like `Instructor`) rather than hand-rolled prompt-engineering-only approaches for any production data-extraction or tool-argument-construction task.
- Define schemas using Pydantic (in Python) for the combined benefit of clear schema definition, automatic JSON Schema generation, and automatic parsing/validation of returned data.
- Implement a bounded retry-with-feedback loop for handling semantic/business-logic validation failures, rather than silently accepting invalid extracted data or failing without any recovery attempt.
- Keep schemas as simple/flat as the task genuinely requires — added structural complexity has real latency and content-quality costs, not just development complexity costs.

---

## 13. Exercises

**Easy:** Define a simple Pydantic schema for extracting a person's name, age, and occupation from a short biography text, and use a real LLM API's structured-output mode to extract matching data.
**Medium:** Add a cross-field validator (e.g., ensuring a "graduation_year" is after a "birth_year" plus some minimum age) and test the retry-with-feedback loop (Section 5) on inputs that initially violate it.
**Hard:** Implement the incremental JSON-validity checker (Section 6) and use it to build a simple, working (if simplified relative to production-grade) constrained-decoding wrapper around a raw (non-structured-output-mode) LLM API, comparing its reliability against unconstrained free-text generation with a JSON-formatting prompt.
**Mathematical:** Using Section 3's logit-masking formula, work through a small concrete example (a tiny 5-token vocabulary, a simple 2-state schema constraint) showing exactly how the valid-token-set filtering changes the effective probability distribution at one generation step.
**Coding:** Build a nested schema (e.g., a list of line items, each with its own sub-fields) for an invoice-extraction task, and test extraction quality/reliability on both simple and deliberately complex/edge-case invoice texts.

---

## 14. Mini Project

Build a **complete, validated actuarial document extraction pipeline**: design a comprehensive Pydantic schema for extracting structured data from insurance policy or claims documents (including at least one meaningful cross-field business-logic validator), implement the retry-with-feedback loop for handling validation failures, test the pipeline across a diverse set of real or synthetic documents, and produce a report distinguishing syntactic success rate (should be near-100% with proper structured-output mode) from semantic accuracy (manually or systematically checked against ground truth) — a genuinely practical, evaluated extraction system directly applicable to your domain.

---

## 15. Interview Preparation

- Explain the reliability spectrum from prompt-engineering-only structured output to native constrained decoding.
- What is constrained decoding, and how does it guarantee schema validity at the token-sampling level?
- Why is syntactic validity insufficient on its own for a reliable data-extraction pipeline — what else must be verified?
- How would you design a retry mechanism for handling semantic (not just syntactic) validation failures in structured extraction?

---

## 16. Summary

Structured outputs convert LLM generation from a hopeful, prompt-engineering-dependent process into a formally guaranteed one via constrained decoding — masking the model's token-sampling distribution (Phase 5 Lesson 7) at each step to permit only schema-valid continuations, the same underlying mechanism as Lesson 5's tool calling, now applied generally to any application needing reliably parseable LLM output. This guarantees syntactic validity but not semantic correctness — a crucial, independent distinction requiring separate validation/grounding strategies (cross-field business-logic checks, RAG grounding from Lesson 1, and the evaluation techniques of Lesson 10) — and together with well-designed Pydantic-based schemas and bounded retry-with-feedback loops, forms the reliable extraction/generation backbone underneath essentially every production LLM application that feeds its output into further programmatic processing.

---

## 17. References

- JSON Schema specification (json-schema.org)
- Willard & Louf — "Efficient Guided Generation for Large Language Models" (2023, the `outlines` library's foundational paper)
- `Instructor` library documentation (Python, structured extraction via Pydantic)
- Anthropic/OpenAI structured-output/JSON-mode API documentation
