# Phase 7 · Lesson 5 — Tool Calling

> Prerequisite: AI Agents, MCP (Lessons 3–4)

---

## 1. Introduction

### What is tool calling?
The mechanism by which an LLM, given a set of available function/tool definitions (with structured schemas describing their names, purposes, and parameters), can generate a structured request to invoke one of those functions with specific arguments — the concrete implementation detail underlying the "Action" step of Lesson 3's ReAct loop and the invocation step of Lesson 4's MCP protocol.

### Why does it exist?
Early "agent" implementations (2022-2023) relied on prompting an LLM to produce free-form text that was then parsed via regex or ad hoc string matching to extract an intended action (as illustrated, deliberately simplified, in Lesson 3's implementation) — fragile, error-prone, and difficult to validate. Native tool/function calling (introduced by OpenAI in mid-2023, rapidly adopted industry-wide) has the model directly output a structured, schema-validated function call object, dramatically improving reliability over free-text parsing.

### Historical background
Function calling was popularized by OpenAI's June 2023 API update, and essentially all major LLM providers (Anthropic, Google, and open-weight model ecosystems) now support an equivalent capability — the model is fine-tuned (Phase 6 Lesson 6) specifically to reliably produce well-formed structured calls when given tool schemas, rather than this being purely a prompt-engineering trick layered on top of a generic model.

### Real-world motivation
Every reliable production agentic system (Lessons 3, 7) uses native tool calling rather than text parsing — this lesson covers exactly how it works, how to design good tool schemas, and the common failure modes to guard against.

---

## 2. Theory

### The tool-calling request/response cycle
1. The calling application sends the LLM a prompt/conversation **plus** a list of available tool definitions (JSON Schema-style descriptions of each tool's name, description, and parameters).
2. The model, if it determines a tool call is appropriate, returns a structured response indicating which tool to call and with what arguments (rather than, or alongside, natural language text).
3. The calling application executes the actual function (the model itself never executes anything — it only *requests* an invocation).
4. The application sends the tool's result back to the model (as a new message in the conversation), and the model continues, now informed by that result.

### Tool schema design — the most consequential practical skill in this lesson
A tool's **name**, **description**, and **parameter schema** (including per-parameter descriptions and type constraints) directly determine how reliably the model selects and correctly uses that tool — this is genuinely a prompt-engineering-adjacent skill (Phase 6 Lesson 9): vague names/descriptions lead to the model choosing the wrong tool or misusing parameters, while clear, specific, example-illustrated schemas measurably improve reliability.

### Parallel vs. sequential tool calls
Modern tool-calling APIs often support the model requesting **multiple** tool calls in a single turn (e.g., "look up the weather in three different cities") — executed in parallel by the application for efficiency, then all results returned together before the model continues — a direct efficiency improvement over Lesson 3's strictly sequential ReAct loop for tasks with independent, parallelizable sub-actions.

### Forced vs. optional tool use
APIs typically support configuring whether the model *must* call a specific tool (or one of a specific set), *may optionally* call any available tool, or is *prevented* from calling tools entirely for a given turn — a genuinely useful control for application developers who sometimes need to guarantee structured output (Lesson 6) or guarantee a specific tool is used, rather than leaving the choice entirely to the model's judgment.

---

## 3. Mathematical Foundations

Tool calling is primarily an engineering/interface topic, but a few quantitative framings matter:

### Schema complexity and selection accuracy (an empirical, not purely theoretical, relationship)
As the number of available tools grows, and as individual tool schemas grow more complex (many parameters, deeply nested structures), empirical tool-selection accuracy tends to degrade — directly analogous to how classification accuracy degrades as the number of classes grows in classical ML (Phase 4 Lesson 1), and motivating careful curation of which tools are actually exposed to the model for a given task rather than exposing every available tool indiscriminately.

### The underlying mechanism — constrained decoding
Under the hood, structured tool-call generation is often implemented via **constrained decoding**: the model's next-token sampling (Phase 5 Lesson 7) is restricted at each step to only tokens consistent with valid JSON matching the target schema — formally, this modifies the softmax probability distribution (Phase 5 Lesson 1) at each generation step by masking out (setting to zero probability) any token that would violate the schema, guaranteeing syntactic validity by construction rather than hoping the model happens to produce well-formed output.

### Cost accounting for tool-augmented conversations
Each tool call/result round-trip adds tokens to the ongoing conversation context — directly reusing Phase 6 Lesson 1's tokenization cost accounting and Lesson 3's iteration-cost discussion: a task requiring $n$ tool calls, each with a result of average size $s$ tokens, adds roughly $n \times s$ tokens to the accumulated context across the interaction, a genuine, quantifiable cost/latency factor in agentic application design.

---

## 4. Algorithm — The Tool-Calling Request Cycle (fully specified)

```
GIVEN a conversation history, a set of tool schemas, and an LLM API supporting native tool calling:
1. SEND: conversation history + tool schemas + (optionally) a tool-choice constraint (forced/optional/none)
2. RECEIVE: the model's response, which is EITHER
     (a) a natural language text response (no tool call needed), OR
     (b) one or more structured tool-call requests (tool name + arguments, schema-validated)
3. IF tool call(s) requested:
     FOR each requested tool call (in parallel, if the API supports it):
         EXECUTE the actual underlying function with the given arguments
         CAPTURE the result (or a structured error if execution failed)
     APPEND each tool's result to the conversation as a "tool result" message
     RETURN TO STEP 1 (send the updated conversation back to the model, which now sees the results)
4. IF no tool call requested (pure text response):
     RETURN the text response to the end user / calling application
```

---

## 5. Python Implementation

```python
"""tool_calling_core.py — schema design, constrained-decoding intuition, and a full request cycle"""
import json
from typing import Callable


def build_tool_schema(name: str, description: str, parameters: dict, required: list[str]) -> dict:
    """Constructs a JSON-Schema-style tool definition, matching the format real LLM APIs expect."""
    return {
        "name": name,
        "description": description,          # CRITICAL: clear, specific descriptions improve selection accuracy
        "parameters": {
            "type": "object",
            "properties": parameters,          # each param should ALSO have its own "description"
            "required": required,
        },
    }


# A well-designed schema: specific description, typed/described parameters, sensible constraints
mortality_lookup_schema = build_tool_schema(
    name="get_mortality_rate",
    description=(
        "Looks up the annual mortality rate (probability of death within one year) "
        "for a given age and smoker status, based on standard actuarial tables."
    ),
    parameters={
        "age": {"type": "integer", "description": "Age in whole years, between 0 and 120.", "minimum": 0, "maximum": 120},
        "smoker": {"type": "boolean", "description": "Whether the individual is a smoker."},
    },
    required=["age", "smoker"],
)


class ToolExecutor:
    """Maps tool NAMES to their actual underlying Python functions, executing validated calls."""
    def __init__(self):
        self.functions: dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        self.functions[name] = func

    def execute(self, tool_call: dict) -> dict:
        name = tool_call["name"]
        args = tool_call["arguments"]
        if name not in self.functions:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = self.functions[name](**args)
            return {"result": result}
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"Execution error in {name}: {e}"}


def get_mortality_rate(age: int, smoker: bool) -> float:
    base_rate = 0.001 * (1.08 ** (age - 20))
    return base_rate * (2.5 if smoker else 1.0)


executor = ToolExecutor()
executor.register("get_mortality_rate", get_mortality_rate)

# Simulating what a real LLM API response containing a tool call looks like:
simulated_llm_tool_call_response = {
    "type": "tool_call",
    "name": "get_mortality_rate",
    "arguments": {"age": 55, "smoker": True},
}

if simulated_llm_tool_call_response["type"] == "tool_call":
    result = executor.execute(simulated_llm_tool_call_response)
    print("Tool execution result:", result)
    # In a real system, `result` would now be appended to the conversation and sent BACK to the LLM (Section 4)
```

---

## 6. Build From Scratch

**A minimal constrained-JSON-generation illustration (to make Section 3's constrained decoding concept concrete, at a toy scale):**
```python
import json

def is_valid_partial_json_prefix(partial: str) -> bool:
    """Extremely simplified check: could this string prefix still lead to valid JSON?
    (Real constrained decoding uses a proper grammar/schema-aware token filter, not this toy heuristic.)"""
    try:
        json.loads(partial)
        return True                      # already complete AND valid
    except json.JSONDecodeError:
        # a genuinely INCOMPLETE-but-still-viable prefix vs. a fundamentally broken one
        # is hard to distinguish with this naive approach -- real systems use proper
        # incremental JSON/grammar parsers integrated into the token sampling loop itself
        return partial.count("{") >= partial.count("}")   # a crude, illustrative heuristic only


def simulate_constrained_generation(candidate_next_tokens: list[str], current_output: str) -> list[str]:
    """Illustrates the CORE IDEA: filter out tokens that would make the output invalid,
    at EVERY generation step -- the real mechanism (simplified from actual grammar-constrained decoding)."""
    valid_next_tokens = []
    for token in candidate_next_tokens:
        candidate = current_output + token
        if is_valid_partial_json_prefix(candidate):
            valid_next_tokens.append(token)
    return valid_next_tokens

# Illustrative: at some point mid-generation, filter candidate next tokens
current = '{"name": "get_mortality_rate", "arguments": {"age": 55'
candidates = [', "smoker": true}}', ' this is not valid json', ']']
valid = simulate_constrained_generation(candidates, current)
print("Valid next-token continuations:", valid)
```
This is a deliberately simplified illustration (real constrained decoding operates at the actual tokenizer/logit level, Phase 6 Lessons 1 and 4, integrating a full grammar parser into the sampling loop, not this toy string-based heuristic) — but it captures the essential mechanism: at every generation step, the space of allowed next tokens is filtered to only those consistent with eventually producing valid, schema-conforming output.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `build_tool_schema`/`ToolExecutor` | Native SDK support in every major LLM provider's API (Anthropic, OpenAI, others) — handles the full request/response cycle, parallel tool calls, and forced-tool-choice configuration |
| `simulate_constrained_generation` (toy) | Real constrained decoding implementations (e.g., outlines, guidance libraries, or providers' built-in structured-output modes) operating directly on the model's token logits during actual generation |
| Manual conversation history management | Most SDKs provide conversation/message-list abstractions handling the tool-call-then-tool-result round-trip pattern automatically |

---

## 8. Visual Explanations

**The tool-calling request cycle:**
```
Conversation + Tool Schemas ──▶ LLM
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                             ▼
            Text response                 Structured tool call(s)
            (no tool needed)              (name + arguments, schema-validated)
                    │                             │
              RETURN to user            EXECUTE the actual function
                                                    │
                                          Tool result appended to conversation
                                                    │
                                          SEND BACK to the LLM (loop continues, Lesson 3)
```

**Constrained decoding: filtering invalid tokens at each generation step:**
```
Partial output: '{"name": "get_mortality_rate", "arguments": {"age": 55'
                                                                        │
                              Candidate next tokens: [', "smoker":...', 'xyz garbage', ']']
                                                                        │
                     Schema-aware filter REJECTS invalid continuations, keeps only:
                                                              [', "smoker": true}}']
                                          (guarantees syntactic + schema validity BY CONSTRUCTION)
```

---

## 9. Practical Examples

**Simple:** design a well-documented tool schema (Section 5) for a simple utility function and verify a real LLM API correctly selects and calls it given an appropriate user request.
**Medium:** implement `ToolExecutor` with 2-3 registered tools and test the full request/execute/respond cycle (Section 4) using a real LLM API's native tool-calling feature.
**Real-world:** design and test tool schemas for an actuarial calculation toolkit (mortality lookup, present-value calculation, unit conversion), systematically testing tool-selection accuracy across a range of realistic user requests, and refining schema descriptions based on observed selection errors.

---

## 10. Real Industry Use Cases

- **Every production LLM agent/assistant with tool access** (Lesson 3): relies on native tool calling, not free-text parsing, for reliability.
- **MCP's tool-invocation layer** (Lesson 4): built directly on top of each underlying LLM provider's native tool-calling capability.
- **Customer support and business-process automation**: tool-calling connects LLMs to CRM systems, order databases, and internal APIs for real business actions (not just information retrieval).
- **Code execution and data analysis assistants**: tool calling is how an LLM triggers actual code execution, data queries, or file operations rather than merely describing what code *would* do.

---

## 11. Common Mistakes

- Writing vague or ambiguous tool descriptions ("does stuff with data") — directly degrades tool-selection accuracy; specific, example-illustrated descriptions measurably improve reliability.
- Exposing too many tools (or overly similar/overlapping tools) to the model simultaneously — increases selection errors; curate the tool set relevant to the current task context rather than exposing an entire tool library indiscriminately.
- Not validating/sanitizing tool arguments before executing the underlying function — even with schema validation at the API level, defensive input handling in the actual tool implementation remains good practice (never blindly trust and `eval()` arbitrary model-generated input).
- Forgetting that the model never executes anything itself — the calling application is always responsible for actual execution, and must handle errors, timeouts, and security boundaries around that execution.

---

## 12. Best Practices (2026)

- Invest real effort in tool schema design (clear names, detailed descriptions, well-typed and well-described parameters) — this is a genuinely high-leverage, underrated skill directly affecting agent reliability.
- Curate the set of tools exposed to the model per-task/context rather than always exposing every available tool.
- Use forced tool-choice configuration when your application logic requires a guaranteed specific tool call (e.g., always requiring structured output via a "format_response" tool, Lesson 6).
- Implement robust error handling and input validation in actual tool execution code, treating model-generated arguments as untrusted input requiring the same defensive practices as any external user input.

---

## 13. Exercises

**Easy:** Design a tool schema for a simple unit-conversion function and manually verify it matches the JSON-Schema format expected by a real LLM API.
**Medium:** Implement the full tool-calling request cycle (Section 5) with a real LLM API and 2 registered tools, testing on requests that should and shouldn't trigger tool use.
**Hard:** Systematically test tool-selection accuracy as you vary schema description quality (vague vs. detailed) across a set of realistic requests, quantifying the accuracy difference.
**Mathematical:** Given an application exposing $n$ tools with an empirically-measured per-tool selection accuracy that degrades as $n$ grows, sketch a simple model (e.g., linear or logarithmic degradation) and discuss how you'd decide the maximum number of tools to expose simultaneously for your application.
**Coding:** Implement parallel tool-call handling (executing multiple requested tool calls concurrently, directly reusing Phase 1 Lesson 2's `asyncio`/concurrency concepts) and measure the latency improvement versus sequential execution for a task requiring 3 independent tool calls.

---

## 14. Mini Project

Build a **well-designed actuarial tool-calling toolkit**: design clear, detailed schemas for 4-5 tools relevant to your domain (mortality/lapse rate lookups, present-value calculations, unit conversions, a RAG-based document search tool from Lesson 1), implement robust input validation and error handling in each tool's execution code, integrate them with a real LLM API's native tool-calling feature, and systematically test tool-selection accuracy and correct-argument-construction across a diverse set of realistic actuarial questions — iterating on schema design based on observed failures.

---

## 15. Interview Preparation

- Explain the full tool-calling request/response cycle and why the model never executes tools itself.
- Why does tool schema quality (naming, descriptions) directly affect tool-selection reliability?
- What is constrained decoding, and how does it guarantee schema-valid structured output?
- What are the tradeoffs of exposing many tools simultaneously to an LLM agent?

---

## 16. Summary

Native tool calling replaces fragile free-text action parsing (Lesson 3's simplified illustration) with schema-validated, structurally reliable function-call generation, implemented under the hood via constrained decoding that filters the model's token sampling (Phase 5 Lesson 7) to guarantee syntactic and schema validity by construction. Tool schema design — clear names, detailed descriptions, well-specified parameters — is a genuinely high-leverage practical skill directly determining agent reliability, and understanding the full request/execute/respond cycle (with the calling application, never the model, responsible for actual execution) is essential for building the reliable agentic systems (Lesson 3), MCP-connected tools (Lesson 4), and structured-output pipelines (Lesson 6) that depend on it throughout the rest of this phase.

---

## 17. References

- OpenAI — "Function calling and other API updates" (June 2023, the announcement that popularized native tool calling)
- Anthropic — Tool use documentation (docs.anthropic.com)
- Willard & Louf — "Efficient Guided Generation for Large Language Models" (2023, on constrained decoding techniques, e.g. the `outlines` library)
- JSON Schema specification (json-schema.org) — the standard format underlying most tool/function schemas
