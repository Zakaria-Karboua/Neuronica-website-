# Phase 7 · Lesson 4 — MCP (Model Context Protocol)

> Prerequisite: AI Agents (Lesson 3), Phase 1 Lesson 7 (SQL, as an analogy for standardized interfaces)

---

## 1. Introduction

### What is MCP?
The Model Context Protocol is an open standard (introduced by Anthropic in late 2024) defining a uniform way for LLM applications to connect to external data sources and tools — a standardized client-server protocol so that any MCP-compatible tool/data-source ("server") can be used by any MCP-compatible LLM application ("client") without custom, bespoke integration code for every single pairing.

### Why does it exist?
Before MCP, connecting an LLM application to $N$ different tools/data sources (a database, a file system, a calendar, a search engine) required $N$ separate, custom integrations, and if you wanted to use those same $N$ tools from $M$ different LLM applications, you potentially needed $N \times M$ integrations — a combinatorial integration problem. MCP standardizes the interface, converting this into $N + M$ integrations (each tool implements the protocol once as a server; each application implements the protocol once as a client), directly analogous to how a standardized plug (USB) lets any compatible device work with any compatible port, rather than requiring a custom cable for every device-computer pairing.

### Historical background
MCP was released by Anthropic in November 2024 as an open specification (not a proprietary, closed protocol), rapidly gaining adoption across the LLM tooling ecosystem through 2025-2026 as both a practical integration standard and, notably, a rare case of a major AI lab open-sourcing foundational infrastructure specifically to grow a shared ecosystem rather than a walled garden.

### Real-world motivation
When you connect Claude (or another MCP-compatible application) to Google Drive, Slack, or a database via a "connector," you are using MCP under the hood — this lesson explains the actual protocol mechanics behind that now-common integration pattern.

---

## 2. Theory

### The client-server architecture
- **MCP Server**: exposes a specific tool/data source's capabilities (e.g., "read files from this file system," "query this database," "search this API") through the standardized protocol — implemented once per tool/data-source, regardless of which LLM applications will consume it.
- **MCP Client**: the LLM application (or an intermediary within it) that connects to one or more MCP servers, discovers their available capabilities, and invokes them on the model's behalf during an agentic interaction (Lesson 3's tool-calling, now standardized).
- **Host**: the overall application (e.g., a chat interface) that manages one or more client connections to different servers.

### The three core primitives MCP standardizes
- **Tools**: executable functions the model can invoke (directly Lesson 5's tool-calling mechanism, now with a standard discovery/invocation protocol rather than an application-specific one).
- **Resources**: read-only data the application can provide as context (e.g., file contents, database query results) — conceptually similar to what gets retrieved and injected into context in a RAG system (Lesson 1), but through a standardized access pattern.
- **Prompts**: reusable, server-defined prompt templates that a client can request and populate — allowing tool/data-source providers to also ship recommended interaction patterns for their specific domain.

### Discovery and capability negotiation
When an MCP client connects to a server, it first performs a **discovery** handshake — the server advertises its available tools, resources, and prompts (with structured descriptions, directly feeding Lesson 5's structured tool-calling schemas), and the client/host application can then decide which of these to expose to the LLM for a given task, without needing any hardcoded, pre-existing knowledge of that specific server's capabilities.

### Transport-agnostic design
MCP is defined independently of the specific communication transport used to carry its messages — implementations commonly use standard I/O (for local tools/processes) or HTTP/SSE (Server-Sent Events, for remote/networked servers) — a deliberate architectural choice allowing the same protocol semantics to work whether a tool runs as a local subprocess or a remote networked service.

---

## 3. Mathematical Foundations

MCP is fundamentally a systems/protocol design topic rather than a mathematically dense one, but a few formal framings are worth making explicit:

### The integration complexity reduction, quantified
Without a standard protocol, connecting $N$ tools to $M$ applications requires (in the worst case, with no shared abstractions at all) $O(N \times M)$ bespoke integrations. With a shared protocol, this becomes $O(N + M)$ — each tool implements the server side once, each application implements the client side once. For even modest $N, M$ (say, 20 tools and 10 applications), this is the difference between $200$ integrations and $30$ — a genuinely large, quantifiable reduction directly analogous to why standardized interfaces (SQL as a standard query interface across databases, Phase 1 Lesson 7; POSIX as a standard OS interface, Phase 1 Lesson 6) have historically driven ecosystem growth.

### Capability discovery as schema matching
When a client discovers a server's available tools, each tool is described via a structured schema (directly Lesson 5's structured-output/function-calling schema format) — the client (or the LLM itself, given the schema in context) must match the current task's needs against the available tools' declared capabilities, conceptually a constraint-satisfaction/matching problem, though in practice handled by the LLM's own reasoning (Lesson 3's agentic loop) rather than a separate formal matching algorithm.

### Protocol versioning and compatibility
As with any standardized protocol (directly analogous to API versioning concerns in Phase 8's software engineering context), MCP must handle capability negotiation gracefully across client/server implementations that may support different protocol versions — a client should be able to work with a server implementing an older or newer compatible version, degrading gracefully rather than failing outright when encountering capabilities it doesn't recognize.

---

## 4. Algorithm — The MCP Connection and Tool-Use Lifecycle (fully specified)

```
1. CONNECTION: Host application establishes a connection to an MCP server
   (via stdio for a local process, or HTTP/SSE for a remote server)

2. INITIALIZATION HANDSHAKE:
     client sends: protocol version, client capabilities
     server responds: protocol version (negotiated), server capabilities (tools/resources/prompts available)

3. DISCOVERY:
     client requests: "list available tools" (and/or resources, prompts)
     server responds: structured list of tool schemas (name, description, input parameters -- Lesson 5's format)

4. DURING AN AGENTIC TASK (Lesson 3's ReAct loop, now using MCP-discovered tools):
     the LLM (given the discovered tool schemas in its context) decides to invoke a tool
     client sends: a "call tool" request with the tool name + arguments
     server EXECUTES the underlying action (e.g., queries a database, reads a file)
     server responds: the tool's result (or a structured error)
     client incorporates the result as an OBSERVATION in the agent's ongoing transcript

5. TEARDOWN: connection closed when the session/task ends
```

---

## 5. Python Implementation

```python
"""mcp_core.py — a simplified, illustrative MCP-style server and client
(the REAL protocol has a formal JSON-RPC-based spec; this captures the conceptual shape)"""
import json
from typing import Callable


class MCPServer:
    """A minimal illustrative MCP server exposing a set of tools via discovery + invocation."""
    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, dict] = {}

    def register_tool(self, name: str, func: Callable, description: str, parameters_schema: dict) -> None:
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters_schema": parameters_schema,   # directly Lesson 5's structured schema format
        }

    def handle_discovery_request(self) -> dict:
        """Returns the structured tool list a client would request during Section 4's discovery step."""
        return {
            "server_name": self.name,
            "tools": [
                {"name": name, "description": t["description"], "parameters": t["parameters_schema"]}
                for name, t in self.tools.items()
            ],
        }

    def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            result = self.tools[tool_name]["func"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}


class MCPClient:
    """A minimal illustrative client connecting to one or more MCP servers."""
    def __init__(self):
        self.connected_servers: dict[str, MCPServer] = {}
        self.discovered_tools: dict[str, tuple[str, dict]] = {}   # tool_name -> (server_name, schema)

    def connect(self, server: MCPServer) -> None:
        self.connected_servers[server.name] = server
        discovery = server.handle_discovery_request()
        for tool in discovery["tools"]:
            self.discovered_tools[tool["name"]] = (server.name, tool)

    def get_all_tool_schemas_for_llm(self) -> list[dict]:
        """This is what gets included in the LLM's context so it knows what tools are available."""
        return [schema for _, schema in self.discovered_tools.values()]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name not in self.discovered_tools:
            return {"error": f"Tool {tool_name} not discovered from any connected server"}
        server_name, _ = self.discovered_tools[tool_name]
        return self.connected_servers[server_name].handle_tool_call(tool_name, arguments)


# Example: a "database" server and a "calculator" server, both connected to one client
db_server = MCPServer("actuarial-db")
db_server.register_tool(
    "get_mortality_rate", lambda age: 0.001 * (1.08 ** (age - 20)),
    "Returns the mortality rate for a given age.", {"age": {"type": "integer"}}
)

calc_server = MCPServer("calculator")
calc_server.register_tool(
    "multiply", lambda a, b: a * b, "Multiplies two numbers.",
    {"a": {"type": "number"}, "b": {"type": "number"}}
)

client = MCPClient()
client.connect(db_server)
client.connect(calc_server)

print("Discovered tool schemas (would go into the LLM's context):")
print(json.dumps(client.get_all_tool_schemas_for_llm(), indent=2))

result = client.invoke_tool("get_mortality_rate", {"age": 65})
print("\nTool call result:", result)
```

---

## 6. Build From Scratch

Section 5 already provides a genuine, if simplified, from-scratch MCP client/server implementation, capturing the protocol's essential shape (discovery, schema exposure, invocation, structured error handling). The valuable additional exercise is integrating this with Lesson 3's agent loop directly:

```python
def run_mcp_aware_agent(task: str, llm_call, mcp_client: "MCPClient", max_iterations: int = 6) -> str:
    """Lesson 3's ReAct loop, now sourcing its tools from MCP-DISCOVERED servers rather than
    a hardcoded TOOLS dict -- the key architectural shift MCP enables."""
    tool_schemas = mcp_client.get_all_tool_schemas_for_llm()
    tool_descriptions = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_schemas)
    transcript = [f"Task: {task}"]

    for _ in range(max_iterations):
        prompt = (
            f"Available tools (discovered via MCP):\n{tool_descriptions}\n\n"
            f"Use 'Action: tool_name[arguments as JSON]' or 'Final Answer: ...'.\n\n"
            + "\n".join(transcript)
        )
        llm_output = llm_call(prompt)

        if "Final Answer:" in llm_output:
            return llm_output.split("Final Answer:")[1].strip()

        # (simplified parsing for illustration -- a real implementation uses Lesson 5's
        #  structured function-calling APIs rather than text parsing)
        transcript.append(f"Thought/Action: {llm_output}")
        # ... parse tool_name/arguments, call mcp_client.invoke_tool(...), append observation ...

    return "Agent did not complete within the iteration limit."
```
The key architectural point this makes concrete: the agent loop's *logic* (Lesson 3) doesn't need to change at all when switching which specific tools are available — it simply asks its connected MCP client what's available and uses whatever comes back, exactly the decoupling MCP is designed to provide.

---

## 7. Library/Tool Comparison

| From scratch (illustrative) | Production tooling |
|---|---|
| `MCPServer`/`MCPClient` (simplified) | Official MCP SDKs (Python, TypeScript) implementing the full JSON-RPC-based specification, including proper error codes, streaming, and authentication |
| Manual discovery/invocation methods | MCP's formal specification defines exact message formats, lifecycle states, and capability negotiation details beyond this lesson's simplified illustration |
| No transport layer shown | Real MCP implementations support stdio (local processes) and HTTP/SSE (remote servers) transports, with proper connection lifecycle management |

---

## 8. Visual Explanations

**Without MCP: N×M bespoke integrations:**
```
App A ──custom──▶ Tool 1        App A ──custom──▶ Tool 2       ... (N x M integrations)
App B ──custom──▶ Tool 1        App B ──custom──▶ Tool 2       ...
App C ──custom──▶ Tool 1        App C ──custom──▶ Tool 2       ...
```

**With MCP: N+M standardized integrations:**
```
App A ──┐                                    ┌── Tool 1 (MCP server)
App B ──┼──▶ [Standard MCP Protocol] ◀───────┼── Tool 2 (MCP server)
App C ──┘                                    └── Tool 3 (MCP server)
  (each app implements the CLIENT side ONCE; each tool implements the SERVER side ONCE)
```

**MCP connection lifecycle (Section 4):**
```
Connect ──▶ Handshake (versions/capabilities) ──▶ Discovery (list tools/resources/prompts)
                                                          │
                                                          ▼
                                          [Agentic loop uses discovered tools, Lesson 3]
                                                          │
                                                          ▼
                                                     Teardown
```

---

## 9. Practical Examples

**Simple:** implement the `MCPServer`/`MCPClient` classes (Section 5) with two simple tools and verify discovery correctly returns both tools' schemas.
**Medium:** connect the Section 6 MCP-aware agent to your `MCPServer` implementation and verify it can complete a task requiring tools from two different (simulated) servers.
**Real-world:** explore the official MCP documentation and identify (or set up) a real existing MCP server (e.g., a file system or database connector) and connect a real MCP client application to it, observing the actual discovery/invocation protocol in action via the official SDK.

---

## 10. Real Industry Use Cases

- **Claude's connector ecosystem**: Google Drive, Slack, and many other third-party integrations available in Claude products are implemented as MCP servers, directly the mechanism referenced in this curriculum's own tool-search/connector-suggestion behavior.
- **Cross-application tool reuse**: any tool implemented as an MCP server can, in principle, be used by any MCP-compatible client application (Claude, other AI assistants, custom-built agent systems) without bespoke per-application integration work.
- **Enterprise internal tooling**: companies increasingly expose internal systems (databases, ticketing systems, internal APIs) as MCP servers specifically so that any current or future MCP-compatible AI application can use them, future-proofing integration investment.
- **The broader MCP ecosystem**: since its 2024 release, a rapidly growing community-maintained registry of MCP servers has emerged, covering everything from popular SaaS tools to specialized developer utilities.

---

## 11. Common Mistakes

- Building bespoke, application-specific tool integrations when an MCP server for that tool/data-source already exists or could reasonably be built once and reused — recreating exactly the N×M integration problem MCP exists to solve.
- Assuming MCP handles authentication/authorization automatically — in practice, security/access-control considerations remain the implementer's responsibility, layered on top of MCP's core protocol semantics.
- Not implementing graceful error handling in an MCP server's tool execution — an unhandled exception should become a structured error observation for the calling agent (Lesson 3's error-recovery point), not a silent failure or an unhandled server crash.
- Overlooking protocol version compatibility when building either a client or server intended for broad ecosystem use — a real, practical consideration as the protocol itself evolves over time.

---

## 12. Best Practices (2026)

- Check whether an MCP server already exists for a tool/data-source you need before building a custom integration — the growing ecosystem increasingly covers common use cases.
- When building a new tool/data-source integration intended for reuse across multiple applications, implement it as an MCP server rather than a bespoke, single-application integration.
- Design tool schemas (Lesson 5) with clear, LLM-friendly descriptions and well-specified parameter types — the quality of an MCP server's schema descriptions directly affects how reliably a connected LLM agent can use it correctly.
- Follow the official MCP specification and SDKs for any real implementation, rather than an ad hoc reimplementation, to ensure compatibility with the broader ecosystem of clients/servers.

---

## 13. Exercises

**Easy:** Implement `MCPServer`/`MCPClient` (Section 5) with a single tool and manually trace through the discovery-then-invocation lifecycle.
**Medium:** Add a second server with two additional tools, connect both servers to one client, and verify the client's `get_all_tool_schemas_for_llm()` correctly aggregates tools from both.
**Hard:** Extend Section 5's implementation with basic protocol version negotiation (client and server each declare a supported version list; connection succeeds only if there's overlap) and test it with intentionally mismatched version lists.
**Mathematical:** Given $N$ tools and $M$ applications, compute the integration count under the bespoke ($O(NM)$) versus standardized-protocol ($O(N+M)$) approaches for a few realistic $(N,M)$ pairs, and plot how the gap grows as both increase.
**Coding:** Integrate the Section 6 MCP-aware agent with a real LLM API (rather than a mocked `llm_call`) and test it end-to-end on a task requiring tools from your Section 5 `MCPServer` implementations.

---

## 14. Mini Project

Build a **complete MCP server for an actuarial/SANAVIR use case**: implement an MCP server exposing tools relevant to your domain (e.g., mortality rate lookups, unit conversions, or SANAVIR sensor-data queries), with well-designed schemas and clear descriptions; connect it to the Lesson 3 ReAct agent (adapted per Section 6 to use MCP-based tool discovery instead of a hardcoded tool dictionary); test the resulting agent on realistic multi-step tasks; and write a short reflection on how this architecture would let the same server be reused by a completely different AI application without any changes to the server itself.

---

## 15. Interview Preparation

- Explain the integration-complexity problem MCP solves, and quantify the difference between bespoke and standardized integration approaches.
- What are the three core primitives MCP standardizes (tools, resources, prompts), and what does each provide?
- Describe the MCP connection lifecycle from initial handshake through tool invocation.
- Why is MCP being an open, transport-agnostic standard (rather than a proprietary, closed protocol) significant for ecosystem growth?

---

## 16. Summary

MCP standardizes how LLM applications connect to external tools and data sources, converting a combinatorial $N \times M$ bespoke-integration problem into a linear $N + M$ standardized one — directly analogous to how SQL (Phase 1 Lesson 7) or POSIX (Phase 1 Lesson 6) standardized interfaces have historically driven ecosystem growth by decoupling implementers on either side of an interface. Built around a client-server architecture with discovery, tool/resource/prompt primitives, and a transport-agnostic design, MCP is precisely the protocol underlying the "connectors" pattern now common in production AI applications, and directly extends Lesson 3's agent loop by decoupling *which tools exist* from *how the agent reasons about using them* — a foundational piece of infrastructure for the more sophisticated agentic workflows (Lesson 7) and multi-agent systems (Lesson 8) covered next in this phase.

---

## 17. References

- Anthropic — "Introducing the Model Context Protocol" (November 2024, the official announcement)
- Model Context Protocol official specification (modelcontextprotocol.io)
- Official MCP SDK documentation (Python and TypeScript implementations)
- Anthropic — "Building Effective Agents" (engineering blog, relevant context on how MCP fits into broader agent architecture)
