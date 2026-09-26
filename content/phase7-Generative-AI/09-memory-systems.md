# Phase 7 · Lesson 9 — Memory Systems

> Prerequisite: RAG, AI Agents, Multi-Agent Systems (Lessons 1, 3, 8)

---

## 1. Introduction

### What are memory systems?
Mechanisms for an LLM application to retain and use information across a single long interaction (short-term/working memory) or across entirely separate sessions over time (long-term/persistent memory) — addressing the fundamental fact that an LLM itself is **stateless**: every API call is independent, with no inherent memory of past interactions unless the calling application explicitly reconstructs and provides that context.

### Why does it exist?
A genuinely useful assistant needs to remember what was discussed earlier in a long conversation (working memory), recall relevant facts from previous, separate conversations (long-term memory), and manage the practical reality that both the LLM's context window (finite) and the cost of including more tokens (Phase 6 Lesson 1) impose hard constraints on how much can simply be "included every time." Memory systems exist to engineer around these constraints intelligently.

### Historical background
Early chatbots (pre-Transformer) often used explicit, hand-coded slot-filling memory (tracking specific structured facts like "user's name"). Modern LLM memory systems (2023-2026) have converged on a hybrid approach: recent conversation history kept verbatim (or lightly summarized) in-context, combined with a RAG-style (Lesson 1) retrieval system over a persistent store of extracted facts/summaries for anything needing to survive beyond the current context window or session.

### Real-world motivation
This curriculum's own memory-filesystem feature (the one governing how this very assistant retains information about you across conversations) is a direct, practical instance of the long-term memory architecture this lesson covers — understanding the underlying design patterns demystifies exactly how such systems work and their genuine tradeoffs.

---

## 2. Theory

### Working memory (within a single session/context window)
The straightforward approach — simply include the full conversation transcript in each new request's context — works until the transcript grows too large for the context window (Phase 6 Lesson 4's context-length constraints) or until cost (Phase 6 Lesson 1's token-based pricing) becomes prohibitive for very long conversations. This directly motivates **summarization** (periodically compressing older parts of the conversation into a shorter summary) and **truncation/sliding-window** (keeping only the most recent $N$ turns, discarding older ones) strategies.

### Long-term memory (across sessions)
Persisting information beyond a single session requires an explicit storage and retrieval mechanism — typically: (a) extracting salient facts/summaries from a conversation (often via a dedicated LLM call specifically for this extraction task), (b) storing them (in a structured database, a vector store for semantic retrieval, or both), and (c) retrieving relevant stored memories at the start of (or during) a new session, injecting them into context much like Lesson 1's RAG retrieval — indeed, long-term memory retrieval is architecturally almost identical to RAG, just with the "documents" being past interaction facts rather than external reference documents.

### Memory extraction — deciding what's worth remembering
Not everything said in a conversation is worth persisting — a genuinely important design question is what counts as a durable, useful fact (stable preferences, ongoing projects, key relationships) versus ephemeral, session-specific content (a one-off question, a transient task detail) not worth the storage/retrieval overhead or the risk of surfacing stale/irrelevant information in a future session.

### Memory consistency and updates
Facts can change over time (a user's job, an ongoing project's status) — a robust long-term memory system needs an update/versioning strategy (does new information overwrite old, or get appended alongside it with a timestamp for later reconciliation?) rather than naively accumulating potentially contradictory facts indefinitely.

---

## 3. Mathematical Foundations

### Context window budget allocation across memory tiers
Given a fixed context window $C$, an application must allocate tokens across: system prompt, retrieved long-term memories, recent conversation history, and the current query — directly Phase 6 Lesson 1's tokenization-cost accounting, now applied as an explicit budget-allocation problem:

$$
\text{system\_prompt} + \text{retrieved\_memories} + \text{recent\_history} + \text{query} \le C
$$

As conversation length grows, the "recent history" allocation must eventually be traded off (via summarization or truncation) to make room, a genuinely practical engineering constraint with no way around it besides intelligent compression.

### Summarization as lossy compression (directly reusing Phase 3 Lesson 6's information theory)
Summarizing older conversation history is, formally, lossy compression — information is discarded, and the summarization LLM call must decide (implicitly, via its own learned judgment) what to preserve versus discard. Unlike Phase 6 Lesson 1's tokenizer-level compression (which is lossless at the token level), conversation summarization is lossy at the *semantic* level — some information is genuinely, irrecoverably lost, a real tradeoff (shorter context, cheaper, but potentially missing a detail that later turns out to matter) rather than a free efficiency gain.

### Retrieval relevance decay over time
For long-term memory retrieval, relevance often should incorporate **recency**, not just semantic similarity (Lesson 1/2's cosine similarity) — a common approach blends similarity score with a time-decay factor:

$$
\text{score} = \text{cos\_sim}(q, m) \times e^{-\lambda \cdot \Delta t}
$$

where $\Delta t$ is the time elapsed since the memory was created/last confirmed relevant, and $\lambda$ controls how quickly older memories are discounted — a design choice directly analogous to Phase 4 Lesson 6's time-series concepts (recent observations weighted more heavily), now applied to memory relevance rather than forecasting.

### The forgetting curve as a design inspiration (loosely analogous, not literally implemented)
Human memory research's "forgetting curve" (information not reinforced decays in recall probability over time) is sometimes used as loose inspiration for memory systems that deliberately let rarely-accessed, unconfirmed memories fade in retrieval priority over time (via the decay term above) rather than persisting every stored fact with equal weight indefinitely — a design analogy worth knowing, though it should be understood as inspiration rather than a claim that LLM memory systems literally model human cognitive processes.

---

## 4. Algorithm — A Hybrid Working + Long-Term Memory System (fully specified)

```
AT THE START OF A NEW SESSION:
1. RETRIEVE relevant long-term memories: embed a representation of the session's likely topic
   (or simply retrieve the most recent/most relevant stored facts about this user/context)
   using Lesson 1/2's RAG retrieval machinery against the persistent memory store
2. INJECT retrieved memories into the system prompt / initial context

DURING THE SESSION (per turn):
3. APPEND the new turn to the working (in-context) conversation history
4. IF the accumulated history approaches the context budget (Section 3):
     SUMMARIZE the oldest portion of the history into a compact summary (a dedicated LLM call)
     REPLACE that oldest portion with the summary in the ongoing context
5. GENERATE the response using: system prompt + retrieved long-term memories + (summarized +
   recent) working history + current query

AT THE END OF THE SESSION (or periodically during a long one):
6. EXTRACT salient, durable facts from the session (a dedicated LLM call, Lesson 6's structured
   output format, specifically distinguishing durable facts from ephemeral session-specific content)
7. STORE these facts in the persistent long-term memory store (with a timestamp, for Section 3's decay)
8. (OPTIONAL) RECONCILE: check whether newly extracted facts conflict with/update existing stored
   facts, applying an update/versioning strategy rather than naive indefinite accumulation
```

---

## 5. Python Implementation

```python
"""memory_systems_core.py — working memory management + long-term memory storage/retrieval"""
import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    content: str
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)


class LongTermMemoryStore:
    """A simplified persistent memory store combining Lesson 1/2's RAG retrieval with time-decay (Section 3)."""
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn      # a real sentence-embedding model (Phase 6 Lesson 2)
        self.entries: list[MemoryEntry] = []

    def store(self, fact: str) -> None:
        embedding = self.embed_fn(fact)
        self.entries.append(MemoryEntry(content=fact, embedding=embedding))

    def retrieve(self, query: str, top_k: int = 5, decay_lambda: float = 1e-7) -> list[str]:
        if not self.entries:
            return []
        query_embedding = self.embed_fn(query)
        now = time.time()
        scored = []
        for entry in self.entries:
            similarity = np.dot(query_embedding, entry.embedding)          # assumes normalized embeddings
            time_decay = np.exp(-decay_lambda * (now - entry.timestamp))    # Section 3's decay formula
            scored.append((entry.content, similarity * time_decay))
        scored.sort(key=lambda x: -x[1])
        return [content for content, _ in scored[:top_k]]


class WorkingMemoryManager:
    """Manages in-context conversation history with summarization when approaching a token budget."""
    def __init__(self, summarize_fn, max_history_tokens: int = 2000, estimate_tokens_fn=len):
        self.summarize_fn = summarize_fn
        self.max_history_tokens = max_history_tokens
        self.estimate_tokens_fn = estimate_tokens_fn   # in practice: a real tokenizer, Phase 6 Lesson 1
        self.summary: str = ""
        self.recent_turns: list[str] = []

    def add_turn(self, turn_text: str) -> None:
        self.recent_turns.append(turn_text)
        total_tokens = self.estimate_tokens_fn(self.summary) + sum(self.estimate_tokens_fn(t) for t in self.recent_turns)
        if total_tokens > self.max_history_tokens and len(self.recent_turns) > 2:
            # Summarize everything EXCEPT the most recent 2 turns (keep immediate context verbatim)
            to_summarize = self.recent_turns[:-2]
            self.summary = self.summarize_fn(self.summary, to_summarize)
            self.recent_turns = self.recent_turns[-2:]

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary]: {self.summary}")
        parts.extend(self.recent_turns)
        return "\n".join(parts)


def extract_durable_facts(conversation_text: str, extract_fn) -> list[str]:
    """A dedicated extraction call distinguishing durable facts from ephemeral content (Section 2)."""
    return extract_fn(conversation_text)   # in practice: a structured-output call (Lesson 6)


# Illustrative usage (mocked functions standing in for real embedding/LLM calls)
mock_embed = lambda text: np.random.default_rng(hash(text) % (2**32)).normal(size=16) / 4   # deterministic mock
long_term_memory = LongTermMemoryStore(embed_fn=mock_embed)
long_term_memory.store("User is a 4th-year actuarial science student.")
long_term_memory.store("User is building a mortality prediction model with XGBoost.")

retrieved = long_term_memory.retrieve("Tell me about the user's ML background")
print("Retrieved long-term memories:", retrieved)
```

---

## 6. Build From Scratch

**A minimal recursive summarization function (making Section 3's lossy-compression tradeoff concrete):**
```python
def recursive_summarize(existing_summary: str, new_turns: list[str], summarize_llm_call) -> str:
    """Combines an existing summary with new content into an UPDATED, still-compact summary --
    avoiding the common bug of just concatenating summaries indefinitely (which defeats the purpose)."""
    combined_input = (
        f"Existing summary: {existing_summary}\n\n"
        f"New conversation content to incorporate:\n" + "\n".join(new_turns) +
        "\n\nProduce an UPDATED summary that incorporates the new content, "
        "staying concise and preserving only durable, important information."
    )
    return summarize_llm_call(combined_input)

# Illustrative mock demonstrating the RECURSIVE nature (each call's output feeds the NEXT call's input)
mock_summarize = lambda text: f"[summary of: {text[:50]}...]"
summary_v1 = recursive_summarize("", ["Turn 1: discussed X", "Turn 2: discussed Y"], mock_summarize)
summary_v2 = recursive_summarize(summary_v1, ["Turn 3: discussed Z"], mock_summarize)
print(summary_v2)
```
The key correctness property this makes explicit: summarization must be **recursive/updating** (each new summary incorporates the previous summary plus new content, staying roughly constant in size) rather than **additive** (naively appending new summaries to old ones indefinitely, which would eventually recreate the exact unbounded-growth problem summarization was meant to solve).

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `LongTermMemoryStore` | A real vector database (Lesson 2) combined with a metadata store for timestamps/decay — production memory frameworks (e.g., Mem0, or custom implementations on top of Pinecone/Weaviate) provide this out of the box |
| `WorkingMemoryManager` | LangChain's/LangGraph's built-in conversation memory abstractions (`ConversationSummaryMemory` and similar), or custom implementations following the same summarization-on-threshold pattern |
| `recursive_summarize` | Same core technique used in production; some frameworks provide built-in recursive/hierarchical summarization utilities for very long documents/conversations |

---

## 8. Visual Explanations

**Working memory: summarization triggered as history grows toward the token budget:**
```
Turn 1, Turn 2, Turn 3, ..., Turn 20   (accumulating, approaching budget)
         │
         ▼ (threshold reached)
[Summary of Turns 1-18] + Turn 19 + Turn 20     (compressed, room made for new turns)
         │
         ▼ (more turns accumulate...)
[UPDATED summary incorporating Turns 1-18 + 19-25] + Turn 26 + Turn 27
   (RECURSIVE: each summary absorbs the previous one, NOT naive concatenation)
```

**Long-term memory retrieval with time-decay (Section 3):**
```
Memory A: "likes hiking" (stored 200 days ago)         -> similarity 0.7, decayed score: LOW
Memory B: "working on XGBoost mortality model" (stored 2 days ago) -> similarity 0.6, decayed score: HIGHER
   (a SLIGHTLY less semantically similar but MUCH more recent memory can outrank an older one)
```

---

## 9. Practical Examples

**Simple:** implement `WorkingMemoryManager` (Section 5) and simulate a 30-turn conversation, observing when summarization triggers.
**Medium:** implement `LongTermMemoryStore` with time-decay retrieval and verify that a recent, moderately-relevant memory can outrank an old, highly-relevant one at an appropriately tuned decay rate.
**Real-world:** design a memory extraction prompt (Lesson 6's structured output) that reliably distinguishes durable facts (stable preferences, ongoing projects) from ephemeral conversation content, testing it across several realistic conversation transcripts and manually assessing extraction quality.

---

## 10. Real Industry Use Cases

- **This very assistant's memory system**: the persistent memory-filesystem feature governing how information about you is retained across conversations is a direct, practical instance of exactly this lesson's long-term memory architecture (extraction, storage, retrieval-with-relevance-ranking).
- **Customer support chatbots**: retain customer history/preferences across sessions specifically to avoid asking users to repeat information already provided in past interactions.
- **Personal AI assistants**: increasingly market "remembering you" as a core feature, built on exactly this lesson's hybrid working-memory + long-term-memory architecture.
- **Long-running coding/research agents** (Lesson 3, 8): use working-memory summarization extensively to manage very long task transcripts that would otherwise exceed context windows during extended, multi-hour agentic sessions.

---

## 11. Common Mistakes

- Naively concatenating conversation history indefinitely without any summarization/truncation strategy — inevitably exceeds the context window or becomes prohibitively expensive on sufficiently long conversations.
- Storing every single conversational detail in long-term memory without any durability filtering — bloats the memory store with irrelevant, ephemeral content that dilutes retrieval quality for genuinely important stored facts.
- Ignoring recency in long-term memory retrieval, relying purely on semantic similarity — can surface outdated, since-superseded facts as if they were still current.
- Using purely additive (non-recursive) summarization — summaries grow without bound across a long conversation, defeating the entire purpose of summarizing in the first place.

---

## 12. Best Practices (2026)

- Use recursive, threshold-triggered summarization for working memory, keeping the most recent turns verbatim and only summarizing older content.
- Design explicit criteria (or a well-tested extraction prompt) for what counts as a durable, storage-worthy fact versus ephemeral session content, rather than storing everything indiscriminately.
- Incorporate recency/time-decay into long-term memory retrieval scoring, not just raw semantic similarity.
- Build explicit fact-update/reconciliation logic for long-term memory, so that superseded information can be corrected/updated rather than accumulating indefinitely as potentially contradictory stored facts.

---

## 13. Exercises

**Easy:** Implement `WorkingMemoryManager` (Section 5) and trace through a 15-turn simulated conversation, printing the context at each turn to observe summarization triggering.
**Medium:** Implement time-decayed retrieval (Section 5) and empirically find a decay rate $\lambda$ at which a memory from "yesterday" outranks a more semantically similar memory from "6 months ago" for a test query.
**Hard:** Implement recursive summarization (Section 6) using a real LLM call, and verify across many summarization rounds that the summary length stays roughly bounded rather than growing over time.
**Mathematical:** Derive the context-window budget allocation constraint (Section 3) for a specific application (e.g., a customer support bot with a fixed system prompt size, targeting a maximum of 5 retrieved long-term memories and the last 10 conversation turns) and compute the maximum safe token budget for each component given a specific model's context window size.
**Coding:** Implement a fact-extraction and reconciliation system: extract facts from a conversation, check whether any newly extracted fact contradicts an existing stored fact (via semantic similarity + an LLM-based conflict check), and implement an update strategy for resolving such conflicts.

---

## 14. Mini Project

Build a **complete hybrid memory system for an actuarial assistant application**: implement working-memory management with recursive summarization for long consulting-style conversations, a long-term memory store with time-decayed retrieval for persisting facts about a user's ongoing projects/preferences across sessions, a fact-extraction pipeline distinguishing durable from ephemeral content, and a simple reconciliation mechanism for updating superseded facts — then simulate a multi-session interaction history and demonstrate the system correctly retrieves and applies relevant prior context in a new session while keeping context-window usage bounded throughout.

---

## 15. Interview Preparation

- Explain the difference between working memory and long-term memory in an LLM application, and why both are typically needed.
- Why must conversation summarization be recursive/updating rather than simply additive?
- How would you design a memory retrieval scoring function that balances semantic relevance against recency?
- What criteria would you use to decide whether a piece of conversation content is worth persisting to long-term memory?

---

## 16. Summary

Memory systems engineer around LLMs' fundamental statelessness — working memory manages a single session's growing transcript via recursive summarization within context-window/cost constraints (Phase 6 Lesson 1), while long-term memory persists durable facts across sessions using architecture nearly identical to Lesson 1's RAG (extraction, storage, retrieval), enhanced with time-decay to appropriately weight recency alongside semantic similarity. Getting the durability-filtering, summarization-recursion, and update/reconciliation details right is what separates a memory system that genuinely improves an assistant's usefulness over time from one that either loses important context or accumulates unbounded, increasingly irrelevant clutter — directly the architecture underlying this curriculum's own memory-filesystem feature.

---

## 17. References

- LangChain documentation on conversation memory abstractions
- Park et al. — "Generative Agents: Interactive Simulacra of Human Behavior" (2023, an influential exploration of memory, reflection, and retrieval in LLM agent architectures)
- Packer et al. — "MemGPT: Towards LLMs as Operating Systems" (2023, a specific architecture for managing memory as a hierarchical, OS-like system directly relevant to this lesson's working/long-term memory distinction)
- Mem0 and similar open-source memory-layer projects' documentation
