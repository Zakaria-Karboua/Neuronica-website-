# Phase 7 · Lesson 1 — Retrieval-Augmented Generation (RAG)

> Prerequisite: Phase 6 (Embeddings, Prompt Engineering especially), Phase 4 Lesson 2 (Unsupervised Learning — similarity/clustering)

---

## 1. Introduction

### What is RAG?
An architecture pattern that combines a retrieval system (finding relevant documents/passages from an external knowledge source) with an LLM's generation capability — instead of relying solely on knowledge baked into the model's pretrained weights (Phase 6 Lesson 5), the model is given relevant retrieved text directly in its context window and asked to generate a response grounded in that provided information.

### Why does it exist?
LLMs have a fixed knowledge cutoff (Phase 6 Lesson 5) and can **hallucinate** — generate fluent, confident-sounding but factually incorrect content, especially about specific facts, recent events, or proprietary/private information never seen during pretraining. RAG directly addresses both problems: retrieved documents can be current (updated independently of the model), and grounding generation in actual retrieved text substantially reduces (though does not eliminate) hallucination, while also enabling the model to answer questions about private/proprietary data it was never trained on.

### Historical background
The RAG architecture was formalized by Lewis et al. (2020, Facebook AI), combining a dense retriever with a sequence-to-sequence generator in an end-to-end trainable system. The far more common 2023-2026 pattern — sometimes called "RAG" more loosely — decouples retrieval and generation entirely: an off-the-shelf embedding model (Phase 6 Lesson 2) handles retrieval, and an off-the-shelf LLM handles generation, with no joint training required — a pragmatic, hugely popular simplification that made RAG accessible to virtually any team with an LLM API key and a vector database.

### Real-world motivation
Nearly every "chat with your documents" or enterprise knowledge-base assistant application is RAG — directly relevant to any SANAVIR or actuarial application where you'd want an LLM to answer questions grounded in your own proprietary data (agricultural sensor documentation, policy wordings, internal reports) rather than relying on the model's general pretrained knowledge.

---

## 2. Theory

### The RAG pipeline, end to end
1. **Indexing** (offline, done once/periodically): split documents into chunks, embed each chunk (Phase 6 Lesson 2), store embeddings in a vector database (Lesson 2, this phase).
2. **Retrieval** (per query, at inference time): embed the user's query using the same embedding model, find the $k$ most similar chunks via vector similarity search.
3. **Augmentation**: insert the retrieved chunks into the LLM's prompt, alongside the original query, typically with clear formatting/instructions indicating "use this context to answer."
4. **Generation**: the LLM generates a response, ideally grounded in and citing the provided context, rather than relying solely on parametric (pretrained) knowledge.

### Chunking strategy — a genuinely consequential design decision
Documents must be split into chunks small enough to fit multiple retrieved results within the LLM's context window, but large enough to preserve coherent meaning (a chunk cut off mid-sentence or missing surrounding context can be retrieved as "relevant" by embedding similarity while being useless or misleading without its full context). Common strategies: fixed-size chunking (simple, often suboptimal), sentence/paragraph-boundary-aware chunking, and semantic chunking (splitting at points of genuine topic change, often itself detected via embedding similarity between adjacent sentences).

### Retrieval quality metrics
- **Recall@k**: does the relevant document appear anywhere in the top $k$ retrieved results?
- **Precision@k**: what fraction of the top $k$ retrieved results are actually relevant?
- **MRR (Mean Reciprocal Rank)**: rewards ranking the relevant result *higher* within the top $k$, not just including it somewhere.

### Why RAG reduces (but doesn't eliminate) hallucination
Grounding generation in retrieved text gives the model concrete, current, verifiable content to draw from — but the model can still misread, misattribute, or subtly distort the retrieved content, or (a documented failure mode) ignore the retrieved context entirely and fall back on parametric knowledge, especially when the retrieved context conflicts with strong pretrained priors. RAG is a substantial mitigation, not a complete solve — a crucial, honest distinction for setting appropriate expectations in any RAG application.

---

## 3. Mathematical Foundations

### Vector similarity search, formalized (directly reusing Phase 6 Lesson 2)
Given a query embedding $q$ and a corpus of document chunk embeddings $\{d_1, \dots, d_n\}$, retrieval finds:

$$
\text{top-}k = \arg\text{top-}k_i \; \text{cos\_sim}(q, d_i) \quad \text{or} \quad \arg\text{top-}k_i \; q \cdot d_i
$$

Exhaustive computation is $O(n)$ per query (comparing against every chunk) — fine for small corpora, but Lesson 2's approximate nearest-neighbor (ANN) structures become essential once $n$ reaches millions of chunks, trading a small amount of retrieval accuracy for dramatically sublinear query time.

### Hybrid search — combining dense (embedding) and sparse (keyword) retrieval
Pure embedding-based (dense) retrieval can sometimes underperform on queries requiring exact keyword/entity matching (e.g., a specific policy number, an exact legal term) — because embeddings capture *semantic* similarity, not always precise lexical matching. **Hybrid search** combines dense retrieval with a classical sparse method (BM25, a refined term-frequency-based ranking function extending the intuitions behind Phase 2 Lesson 5's basic frequency analysis), typically via a weighted combination or re-ranking of both methods' results:

$$
\text{score}_{hybrid} = \alpha \cdot \text{score}_{dense} + (1-\alpha)\cdot \text{score}_{sparse}
$$

with $\alpha$ tuned empirically, often improving retrieval robustness over either method alone.

### Re-ranking — a second-stage precision refinement
Retrieve a larger initial candidate set (e.g., top 50) via fast approximate vector search, then apply a more expensive but more accurate **cross-encoder** re-ranker (a model that jointly processes the query and each candidate document together, rather than comparing pre-computed independent embeddings) to re-score and select the final top $k$ — a classic recall-then-precision two-stage pattern, directly analogous to information retrieval's long-standing "retrieve-then-rerank" architecture, now applied with embedding models and Transformers.

### Context window budget allocation
Given a fixed LLM context window of size $C$ tokens, and needing to fit the system prompt, retrieved chunks, conversation history, and the user's query, chunk size and retrieval count $k$ must be chosen so that $\sum(\text{chunk sizes}) + \text{overhead} \le C$ — a genuinely practical engineering constraint directly connecting back to Phase 6 Lesson 1's tokenization cost/length considerations.

---

## 4. Algorithm — The Full RAG Query Pipeline (fully specified)

```
INDEXING (offline, run once or on a schedule as documents update):
FOR each document in the corpus:
    split into chunks (using a chosen chunking strategy, Section 2)
    FOR each chunk:
        embed the chunk (Phase 6 Lesson 2's embedding model)
        store (chunk_text, chunk_embedding, metadata) in the vector database (Lesson 2, this phase)

QUERY TIME (per user request):
1. embed the user's query using the SAME embedding model used for indexing
2. RETRIEVE: vector similarity search -> top-k candidate chunks (optionally: hybrid dense+sparse, Section 3)
3. (OPTIONAL) RE-RANK: apply a cross-encoder to re-score and select the final top-k' <= k chunks
4. CONSTRUCT the augmented prompt:
     system_instruction + "Use the following context to answer:\n" + retrieved_chunks + user_query
5. GENERATE: pass the augmented prompt to the LLM (Phase 6), optionally requesting citations
6. (OPTIONAL) POST-PROCESS: verify the response's claims against the retrieved context (Lesson 10's
   evaluation/guardrails territory), flag or reject ungrounded claims
RETURN the generated response (ideally with citations to the specific retrieved chunks used)
```

---

## 5. Python Implementation

```python
"""rag_core.py — a complete, minimal RAG pipeline using sentence-transformers + a simple in-memory index"""
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleRAGPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def chunk_document(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
        """Simple fixed-size chunking WITH OVERLAP (overlap helps preserve context across chunk boundaries)."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap             # step forward LESS than chunk_size -> overlapping windows
        return chunks

    def index_documents(self, documents: list[str]) -> None:
        for doc in documents:
            self.chunks.extend(self.chunk_document(doc))
        self.embeddings = self.embedder.encode(self.chunks, normalize_embeddings=True)   # unit-normalized

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        similarities = self.embeddings @ query_embedding    # dot product of NORMALIZED vectors = cosine sim
        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.chunks[i], similarities[i]) for i in top_indices]

    def build_augmented_prompt(self, query: str, top_k: int = 3) -> str:
        retrieved = self.retrieve(query, top_k)
        context = "\n\n".join(f"[Source {i+1}]: {chunk}" for i, (chunk, score) in enumerate(retrieved))
        return (
            f"Use ONLY the following context to answer the question. "
            f"If the context doesn't contain the answer, say so explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite sources like [Source 1]):"
        )


if __name__ == "__main__":
    docs = [
        "Precision agriculture uses sensor data including soil moisture, temperature, and NDVI "
        "vegetation indices to optimize irrigation and fertilization decisions across a farm's plots.",
        "Actuarial mortality tables are typically constructed using historical claims experience, "
        "adjusted for trend and credibility-weighted against broader population tables when experience is thin.",
    ]
    rag = SimpleRAGPipeline()
    rag.index_documents(docs)
    prompt = rag.build_augmented_prompt("How is soil moisture used in farming decisions?")
    print(prompt)
    # This final `prompt` would then be passed to an actual LLM (Phase 6) for grounded generation
```

---

## 6. Build From Scratch

**A minimal BM25 sparse retrieval implementation (for hybrid search, Section 3):**
```python
import numpy as np
from collections import Counter

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = np.mean(self.doc_lengths)
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.n_docs = len(corpus)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict:
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        return {term: np.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        freqs = self.doc_freqs[doc_idx]
        for term in query_terms:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0)
            tf = freqs[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        query_terms = query.lower().split()
        scores = [(i, self.score(query_terms, i)) for i in range(self.n_docs)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]
```
BM25's formula directly generalizes the intuition from Phase 2 Lesson 5's frequency analysis: term frequency (`tf`) matters, but is dampened (via `k1`) so that a term appearing 20 times isn't considered 20x more relevant than one appearing once, and is normalized by document length (`b`) so that longer documents don't win purely by containing more words overall — combining this with dense embedding search (Section 3's hybrid approach) captures both keyword-precision and semantic-similarity retrieval strengths.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `SimpleRAGPipeline` (in-memory, `numpy` similarity search) | Vector databases (Lesson 2, this phase) — Pinecone, Weaviate, Qdrant, `pgvector` — scale to millions/billions of vectors with approximate nearest-neighbor indexing |
| `BM25` from scratch | `rank-bm25` (Python library), or Elasticsearch/OpenSearch's built-in BM25 implementation for production-scale hybrid search |
| Manual chunking | LangChain's/LlamaIndex's text splitters — support recursive, semantic, and format-aware (Markdown/code-aware) chunking strategies out of the box |
| Manual prompt construction | RAG orchestration frameworks (LangChain, LlamaIndex) — provide higher-level abstractions for the full pipeline, at the cost of some additional complexity/opacity |

---

## 8. Visual Explanations

**The full RAG pipeline (indexing + query time):**
```
INDEXING (offline):                          QUERY TIME (per request):
Documents ──▶ Chunk ──▶ Embed ──▶ Vector DB    User Query ──▶ Embed (SAME model) ──┐
                                                                                      ▼
                                                                          Vector DB similarity search
                                                                                      │
                                                                                      ▼
                                                                        Top-k retrieved chunks
                                                                                      │
                                                                                      ▼
                                                            Augmented Prompt (query + retrieved context)
                                                                                      │
                                                                                      ▼
                                                                              LLM ──▶ Grounded Response
```

**Hybrid search combining dense + sparse retrieval:**
```
Query: "policy 48291 exclusions"
                │
       ┌────────┴────────┐
       ▼                 ▼
 Dense (embedding)   Sparse (BM25)
 finds SEMANTICALLY   finds EXACT keyword
 similar clauses      match on "48291"
       │                 │
       └────────┬────────┘
                ▼
      Combined/re-ranked results
   (catches BOTH semantic relevance AND exact identifier matching)
```

---

## 9. Practical Examples

**Simple:** build the `SimpleRAGPipeline` (Section 5), index a few short documents, and retrieve the top-3 most relevant chunks for a test query.
**Medium:** implement BM25 (Section 6) alongside dense retrieval on the same corpus, and compare which method retrieves better results for a query containing a specific exact identifier versus a query requiring semantic understanding.
**Real-world:** build a RAG system over a collection of actuarial/insurance policy documents (or SANAVIR agricultural sensor documentation), experimenting with different chunk sizes and overlap settings, and qualitatively evaluate how chunking choices affect retrieval quality and answer groundedness for a set of realistic test questions.

---

## 10. Real Industry Use Cases

- **Enterprise knowledge-base assistants**: the dominant, most common production RAG application — "chat with your company's documents/wiki/policies."
- **Legal and insurance document Q&A**: directly relevant to your domain — RAG systems over policy wordings, claims history, and regulatory documents, specifically to ground answers in the actual, current, authoritative text rather than the LLM's general (and potentially outdated or jurisdiction-mismatched) pretrained knowledge.
- **Customer support automation**: RAG over product documentation/FAQs/support tickets, grounding responses in a company's actual current documentation.
- **Code assistants**: RAG over a codebase's own files (retrieving relevant existing code/documentation) to ground code generation/explanation in the actual project context, rather than generic patterns from pretraining alone.

---

## 11. Common Mistakes

- Choosing a fixed chunk size without considering document structure — cutting mid-sentence or mid-table destroys the chunk's standalone meaning, degrading both retrieval relevance and generation quality even when the "wrong half" of a relevant passage happens to be retrieved.
- Relying purely on dense (embedding) retrieval for queries involving exact identifiers, codes, or rare proper nouns that embeddings may not represent precisely — hybrid search (Section 3) directly addresses this.
- Not testing retrieval quality independently from generation quality — a RAG system's poor overall answers might stem from bad retrieval (wrong chunks retrieved) OR bad generation (right chunks retrieved, but the LLM ignores/misuses them) — these require different fixes, and conflating them wastes debugging effort.
- Assuming RAG eliminates hallucination entirely — a model can still misread or ignore retrieved context; production systems should include verification/evaluation (Lesson 10) rather than assuming grounding alone guarantees correctness.

---

## 12. Best Practices (2026)

- Evaluate retrieval and generation quality separately (Recall@k/MRR for retrieval; groundedness/faithfulness metrics for generation, Lesson 10) to correctly diagnose where a RAG system's weaknesses actually lie.
- Use hybrid (dense + sparse) retrieval as a robust default, especially for domains (like insurance/legal) where exact terminology/identifiers matter alongside semantic understanding.
- Use a re-ranking stage (Section 3) when initial retrieval quality is insufficient — often a meaningfully cost-effective quality improvement for a modest latency increase.
- Explicitly instruct the LLM (via prompt engineering, Phase 6 Lesson 9) to cite sources and to explicitly state when the provided context doesn't contain a sufficient answer, rather than falling back on ungrounded parametric knowledge.

---

## 13. Exercises

**Easy:** Implement fixed-size chunking with and without overlap, and manually inspect how overlap affects whether important context survives near chunk boundaries.
**Medium:** Build the full `SimpleRAGPipeline` (Section 5) and evaluate Recall@3 on a small labeled set of (query, relevant document) pairs you construct yourself.
**Hard:** Implement BM25 (Section 6) and combine it with dense retrieval via a weighted hybrid score, empirically tuning the weighting parameter $\alpha$ (Section 3) on a small test set containing both semantic and exact-match-style queries.
**Mathematical:** Derive why cosine similarity between L2-normalized vectors is mathematically equivalent to their dot product, justifying the `normalize_embeddings=True` + dot-product pattern used in Section 5.
**Coding:** Implement a re-ranking stage using a cross-encoder model (e.g., from `sentence-transformers`' cross-encoder collection) on top of an initial dense-retrieval candidate set, and measure whether it improves ranking quality (MRR) on a small test set.

---

## 14. Mini Project

Build a **complete RAG system over actuarial/insurance policy documents**: implement document chunking with a thoughtful strategy (respecting paragraph/clause boundaries where possible), index using dense embeddings with a hybrid BM25 fallback, implement a re-ranking stage, construct a well-engineered augmented prompt (Phase 6 Lesson 9) requesting explicit citations, and evaluate the system on a set of realistic test questions — measuring retrieval Recall@k and manually assessing generation groundedness (does the answer actually reflect what's in the cited sources) — producing a genuinely practical, evaluated RAG application directly relevant to your domain.

---

## 15. Interview Preparation

- Explain the full RAG pipeline from document indexing through final generation.
- Why might dense (embedding-based) retrieval alone be insufficient for certain queries, and what does hybrid search add?
- What is re-ranking, and why is it typically applied as a second stage after initial retrieval rather than replacing it?
- Does RAG eliminate hallucination? Explain what it does and doesn't guarantee.

---

## 16. Summary

RAG grounds LLM generation in retrieved, current, verifiable external text rather than relying solely on fixed pretrained knowledge — a pipeline of chunking, embedding-based (optionally hybrid dense+sparse) retrieval, optional re-ranking, and prompt augmentation that directly builds on Phase 6's embeddings and prompt engineering lessons. It substantially reduces (without eliminating) hallucination and enables grounding in private/proprietary/current data the model was never trained on — precisely the architecture underlying most practical "chat with your documents" applications, and the natural foundation for Lesson 2's vector databases and the agentic systems (Lessons 3+) that follow.

---

## 17. References

- Lewis et al. — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020, the original RAG paper)
- Robertson & Zaragoza — "The Probabilistic Relevance Framework: BM25 and Beyond" (2009, the definitive BM25 reference)
- Gao et al. — "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023/2024, a comprehensive modern survey)
- LangChain and LlamaIndex official documentation (widely-used RAG orchestration frameworks)
