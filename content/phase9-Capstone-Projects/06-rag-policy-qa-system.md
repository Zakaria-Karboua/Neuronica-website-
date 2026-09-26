# Phase 9 · Project 6 — RAG-Based Insurance Policy Q&A System

> Difficulty: ★★★☆☆ | Primary phases exercised: 7 (Lessons 1–2, 6, 10), 8 (Lessons 1, 8, 11)

---

## Overview

A "chat with your policy documents" system: users ask natural-language questions about insurance policy wordings, and the system retrieves relevant clauses and generates a grounded, cited answer — the first genuinely RAG-based project in this sequence, introducing vector databases, hybrid search, and faithfulness evaluation as first-class production concerns.

---

## System Architecture

```
Policy documents ──▶ Chunking (paragraph-aware) ──▶ Embedding ──▶ Vector DB (Qdrant/pgvector)
                                                                          │
User question ──▶ FastAPI ──▶ Embed query ──▶ Hybrid search (dense + BM25) ──▶ Re-rank
                                                                          │
                                                                          ▼
                                                    Augmented prompt ──▶ LLM ──▶ Cited answer
                                                                          │
                                                                          ▼
                                              Faithfulness check (Phase 7 Lesson 10) ──▶ response or flag
```

---

## Folder Structure

```
policy-qa-rag/
├── src/
│   ├── ingestion/
│   │   ├── chunking.py           # paragraph/clause-aware, Phase 7 Lesson 1
│   │   └── indexer.py            # embed + upsert to vector DB
│   ├── retrieval/
│   │   ├── dense_search.py
│   │   ├── bm25_search.py        # Phase 7 Lesson 1's hybrid search
│   │   └── reranker.py
│   ├── generation/
│   │   ├── prompt_builder.py     # citation-requesting prompt template
│   │   └── faithfulness_check.py # Phase 7 Lesson 10's entailment check
│   ├── api/main.py
│   └── evaluation/
│       ├── retrieval_eval.py      # Recall@k, MRR
│       └── generation_eval.py     # faithfulness, answer relevance
├── tests/
│   ├── unit/test_chunking_boundaries.py
│   ├── integration/test_end_to_end_qa.py
│   └── eval/golden_qa_set.json    # a curated (question, expected_source_clause) test set
├── docker-compose.yml              # API + vector DB + BM25 index, all containerized
└── docs/evaluation_report.md
```

---

## Documentation

`docs/evaluation_report.md` reports Recall@k/MRR for retrieval and faithfulness/answer-relevance scores for generation *separately* (Phase 7 Lesson 1's diagnostic principle: a poor answer might stem from either stage, and conflating them wastes debugging effort) — plus explicit documentation of chunking strategy and why it was chosen (paragraph-boundary-aware chunking with overlap, given policy documents' clause structure).

---

## Testing

Golden Q&A regression set (a hand-curated set of realistic questions with known correct source clauses) run on every CI build — retrieval Recall@3 and generation faithfulness must not regress below documented thresholds; chunking boundary tests verifying clauses aren't split mid-sentence; hybrid-search tests confirming an exact-policy-number query is retrieved correctly even when dense-only search would rank it lower.

---

## Dockerization

`docker-compose.yml` orchestrates the FastAPI app, a vector database container (Qdrant), and (if using a self-hosted LLM rather than an API) a model-serving container — this is the first project requiring multi-container local orchestration, a direct stepping stone toward Kubernetes multi-service deployment.

---

## CI/CD

Quality gate includes the golden Q&A set's faithfulness and Recall@k scores (Phase 7 Lesson 10 + Phase 8 Lesson 4's gating pattern) — a genuinely different gate shape than classical-ML projects, since there's no single scalar accuracy metric but a small suite of retrieval + generation quality signals, each independently gated.

---

## Deployment

Kubernetes: separate Deployments for the API and the vector database (or a managed vector DB service, Phase 8 Lesson 5's build-vs-buy decision applied here), with the API's readiness probe checking actual vector-DB connectivity, not just its own process health.

---

## Monitoring

Retrieval quality tracked over time (Recall@k on a rotating sample of production queries with periodic human-labeled relevance judgments); faithfulness score distribution; citation-rate (what fraction of responses include a verifiable citation) as a proxy quality signal; token cost per query (Phase 8 Lesson 11).

---

## Evaluation

Full Phase 7 Lesson 10 pipeline: faithfulness (claim-level entailment checking against retrieved context), answer relevance (LLM-as-judge, validated against human ratings via Cohen's kappa before trusting it at scale), and context relevance (is retrieval itself surfacing the right documents) — each reported with confidence intervals.

---

## Scalability

Index scaling considerations as the policy corpus grows (Phase 7 Lesson 2's HNSW/IVF tradeoffs) — benchmarked recall/latency at the corpus's current scale and a projected 10x-larger future scale, with an explicit reindexing/migration plan documented.

---

## Security Considerations

Prompt-injection risk assessment (Phase 8 Lesson 9): policy documents are a *controlled* corpus (not arbitrary user-uploaded or web content), substantially reducing indirect-injection risk compared to a general web-browsing agent — but the input-separation prompt structure (Phase 8 Lesson 9's Layer 1) is still applied as defense in depth, and access control ensures users can only query documents they're authorized to see (a genuine multi-tenant data-isolation requirement).

---

## Definition of Done

- [ ] Golden Q&A evaluation set passes documented Recall@k and faithfulness thresholds
- [ ] Hybrid search verified to outperform dense-only search on exact-identifier queries
- [ ] Multi-container local deployment (docker-compose) and Kubernetes deployment both working
- [ ] Retrieval and generation quality monitored and reported separately in production
- [ ] Multi-tenant document access control verified via test
