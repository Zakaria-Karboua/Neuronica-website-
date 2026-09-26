# Phase 6 · Lesson 2 — Embeddings

> Prerequisite: Tokenization (Lesson 1), Phase 3 Linear Algebra, Phase 4 Lesson 2 (Unsupervised Learning)

---

## 1. Introduction

### What are embeddings?
Dense, continuous vector representations of discrete objects (tokens, words, sentences, images) learned such that semantically/functionally similar objects end up close together in the vector space — the mechanism by which discrete token IDs (Lesson 1) become the continuous-valued inputs that Transformer layers (Phase 5) can process via matrix multiplication and attention.

### Why does it exist?
A one-hot encoding (Phase 2 Lesson 6) of a 50,000-word vocabulary is a 50,000-dimensional sparse vector with no notion of similarity — "cat" and "dog" are equally "far apart" (in one-hot space) as "cat" and "refrigerator." Embeddings learn a much lower-dimensional (hundreds to thousands of dimensions), dense representation where geometric proximity reflects genuine semantic/functional similarity — a representation neural networks can actually exploit via standard operations (dot products, Phase 5 Lesson 6's attention).

### Historical background
Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) were the first widely-adopted methods for learning static word embeddings from large text corpora via self-supervised objectives (predicting context words). Their famous vector-arithmetic property ($\text{king} - \text{man} + \text{woman} \approx \text{queen}$) demonstrated embeddings capture genuine relational/semantic structure, not just similarity. Modern Transformer-based models (Phase 5-6) produce **contextual** embeddings — a word's vector depends on its surrounding context, a fundamental advance over Word2Vec/GloVe's single, fixed vector per word regardless of context.

### Real-world motivation
Every LLM's first layer is an embedding lookup; every RAG system (Phase 7) depends entirely on sentence/document embeddings for semantic search; and understanding embedding geometry (what "similar" means in this space, and its limitations) is essential for correctly building and debugging any retrieval or semantic-similarity system.

---

## 2. Theory

### Static vs. contextual embeddings
- **Static embeddings** (Word2Vec, GloVe): one fixed vector per word, regardless of context — "bank" (river) and "bank" (financial institution) get the *same* vector, an acknowledged, real limitation.
- **Contextual embeddings** (BERT, GPT, and every modern Transformer): each token's embedding is computed *in context* via self-attention (Phase 5 Lesson 6) — "bank" gets a different vector depending on its surrounding sentence, correctly disambiguating meaning based on context.

### Word2Vec's two training objectives (foundational, still conceptually important)
- **Skip-gram**: given a center word, predict surrounding context words.
- **CBOW (Continuous Bag of Words)**: given surrounding context words, predict the center word.
Both are **self-supervised** — no human labels needed, just raw text — a training paradigm directly foreshadowing Lesson 5's LLM pretraining objective (predict a token from its context).

### The embedding matrix
For a vocabulary of size $V$ and embedding dimension $d$, the embedding layer is simply a learnable $V \times d$ matrix $E$; looking up token $i$'s embedding is literally selecting row $i$: $E[i,:]$ — a direct, simple operation, but for large vocabularies and dimensions, this matrix is often one of a model's largest parameter blocks (directly connecting to Lesson 1's vocabulary-size tradeoff).

### Sentence/document embeddings (pooling contextual token embeddings)
To get a single vector representing an entire sentence/document (needed for Phase 7's RAG retrieval), contextual token embeddings must be **pooled** — common strategies: mean pooling (average all token embeddings), using a special `[CLS]` token's embedding (BERT's convention), or using specialized sentence-embedding models (e.g., Sentence-BERT) explicitly trained so that pooled representations have good similarity properties (a genuinely non-trivial requirement — naively pooling embeddings from a model *not* specifically trained for this often produces poor-quality similarity rankings).

---

## 3. Mathematical Foundations

### Cosine similarity — the standard embedding comparison metric

$$
\text{cos\_sim}(u,v) = \frac{u \cdot v}{\|u\|\|v\|}
$$

Cosine similarity measures the angle between two vectors, ignoring magnitude — appropriate for embeddings because a vector's *direction* typically encodes semantic content while magnitude often reflects less meaningful factors (e.g., word frequency effects in some embedding schemes) — directly reusing Phase 3 Lesson 1's inner-product/norm machinery.

### Word2Vec's skip-gram objective, formalized (softmax approximation problem)

$$
P(w_{context}|w_{center}) = \frac{\exp(v_{context}\cdot v_{center})}{\sum_{w \in V}\exp(v_w \cdot v_{center})}
$$

The denominator sums over the *entire* vocabulary $V$ — computationally prohibitive for large vocabularies (every training step would require a full softmax over potentially hundreds of thousands of words). **Negative sampling** approximates this by only updating a small random sample of "negative" (non-context) words per step, converting an expensive multi-class classification into cheap binary classification against a handful of negative examples — a specific, practical trick directly enabling Word2Vec's efficiency, and conceptually related to techniques still used in modern contrastive learning objectives.

### The famous analogy property, explained

$$
v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}
$$

This emerges because the *directions* in embedding space corresponding to certain relationships (e.g., "gender," "royal status") turn out to be roughly consistent/linear across many word pairs, a genuinely remarkable and only partially theoretically explained empirical property of embeddings trained via co-occurrence-based objectives — useful for intuition-building but should not be over-relied upon as a universally robust or fully "solved" property (it holds well for some relationship types and much less reliably for others).

### Embedding space anisotropy (a known, practically important limitation)
Empirically, embeddings from many pretrained Transformer models are **anisotropic** — they occupy a narrow cone in the high-dimensional space rather than being spread out uniformly, which can make raw cosine similarities systematically compressed/less discriminative than expected. This is precisely why specialized sentence-embedding training (contrastive objectives, explicitly optimizing for well-separated similarity geometry) meaningfully outperforms naively pooling embeddings from a general-purpose language model for retrieval tasks (Phase 7).

---

## 4. Algorithm — Skip-Gram with Negative Sampling (conceptual)

```
GIVEN a text corpus, window size w, embedding dimension d, number of negative samples k:
INITIALIZE embedding matrices (center-word embeddings E_center, context-word embeddings E_context) randomly
FOR each position i in the corpus:
    center_word = corpus[i]
    FOR each context_word within window w of position i:
        # POSITIVE example: (center_word, context_word) actually co-occur
        loss += -log(sigmoid(E_center[center_word] . E_context[context_word]))

        # NEGATIVE examples: k randomly sampled words that DON'T actually appear in this context
        FOR k negative_word samples (sampled proportional to a smoothed unigram frequency distribution):
            loss += -log(sigmoid(-E_center[center_word] . E_context[negative_word]))

    UPDATE E_center, E_context via gradient descent (Phase 3 Lesson 5) to minimize this loss
RETURN E_center (typically used as the final word embeddings)
```
Negative sampling turns what would be a $|V|$-way softmax classification (Section 3's computational problem) into $k+1$ simple binary classifications per training example — a dramatic, practically essential efficiency gain for training on realistically large vocabularies.

---

## 5. Python Implementation

```python
"""embeddings_core.py — embedding lookup, cosine similarity, and a tiny skip-gram trainer"""
import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)


def most_similar(query_vec: np.ndarray, embedding_matrix: np.ndarray, vocab: list[str], top_k: int = 5):
    sims = np.array([cosine_similarity(query_vec, embedding_matrix[i]) for i in range(len(vocab))])
    top_indices = np.argsort(-sims)[:top_k]
    return [(vocab[i], sims[i]) for i in top_indices]


class SkipGramTrainer:
    """A simplified, from-scratch skip-gram model with negative sampling."""
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.E_center = rng.normal(0, 0.1, (vocab_size, embed_dim))
        self.E_context = rng.normal(0, 0.1, (vocab_size, embed_dim))

    def sigmoid(self, z): return 1 / (1 + np.exp(-z))

    def train_step(self, center_id: int, context_id: int, negative_ids: list[int], lr: float = 0.01):
        v_center = self.E_center[center_id]
        v_context = self.E_context[context_id]

        # Positive pair: push similarity UP
        pos_score = self.sigmoid(v_center @ v_context)
        grad_pos = (pos_score - 1)                        # derivative of -log(sigmoid(x)) w.r.t. x
        self.E_center[center_id] -= lr * grad_pos * v_context
        self.E_context[context_id] -= lr * grad_pos * v_center

        # Negative pairs: push similarity DOWN
        for neg_id in negative_ids:
            v_neg = self.E_context[neg_id]
            neg_score = self.sigmoid(v_center @ v_neg)
            grad_neg = neg_score                            # derivative of -log(sigmoid(-x)) w.r.t x, simplified
            self.E_center[center_id] -= lr * grad_neg * v_neg
            self.E_context[neg_id] -= lr * grad_neg * v_center


if __name__ == "__main__":
    vocab = ["king", "queen", "man", "woman", "apple", "banana", "fruit", "throne"]
    vocab_size, embed_dim = len(vocab), 16
    rng = np.random.default_rng(0)
    model = SkipGramTrainer(vocab_size, embed_dim)

    # Simulate training: (king, throne) and (queen, throne) as positive co-occurrences,
    # random unrelated words as negatives -- a toy illustration, not a real corpus
    for _ in range(500):
        model.train_step(vocab.index("king"), vocab.index("throne"),
                          negative_ids=[vocab.index("apple"), vocab.index("banana")])
        model.train_step(vocab.index("queen"), vocab.index("throne"),
                          negative_ids=[vocab.index("apple"), vocab.index("fruit")])

    print(most_similar(model.E_center[vocab.index("king")], model.E_center, vocab, top_k=3))
```

---

## 6. Build From Scratch

**Mean-pooling sentence embeddings from contextual token embeddings (the RAG-relevant operation, Phase 7 preview):**
```python
import numpy as np

def mean_pool_embeddings(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """token_embeddings: (seq_len, d). attention_mask: (seq_len,) -- 1 for real tokens, 0 for padding."""
    mask_expanded = attention_mask[:, None]                          # (seq_len, 1) for broadcasting
    summed = np.sum(token_embeddings * mask_expanded, axis=0)
    counts = np.clip(mask_expanded.sum(axis=0), a_min=1e-9, a_max=None)
    return summed / counts

# Example: a 5-token sequence, last 2 are padding
rng = np.random.default_rng(0)
token_embs = rng.normal(size=(5, 8))
mask = np.array([1, 1, 1, 0, 0])   # only first 3 tokens are real content
sentence_embedding = mean_pool_embeddings(token_embs, mask)
print("Sentence embedding shape:", sentence_embedding.shape)   # (8,) -- one vector for the WHOLE sentence
```
**Critically, the attention mask must be applied correctly** — naively averaging over *all* positions (including padding) silently corrupts the sentence embedding with padding-token noise, a genuinely common bug when building retrieval systems (Phase 7) from scratch.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `SkipGramTrainer` | Pretrained Word2Vec/GloVe vectors (`gensim`), or simply use contextual embeddings from any modern Transformer instead |
| `mean_pool_embeddings` | `sentence-transformers` library — provides models specifically trained (via contrastive objectives) for high-quality sentence embeddings, substantially outperforming naive mean-pooling of a general-purpose LLM's token embeddings |
| Manual cosine similarity loop | `sklearn.metrics.pairwise.cosine_similarity`, or vector databases (Phase 7) with optimized approximate nearest-neighbor search at scale |

---

## 8. Visual Explanations

**Static vs. contextual embeddings (disambiguating "bank"):**
```
STATIC (Word2Vec/GloVe):                    CONTEXTUAL (BERT/GPT-style, via self-attention):
"bank" -> [0.2, -0.5, 0.8, ...]              "river bank"  -> [0.1, -0.5, 0.3, ...]  (context-adjusted)
   (SAME vector regardless of context,        "bank account" -> [0.6, 0.2, -0.4, ...] (DIFFERENT vector)
    even for river-bank vs financial-bank)     (Phase 5 Lesson 6's self-attention mixes in surrounding words)
```

**Embedding space analogy geometry:**
```
        queen ●
             ╱│
            ╱ │  (vector: woman - man)
           ╱  │
   king ●─────┘
        (vector: king - man + woman ≈ queen's position, empirically -- a genuinely
         useful intuition but not a universally reliable guarantee across all relationship types)
```

---

## 9. Practical Examples

**Simple:** load pretrained GloVe vectors and find the 5 most similar words to "computer" via cosine similarity.
**Medium:** train the from-scratch skip-gram model (Section 5) on a small real text corpus and qualitatively inspect whether related words end up close together.
**Real-world:** use a `sentence-transformers` model to embed a set of actuarial/insurance policy documents, compute pairwise cosine similarities, and identify near-duplicate or highly similar policy clauses — a direct, practical precursor to Phase 7's RAG retrieval systems.

---

## 10. Real Industry Use Cases

- **Every RAG system** (Phase 7): document/query embeddings and cosine/dot-product similarity search are the entire retrieval mechanism.
- **Recommendation systems**: user and item embeddings (learned via objectives structurally similar to Word2Vec's co-occurrence-based training) power collaborative filtering at companies like Spotify, Netflix, and Amazon.
- **Semantic search and deduplication**: legal/financial document review, plagiarism detection, and customer support ticket routing all rely on embedding-based similarity rather than exact keyword matching.
- **Every LLM's input layer**: the embedding matrix (Section 2) is the literal first learned parameter block in GPT, Claude, Llama, and every other Transformer-based model.

---

## 11. Common Mistakes

- Using a general-purpose LLM's raw token embeddings (naively mean-pooled) for retrieval instead of a model specifically trained for sentence-embedding quality (`sentence-transformers`) — often yields meaningfully worse similarity rankings due to the anisotropy issue (Section 3).
- Forgetting to mask out padding tokens during mean pooling — silently corrupts sentence embeddings with padding noise.
- Comparing embeddings from two *different* embedding models directly — embedding spaces from different models/training runs are not comparable to each other; cosine similarity is only meaningful *within* a single consistent embedding space.
- Over-interpreting the word-analogy property as a fully general, always-reliable feature of embedding spaces, rather than a genuinely interesting but imperfect empirical phenomenon.

---

## 12. Best Practices (2026)

- Use dedicated sentence/document embedding models (`sentence-transformers`, or proprietary embedding APIs from major LLM providers) for any retrieval/similarity application, rather than naively pooling a general-purpose LLM's token embeddings.
- Always verify attention-mask handling when pooling contextual embeddings — a small, easy-to-verify detail with outsized correctness consequences.
- Normalize embeddings (unit length) before storing them in a vector database if your similarity metric is cosine similarity — many vector search libraries default to (faster) dot-product search, which is mathematically equivalent to cosine similarity only when vectors are pre-normalized.
- Be aware of and account for embedding anisotropy when evaluating a retrieval system's similarity scores — raw score magnitudes are less directly interpretable than relative rankings.

---

## 13. Exercises

**Easy:** Load pretrained word embeddings and compute cosine similarity between several word pairs, checking that intuitively related words score higher than unrelated ones.
**Medium:** Implement the word analogy operation ($v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$) and find the nearest neighbor to the resulting vector, evaluating whether it correctly retrieves "queen."
**Hard:** Train the from-scratch skip-gram model (Section 5) on a moderately-sized real text corpus (e.g., a public domain book) and qualitatively evaluate the resulting embeddings' quality via nearest-neighbor inspection for several test words.
**Mathematical:** Derive the gradient of the negative-sampling loss with respect to the center-word embedding, confirming it matches the `grad_pos`/`grad_neg` update rules used in Section 5's implementation.
**Coding:** Implement and compare mean pooling vs. `[CLS]`-token pooling for sentence embeddings using a pretrained Transformer model, evaluating which produces better similarity rankings on a small labeled sentence-similarity dataset.

---

## 14. Mini Project

Build a **semantic document similarity tool** for actuarial/insurance text: using `sentence-transformers`, embed a collection of policy clauses or claims descriptions, compute a full pairwise cosine similarity matrix, identify clusters of near-duplicate or highly related content (directly connecting to Phase 4 Lesson 2's clustering techniques, now applied in embedding space), and visualize the embedding space in 2D via dimensionality reduction (Phase 4 Lesson 2's PCA or t-SNE) to qualitatively inspect whether semantically related documents indeed cluster together.

---

## 15. Interview Preparation

- Explain the difference between static embeddings (Word2Vec) and contextual embeddings (BERT/GPT-style).
- What is negative sampling, and what computational problem does it solve?
- Why is cosine similarity typically preferred over Euclidean distance for comparing embeddings?
- What is embedding anisotropy, and why does it motivate specialized sentence-embedding training over naive pooling?

---

## 16. Summary

Embeddings convert discrete tokens into dense, geometrically meaningful vectors — static embeddings (Word2Vec/GloVe) learn one fixed vector per word via self-supervised co-occurrence prediction (using negative sampling for computational tractability), while modern contextual embeddings (produced by every Transformer layer, Phase 5) generate a different vector for the same word depending on surrounding context, correctly resolving ambiguity that static embeddings cannot. Cosine similarity is the standard comparison metric, though embedding space anisotropy means naively pooled general-purpose embeddings underperform specialized sentence-embedding models for retrieval — a direct, practical consideration that shapes exactly how Phase 7's RAG systems are built.

---

## 17. References

- Mikolov et al. — "Efficient Estimation of Word Representations in Vector Space" (2013, Word2Vec)
- Pennington, Socher, Manning — "GloVe: Global Vectors for Word Representation" (2014)
- Reimers & Gurevych — "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019, the foundational sentence-embeddings paper)
- Ethayarajh, K. — "How Contextual are Contextualized Word Representations?" (2019, the key paper on embedding anisotropy)
