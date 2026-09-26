# Phase 6 · Lesson 1 — Tokenization

> Prerequisite: Phase 5 (Transformers, Lesson 7), Phase 1 Lesson 4 (Data Structures — heaps, DP)

---

## 1. Introduction

### What is tokenization?
The process of converting raw text into a sequence of discrete units ("tokens") that a language model can consume as input IDs — the very first step of every LLM pipeline, both at training and inference time, and the interface between human-readable text and the numeric tensors Phase 5's Transformer architecture actually operates on.

### Why does it exist?
Neural networks operate on numbers, not characters or words directly — some mapping from text to a fixed, finite vocabulary of integer IDs is unavoidable. The specific *scheme* chosen (character-level, word-level, or subword-level) has enormous practical consequences: vocabulary size, sequence length, handling of rare/unseen words, and even how efficiently different languages are represented — all of which directly affect training cost, inference speed, and model quality.

### Historical background
Early neural NLP used word-level tokenization (a fixed vocabulary of whole words) — simple, but hopeless with rare words, typos, or morphologically rich languages (an "unknown word" problem with no graceful fallback). Byte-Pair Encoding (BPE), originally a 1994 data compression algorithm, was repurposed for NLP tokenization (Sennrich et al., 2015) and became the dominant approach, later refined into variants (WordPiece for BERT, and the byte-level BPE used by GPT-2 onward) that remain the standard in 2026's frontier LLMs.

### Real-world motivation
Tokenization directly determines what an LLM can and cannot represent efficiently: why LLMs are notoriously bad at character-level tasks (like counting letters in a word), why non-English languages often require more tokens per unit of text (affecting both cost and effective context length), and why certain arithmetic/code tasks are sensitive to how numbers get tokenized — all practical realities every LLM application developer eventually encounters directly.

---

## 2. Theory

### The tokenization granularity spectrum
| Granularity | Vocabulary size | Sequence length | Handles unseen words? |
|---|---|---|---|
| Character-level | Tiny (~100s) | Very long | Yes, trivially |
| Word-level | Huge (100,000s+) | Short | No — "unknown token" fallback needed |
| Subword-level (BPE/WordPiece) | Moderate (10,000s-100,000s) | Moderate | Yes — falls back to smaller subword/character units |

Subword tokenization is the practical sweet spot: common words remain single tokens (efficient), while rare/unseen words decompose into smaller, still-meaningful subword pieces (graceful degradation) rather than a single opaque "unknown" token.

### Byte-Pair Encoding (BPE), the dominant algorithm
Starting from individual characters (or bytes), BPE **iteratively merges the most frequent adjacent pair** of tokens into a new single token, repeating for a fixed number of merges (determined by the target vocabulary size) — a bottom-up, frequency-driven procedure that naturally learns common words as single tokens (through repeated merging) while leaving rare words as sequences of smaller, still-interpretable pieces.

### WordPiece (BERT's tokenizer) — a related but distinct variant
Similar iterative merging, but instead of choosing the most *frequent* pair, WordPiece chooses the pair that most increases the **likelihood** of the training corpus under a simple language model — a subtly different, likelihood-driven (rather than pure-frequency-driven) merge criterion.

### Byte-level BPE (GPT-2 onward) — handling ANY input robustly
Rather than operating on Unicode characters directly (which could include tens of thousands of distinct characters across all languages/emoji/symbols), byte-level BPE operates on raw UTF-8 **bytes** (always exactly 256 possible values) — guaranteeing *any* input text, in any language or containing any symbol, can always be tokenized without ever needing an "unknown token" fallback, a genuinely elegant robustness property.

---

## 3. Mathematical Foundations

### BPE's merge criterion, formalized
At each iteration, given the current tokenization of the training corpus, count all adjacent token pair frequencies:

$$
\text{pair\_freq}(a,b) = \text{count of occurrences where token } a \text{ is immediately followed by token } b
$$

Merge the pair $(a^*,b^*) = \arg\max_{(a,b)} \text{pair\_freq}(a,b)$ into a new single token $ab$, and repeat until reaching the target vocabulary size. This is precisely a **greedy algorithm** (Phase 1 Lesson 4) — locally optimal at each step, not globally optimal over the full sequence of merges, but computationally tractable and empirically effective.

### Connection to Huffman coding and information theory (Phase 3 Lesson 6, directly reused)
BPE is conceptually related to (though not identical to) Huffman coding: both build a variable-length encoding scheme where more frequent patterns get shorter representations (fewer, more information-dense tokens) — Phase 3 Lesson 6's entropy-as-compression-limit intuition applies directly: a well-tokenized corpus, on average, requires fewer tokens per unit of "information" (in the Shannon sense) than a poorly-chosen tokenization scheme.

### Vocabulary size as a genuine hyperparameter tradeoff
Larger vocabulary → shorter average sequences (more information per token) but a larger embedding matrix (Lesson 2, this phase — the embedding matrix is $|\text{vocab}| \times d_{model}$, so vocabulary size directly and linearly affects one of the model's largest parameter blocks) and a larger, more expensive final softmax layer over the vocabulary. Smaller vocabulary → longer sequences (worse for attention's $O(n^2)$ cost, Phase 5 Lesson 6) but a smaller embedding/output layer. Real LLM vocabularies (commonly 32,000-256,000+ tokens in 2026) represent an empirically-tuned balance point.

### Tokenization efficiency across languages (a real, documented fairness/cost concern)
Because BPE vocabularies are learned primarily from training-corpus frequency statistics (historically dominated by English-heavy web text, though increasingly multilingual-balanced in modern training runs), languages underrepresented in training data often require systematically *more tokens* to represent the same semantic content — directly increasing API cost (billed per token) and effectively shrinking usable context length for those languages, a genuine, quantifiable, and actively-being-addressed disparity in current LLM tooling.

---

## 4. Algorithm — Byte-Pair Encoding Training (fully specified)

```
GIVEN a training corpus and a target vocabulary size V:
1. INITIALIZE vocabulary as the set of individual bytes/characters present in the corpus
2. REPRESENT the corpus as sequences of these initial units
3. WHILE current vocabulary size < V:
     a. COUNT frequency of every adjacent pair of tokens across the ENTIRE corpus
     b. FIND the most frequent pair (a, b)                       -- O(corpus length) per iteration, naively
     c. MERGE every occurrence of (a, b) into a new single token "ab"
     d. ADD "ab" to the vocabulary
4. RETURN the final vocabulary and the ordered list of merge rules (needed to tokenize NEW text later)

TOKENIZING NEW TEXT (at inference time):
1. Start with the text split into individual bytes/characters
2. APPLY the learned merge rules IN THE SAME ORDER they were learned during training
   (greedily merge whichever learned pair appears, following the training-time priority order)
3. RETURN the final sequence of tokens (vocabulary IDs)
```
The naive frequency-counting step is $O(n)$ per iteration (corpus length $n$), and with $V$ merge iterations, naive BPE training is $O(nV)$ — real implementations use a priority queue (a **heap**, Phase 1 Lesson 4) to efficiently track and update the most-frequent-pair candidate without a full corpus rescan at every single iteration, a direct, practical application of that lesson's data structures to a genuinely large-scale problem.

---

## 5. Python Implementation

```python
"""tokenization_core.py — a from-scratch, simplified BPE tokenizer"""
from collections import Counter, defaultdict


def get_pair_counts(corpus_tokens: list[tuple[str, ...]]) -> Counter:
    """corpus_tokens: a list of (word represented as a tuple of characters/subwords, frequency) is
    typically used in practice; here we simplify to a flat list of tokenized 'words'."""
    pairs = Counter()
    for word in corpus_tokens:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += 1
    return pairs


def merge_pair(corpus_tokens: list[tuple[str, ...]], pair: tuple[str, str]) -> list[tuple[str, ...]]:
    merged_corpus = []
    for word in corpus_tokens:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])   # merge the pair into one token
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged_corpus.append(tuple(new_word))
    return merged_corpus


def train_bpe(text: str, num_merges: int = 20) -> tuple[list[tuple[str, str]], set[str]]:
    words = text.split()
    corpus_tokens = [tuple(word) + ("</w>",) for word in words]   # </w> marks word boundaries
    vocabulary = set(ch for word in corpus_tokens for ch in word)
    merge_rules = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(corpus_tokens)
        if not pair_counts:
            break
        best_pair = pair_counts.most_common(1)[0][0]     # THE greedy choice (Section 3)
        corpus_tokens = merge_pair(corpus_tokens, best_pair)
        vocabulary.add(best_pair[0] + best_pair[1])
        merge_rules.append(best_pair)

    return merge_rules, vocabulary


def tokenize_with_bpe(word: str, merge_rules: list[tuple[str, str]]) -> list[str]:
    tokens = list(word) + ["</w>"]
    for pair in merge_rules:                              # apply merges IN LEARNED ORDER
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


if __name__ == "__main__":
    corpus = "low lower lowest new newer newest wide wider widest"
    merges, vocab = train_bpe(corpus, num_merges=15)
    print("Learned merges (in order):", merges[:5], "...")
    print("Tokenizing 'lowering':", tokenize_with_bpe("lowering", merges))
    print("Tokenizing 'newest':", tokenize_with_bpe("newest", merges))
```

---

## 6. Build From Scratch

Section 5 already provides a genuine from-scratch BPE trainer/tokenizer. The natural extension is measuring **tokenization efficiency** (tokens per word) — directly illustrating Section 3's cross-language/vocabulary-size tradeoffs:

```python
def average_tokens_per_word(text: str, merge_rules: list[tuple[str, str]]) -> float:
    words = text.split()
    total_tokens = sum(len(tokenize_with_bpe(w, merge_rules)) for w in words)
    return total_tokens / len(words)

# Compare a word seen often during "training" vs. a genuinely novel one
print("Tokens/word for 'newest' (seen in training):",
      len(tokenize_with_bpe("newest", merges)))
print("Tokens/word for 'unbelievability' (never seen):",
      len(tokenize_with_bpe("unbelievability", merges)))
# Expect: the UNSEEN word decomposes into MANY MORE, smaller tokens -- BPE's graceful degradation in action
```

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `train_bpe`/`tokenize_with_bpe` | Hugging Face `tokenizers` library (Rust-backed, extremely fast) — used to train/apply BPE, WordPiece, or Unigram tokenizers at real corpus scale |
| Word-boundary-marked character pairs | `tiktoken` (OpenAI's byte-level BPE implementation) — operates directly on UTF-8 bytes, handling any Unicode input robustly (Section 2) |
| Naive $O(n)$-per-iteration pair counting | Priority-queue-based incremental counting (Phase 1 Lesson 4's heap), avoiding full corpus rescans every merge iteration |

---

## 8. Visual Explanations

**BPE merge iterations building up a vocabulary:**
```
Initial:  l o w </w>   l o w e r </w>   l o w e s t </w>   ...
Merge 1 (most frequent pair, e.g. "l","o"): lo w </w>   lo w e r </w>   lo w e s t </w>
Merge 2 ("lo","w"):                          low </w>    low e r </w>    low e s t </w>
Merge 3 ("e","r"):                           low </w>    low er </w>     low e s t </w>
   ... (continues until target vocabulary size reached) ...
Final: "low" becomes a SINGLE token (frequent), "lowest" might stay as "low"+"est" (less frequent combination)
```

**Tokenization efficiency across a rare vs. common word:**
```
"the"        -> [the]                       (1 token -- extremely common, merged fully)
"tokenization" -> [token, ization]           (2 tokens -- moderately common subwords)
"antidisestablishmentarianism" -> [anti, dis, establish, ment, arian, ism]  (6 tokens -- rare, decomposes heavily)
```

---

## 9. Practical Examples

**Simple:** train a tiny BPE tokenizer on a small text corpus and inspect the first 10 learned merges.
**Medium:** tokenize both a common word and a deliberately rare/invented word using your trained tokenizer, and compare the resulting token counts.
**Real-world:** use `tiktoken` (or Hugging Face `tokenizers`) to tokenize the same paragraph of text in English versus a non-Latin-script language (e.g., Arabic, given your bilingual French/Arabic context), and compare the token counts — directly, empirically demonstrating the cross-language tokenization efficiency disparity from Section 3.

---

## 10. Real Industry Use Cases

- **Every production LLM API** (OpenAI, Anthropic, and others): bills usage per token, making tokenization efficiency a direct, measurable cost factor — understanding tokenization is directly relevant to estimating/optimizing API costs for any LLM application you build.
- **GPT-family models**: use byte-level BPE (`tiktoken`), guaranteeing robust handling of arbitrary Unicode input, code, and even malformed text.
- **BERT and encoder models**: use WordPiece, a closely related but distinct subword algorithm.
- **Multilingual model development**: tokenizer vocabulary design (ensuring balanced representation across languages) is an active, ongoing area of practical effort to address the cross-language efficiency disparity (Section 3).

---

## 11. Common Mistakes

- Assuming tokens correspond to words — leads to systematically wrong intuitions about model context limits, cost estimation, and why LLMs struggle with character-level tasks (counting letters, reversing strings) that seem trivial to humans but require reasoning *across* opaque subword token boundaries.
- Ignoring cross-language tokenization disparities when estimating cost/context budget for multilingual applications.
- Manually reimplementing tokenization logic that doesn't exactly match the target model's actual tokenizer — even small mismatches (different merge rules, different special token handling) silently produce different token IDs than the model was trained on, degrading performance unpredictably.
- Forgetting special tokens (beginning-of-sequence, end-of-sequence, padding) that real tokenizers add automatically — omitting them when constructing inputs manually can break a pretrained model's expected input format.

---

## 12. Best Practices (2026)

- Always use the exact tokenizer that ships with/matches your target pretrained model (via Hugging Face `AutoTokenizer.from_pretrained(...)`) rather than a custom or approximate tokenization scheme.
- When estimating API costs or context budgets, tokenize representative sample text with the actual target tokenizer rather than assuming a fixed words-to-tokens ratio (especially critical for non-English content).
- For genuinely novel domains (highly specialized technical vocabulary, code, or a new language not well-represented in existing tokenizers), consider training a custom tokenizer/vocabulary extension rather than relying on an ill-fitting general-purpose one.
- Be aware that certain tokenization quirks (e.g., how numbers are split into tokens) directly affect an LLM's arithmetic reasoning reliability — a known, documented practical limitation traceable directly to tokenization design choices.

---

## 13. Exercises

**Easy:** Train a small BPE tokenizer (Section 5) on a short corpus and manually trace the first 3 merge decisions, confirming they match the most-frequent-pair criterion.
**Medium:** Compare tokenization output (via Hugging Face `tokenizers` or `tiktoken`) for the same sentence in two different languages, reporting the token-count ratio.
**Hard:** Implement a priority-queue-based (heap, Phase 1 Lesson 4) BPE trainer that avoids the naive full-corpus rescan at every merge iteration, and empirically benchmark its speed against the naive Section 5 implementation on a larger corpus.
**Mathematical:** Using Phase 3 Lesson 6's entropy framework, compute the empirical entropy (bits per character) of a text corpus before and after BPE tokenization, and discuss the relationship to compression efficiency.
**Coding:** Extend the Section 5 BPE implementation to byte-level operation (working on UTF-8 byte sequences rather than characters) and verify it can tokenize text containing emoji/non-Latin scripts without any "unknown" fallback.

---

## 14. Mini Project

Build a **tokenization cost/efficiency analysis tool**: using a real tokenizer (`tiktoken` or Hugging Face `AutoTokenizer`), tokenize a substantial parallel corpus of the same content in French, Arabic, and English (directly relevant to your bilingual background), compute and compare tokens-per-word and tokens-per-character ratios across languages, estimate the practical cost/context-length implications for an LLM application serving all three languages, and write a short report on any disparities found and their real-world consequences for API cost and effective context budget.

---

## 15. Interview Preparation

- Explain the Byte-Pair Encoding algorithm and why it's preferred over pure word-level or character-level tokenization.
- Why does byte-level BPE (as used in GPT-family tokenizers) never need an "unknown token" fallback?
- How does vocabulary size affect the tradeoff between sequence length and embedding/output layer size?
- Why might the same piece of text require different numbers of tokens across different languages, and what practical consequences does this have?

---

## 16. Summary

Tokenization converts raw text into the discrete unit sequences that Transformer architectures (Phase 5) actually process, with subword algorithms (BPE, WordPiece) striking a practical balance between word-level tokenization's fragility on rare words and character-level tokenization's excessive sequence length — BPE's greedy, frequency-driven merge procedure (a direct application of Phase 1 Lesson 4's greedy algorithms and Phase 3 Lesson 6's information-theoretic compression intuition) is the dominant scheme underlying essentially every production LLM in 2026. Understanding tokenization concretely explains several otherwise-mysterious LLM behaviors (arithmetic quirks, character-counting failures, cross-language cost disparities) that will recur throughout the rest of this phase.

---

## 17. References

- Sennrich, Haddow, Birch — "Neural Machine Translation of Rare Words with Subword Units" (2015, the paper that introduced BPE to NLP)
- Radford et al. — "Language Models are Unsupervised Multitask Learners" (2019, GPT-2, establishing byte-level BPE)
- Hugging Face — `tokenizers` library documentation
- OpenAI — `tiktoken` library and documentation
