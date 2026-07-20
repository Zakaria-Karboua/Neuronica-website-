# Phase 1 · Lesson 4 — Data Structures & Algorithms

> Prerequisite: Lessons 1–3

---

## 1. Introduction

### What is this topic?
The study of how to organize data (structures) and manipulate it efficiently (algorithms), measured formally through time/space complexity. This is CS-fundamentals territory, but every ML/AI system you build rests on it: a vector database's ANN index is a graph/tree structure; a tokenizer's BPE merge step is a priority queue; attention's KV-cache is a specialized array structure; a training data loader's shuffling is a randomized algorithm.

### Why does it exist?
As data scales, the *asymptotic* behavior of your code — not its constant-factor speed on a laptop — determines whether it works on real datasets. An $O(n^2)$ algorithm that's "fast enough" on 1,000 rows will not survive 100 million rows, a routine scale for production ML data.

### Historical background
Formal algorithm analysis traces to Knuth's *The Art of Computer Programming* (1968–). Big-O notation was popularized by Donald Knuth and rigorously connects to complexity theory (P vs NP), though for AI engineering we mostly stay in the practical realm of designing/choosing data structures.

### Real-world motivation
Interview processes at every major AI lab still test DSA fundamentals — but more importantly, choosing `set` over `list` for membership checks, or a heap over full sorting for top-k retrieval, is the difference between a data pipeline that runs in seconds vs hours.

---

## 2. Theory

### Core structures
| Structure | Access | Search | Insert | Delete | Notes |
|---|---|---|---|---|---|
| Array/list | $O(1)$ | $O(n)$ | $O(1)$ amortized (end) | $O(n)$ | contiguous memory, cache-friendly |
| Linked list | $O(n)$ | $O(n)$ | $O(1)$ (given node) | $O(1)$ (given node) | pointer-chasing, poor cache locality |
| Hash table (dict/set) | — | $O(1)$ avg | $O(1)$ avg | $O(1)$ avg | $O(n)$ worst case under collisions |
| Binary Search Tree (balanced) | — | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | maintains sorted order |
| Heap (priority queue) | — | $O(n)$ | $O(\log n)$ | $O(\log n)$ pop-min/max | efficient top-k / scheduling |
| Graph (adjacency list) | — | $O(V+E)$ traversal | — | — | models relations: knowledge graphs, dependency DAGs |

### Core algorithmic paradigms
- **Divide and conquer**: split, solve recursively, combine (merge sort, quicksort).
- **Dynamic programming**: solve overlapping subproblems once, cache results (this is memoization from Lesson 2, generalized).
- **Greedy algorithms**: make the locally optimal choice at each step (works when the problem has the "greedy choice property," e.g. Huffman coding, which underlies tokenizer vocabulary construction ideas).
- **Graph traversal**: BFS (shortest path, unweighted), DFS (connectivity, cycle detection), Dijkstra (weighted shortest path).

---

## 3. Mathematical Foundations

### Big-O, Big-Ω, Big-Θ
$$
f(n) = \Theta(g(n)) \iff f(n) = O(g(n)) \text{ and } f(n) = \Omega(g(n))
$$
$\Theta$ gives a *tight* bound (both upper and lower), which is what you should aim to reason about, not just worst-case $O$.

### Master Theorem (for divide-and-conquer recurrences)
For $T(n) = aT(n/b) + f(n)$:
$$
T(n) =
\begin{cases}
\Theta(n^{\log_b a}) & \text{if } f(n) = O(n^{\log_b a - \epsilon}) \\
\Theta(n^{\log_b a}\log n) & \text{if } f(n) = \Theta(n^{\log_b a}) \\
\Theta(f(n)) & \text{if } f(n) = \Omega(n^{\log_b a + \epsilon})
\end{cases}
$$
Applied to merge sort ($a=2, b=2, f(n)=n$): $\log_b a = 1$, $f(n) = \Theta(n^1)$ → case 2 → $T(n) = \Theta(n \log n)$.

### Expected complexity of hashing under uniform hashing assumption
With load factor $\alpha = n/m$ (items/buckets), expected chain length under separate chaining is $\Theta(1 + \alpha)$ — this is *why* hash tables resize once $\alpha$ exceeds a threshold (~0.75), keeping expected operations $O(1)$.

---

## 4. Algorithm — Binary Search (worked example)

**Pseudocode:**
```
BINARY_SEARCH(A[0..n-1], target):
    low, high = 0, n - 1
    while low <= high:
        mid = low + (high - low) // 2
        if A[mid] == target: return mid
        elif A[mid] < target: low = mid + 1
        else: high = mid - 1
    return -1   # not found
```
**Complexity:** Each iteration halves the search space, so after $k$ iterations the space is $n/2^k$; the loop ends when $n/2^k \le 1 \Rightarrow k = \log_2 n$. Time: $O(\log n)$. Space: $O(1)$ iterative (or $O(\log n)$ if written recursively, due to call stack).

**Correctness invariant:** at every loop iteration, if `target` exists in `A`, it exists within `A[low..high]` — this invariant, stated explicitly, is exactly what interviewers want to hear.

---

## 5. Python Implementation

```python
"""dsa_core.py — clean, typed implementations of core structures/algorithms"""
from __future__ import annotations
import heapq
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


def binary_search(arr: list[int], target: int) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def top_k_frequent(items: list[str], k: int) -> list[str]:
    """Using a min-heap of size k -> O(n log k) instead of full O(n log n) sort."""
    from collections import Counter
    counts = Counter(items)
    return [w for w, _ in heapq.nlargest(k, counts.items(), key=lambda kv: kv[1])]


def bfs_shortest_path(graph: dict[str, list[str]], start: str, goal: str) -> list[str]:
    """Unweighted shortest path — O(V + E)."""
    frontier = deque([[start]])
    visited = {start}
    while frontier:
        path = frontier.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(path + [neighbor])
    return []


class MinHeap(Generic[T]):
    """Thin, explicit wrapper making heap operations self-documenting."""
    def __init__(self) -> None:
        self._data: list[T] = []

    def push(self, item: T) -> None:
        heapq.heappush(self._data, item)

    def pop(self) -> T:
        return heapq.heappop(self._data)

    def __len__(self) -> int:
        return len(self._data)


if __name__ == "__main__":
    print(binary_search([1, 3, 5, 7, 9, 11], 7))          # 3
    print(top_k_frequent(["a", "b", "a", "c", "a", "b"], 2))  # ['a', 'b']
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    print(bfs_shortest_path(graph, "A", "D"))              # ['A', 'B', 'D']
```

---

## 6. Build From Scratch

**Merge sort (divide and conquer, from scratch):**
```python
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: list[int], right: list[int]) -> list[int]:
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
$T(n) = 2T(n/2) + O(n) \Rightarrow \Theta(n \log n)$ by the Master Theorem — matching Python's built-in `sorted()` (Timsort), which is also $O(n \log n)$ worst case but exploits existing runs for $O(n)$ best case on nearly-sorted data.

**A binary min-heap (mirrors `heapq`):**
```python
class BinaryHeap:
    def __init__(self):
        self.data = []

    def push(self, val):
        self.data.append(val)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        self.data[0], self.data[-1] = self.data[-1], self.data[0]
        top = self.data.pop()
        self._sift_down(0)
        return top

    def _sift_up(self, i):
        parent = (i - 1) // 2
        if i > 0 and self.data[i] < self.data[parent]:
            self.data[i], self.data[parent] = self.data[parent], self.data[i]
            self._sift_up(parent)

    def _sift_down(self, i):
        n = len(self.data)
        smallest = i
        for child in (2 * i + 1, 2 * i + 2):
            if child < n and self.data[child] < self.data[smallest]:
                smallest = child
        if smallest != i:
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            self._sift_down(smallest)
```

---

## 7. Library Implementation (Comparison)

| From scratch | Standard library / real world |
|---|---|
| `merge_sort` | `sorted()`/`list.sort()` — Timsort, highly optimized C, adaptive to partial order |
| `BinaryHeap` | `heapq` — C-optimized array-based binary heap |
| `bfs_shortest_path` | `networkx.shortest_path` — production graph library with many algorithms |
| Hand-rolled hash table (Lesson 1) | `dict`/`set` |

---

## 8. Visual Explanations

**Binary search space shrinking:**
```
[1 3 5 7 9 11 13]   target=13
 low─────────high
 mid=7 -> too small -> low = index(9)
       [9 11 13]
        mid=11 -> too small -> low = index(13)
             [13]  found!
```

**Heap as an array (implicit complete binary tree):**
```
Array: [2, 5, 4, 8, 9, 7]
Tree:
           2
         /   \
        5     4
       / \   /
      8   9 7
Parent(i) = (i-1)//2 ; Left(i) = 2i+1 ; Right(i) = 2i+2
```

---

## 9. Practical Examples

**Simple:** find the two numbers in a list summing to a target (two-pointer on sorted array, $O(n \log n)$, or hash-set one-pass, $O(n)$).
**Medium:** implement top-k most similar items given a similarity score list — direct precursor to nearest-neighbor retrieval in RAG (Phase 7).
**Real-world:** deduplicate 50 million patient records by a composite key using a hash-set streaming approach instead of $O(n^2)$ pairwise comparison — directly relevant to cleaning large actuarial/insurance datasets.

---

## 10. Real Industry Use Cases

- **Vector databases (Pinecone, Weaviate, FAISS)**: approximate nearest neighbor search structures (HNSW graphs, IVF trees) are direct extensions of the graph/tree structures here.
- **Tokenizers (BPE)**: use priority queues (heaps) to always merge the most frequent adjacent token pair first.
- **Feature stores / data pipelines** (Netflix, Meta): rely on hash-based joins and deduplication at billion-row scale.
- **LLM inference schedulers (vLLM)**: use priority queues to schedule which requests get GPU time next (continuous batching).

---

## 11. Common Mistakes

- Using nested loops (`for x in a: for y in b: if x==y`) for set intersection — $O(n \cdot m)$ instead of $O(n+m)$ via `set(a) & set(b)`.
- Sorting an entire list ($O(n \log n)$) just to get the top-3 elements when a heap does it in $O(n \log 3) \approx O(n)$.
- Ignoring worst-case behavior of hash tables under adversarial/pathological key distributions (rare but real in security-sensitive contexts).
- Off-by-one errors in binary search bounds — always state and verify the loop invariant.

---

## 12. Best Practices (2026)

- Reach for `collections` (`deque`, `Counter`, `defaultdict`, `OrderedDict`) before hand-rolling structures — they are C-optimized and battle-tested.
- Profile before optimizing — use `timeit`/`cProfile` to confirm the bottleneck is actually algorithmic before micro-optimizing.
- For genuinely large-scale structures (billion-item ANN indices), use specialized libraries (FAISS, HNSWlib) rather than reimplementing — but you must understand the underlying complexity to choose/tune them correctly (Phase 7).
- Know your language's sort stability and complexity guarantees (`sorted()` is stable, $O(n \log n)$ worst case, unlike naive quicksort's $O(n^2)$ worst case).

---

## 13. Exercises

**Easy:** Implement `is_anagram(s1, s2)` in $O(n)$ using a hash-based counter.
**Medium:** Implement quicksort with a random pivot and analyze expected vs worst-case complexity.
**Hard:** Implement Dijkstra's algorithm using a min-heap for weighted shortest path, and explain why it fails with negative edge weights.
**Mathematical:** Derive, via the Master Theorem, the complexity of a recurrence $T(n) = 4T(n/2) + n^2$ (should land in case 2, giving $\Theta(n^2 \log n)$).
**Coding:** Implement an LRU cache with $O(1)$ get/put using a `dict` + doubly linked list (revisit from Lesson 1, now with full DSA vocabulary).

---

## 14. Mini Project

Build a **Top-K Similar Documents Finder**: given a list of text documents represented as pre-computed similarity scores to a query, use a min-heap to efficiently maintain the top-k most similar in $O(n \log k)$, benchmark it against a naive full-sort approach on increasing $n$ (1K, 100K, 10M), and plot/report the empirical divergence matching the theoretical $O(n \log k)$ vs $O(n \log n)$ prediction. This is the exact algorithmic core of a retrieval step in RAG, minus the actual embedding model (Phase 7).

---

## 15. Interview Preparation

- Explain time/space complexity of your favorite sorting algorithm and when you'd choose it over another.
- Why is a hash table $O(1)$ average but $O(n)$ worst case? When does worst case actually happen?
- Design question: how would you find the median of a continuously arriving data stream in better than $O(n \log n)$ per query? (two-heap technique)
- Whiteboard: implement BFS and explain why it — not DFS — guarantees shortest path in unweighted graphs.

---

## 16. Summary

Data structures and algorithms are the substrate every "fancy" AI system quietly depends on: heaps power top-k retrieval and priority scheduling; hash tables power nearly every $O(1)$ lookup in your ML pipeline; graph traversal underlies both classical search and modern vector-index construction; and Big-O reasoning is what lets you predict — before writing a single line — whether your pipeline will survive going from a 10K-row prototype to a 500M-row production dataset.

---

## 17. References

- Cormen, Leiserson, Rivest, Stein — *Introduction to Algorithms (CLRS)*
- Sedgewick & Wayne — *Algorithms* (4th ed.)
- NeetCode / LeetCode — practical pattern-based interview prep
- `heapq`, `collections`, `bisect` official Python docs
