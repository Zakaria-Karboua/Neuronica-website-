# Phase 5 · Lesson 2 — Backpropagation

> Prerequisite: Neural Networks (Lesson 1), Phase 3 Calculus (Lesson 2's chain rule and autodiff engine)

---

## 1. Introduction

### What is backpropagation?
The algorithm for efficiently computing the gradient of a loss function with respect to every parameter in a multi-layer neural network, by applying the chain rule (Phase 3 Lesson 2) systematically backward through the network's computation graph. It is not a separate learning algorithm from gradient descent (Phase 3 Lesson 5) — it is specifically the *gradient computation* step; gradient descent (or Adam, etc.) is what then *uses* that gradient to update weights.

### Why does it exist?
Naively computing the gradient of a loss with respect to each of a network's millions of parameters via finite differences (Phase 3 Lesson 2's `numerical_gradient`) would require one full forward pass *per parameter* — computationally infeasible at any real scale. Backpropagation computes *all* parameter gradients in a single backward pass, at a cost comparable to one additional forward pass — an efficiency breakthrough with no viable substitute at scale.

### Historical background
The mathematical technique (reverse-mode automatic differentiation) predates its neural network application, but Rumelhart, Hinton, and Williams's 1986 paper "Learning representations by back-propagating errors" is what popularized it specifically for training multi-layer networks, directly enabling the practical use of hidden layers (which Minsky & Papert's 1969 critique of perceptrons had shown were necessary but, at the time, impractical to train).

### Real-world motivation
Every `.backward()` call you'll make in PyTorch (Lesson 8, this phase) executes exactly this algorithm. Understanding it by hand — as you'll do in this lesson's mini project — is what separates treating deep learning frameworks as a black box from genuinely understanding what happens when a model trains, including *why* certain architectures (Lessons 3-7) were specifically designed to keep backpropagation numerically well-behaved.

---

## 2. Theory

### The computational graph view (directly extending Phase 3 Lesson 2)
A neural network's forward pass is a computation graph: inputs and parameters are leaf nodes, each operation (matrix multiply, add bias, apply activation) is an internal node, and the loss is the final output node. Backpropagation is reverse-mode automatic differentiation applied to this specific graph structure — Phase 3 Lesson 2's `Value` autodiff engine, scaled up to full weight matrices instead of scalars.

### Forward pass vs. backward pass (what's cached and why)
The **forward pass** computes and *caches* every intermediate value (pre-activations $z^{(l)}$, activations $h^{(l)}$) — these cached values are required during the backward pass to compute local derivatives (e.g., $\text{ReLU}'(z)$ needs to know the sign of $z$, which was only computed during the forward pass). This caching is *why* training memory usage scales with network depth and batch size — a genuinely important practical constraint (motivating techniques like gradient checkpointing, which trades recomputation for memory).

### Layer-by-layer gradient flow
For a layer $z^{(l)} = W^{(l)}h^{(l-1)} + b^{(l)}$, $h^{(l)} = \phi(z^{(l)})$, given the gradient of the loss with respect to this layer's output $\delta^{(l)} = \partial L/\partial h^{(l)}$ (received from the layer *above*, since we work backward), backpropagation computes:
$$
\frac{\partial L}{\partial z^{(l)}} = \delta^{(l)} \odot \phi'(z^{(l)}), \qquad
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot (h^{(l-1)})^T, \qquad
\frac{\partial L}{\partial h^{(l-1)}} = (W^{(l)})^T \cdot \frac{\partial L}{\partial z^{(l)}}
$$
The last term is exactly what gets passed *down* to the next (earlier) layer as its incoming $\delta^{(l-1)}$ — the recursive structure that gives backpropagation its name and its efficiency.

---

## 3. Mathematical Foundations

### Full derivation for a 2-layer network with softmax + cross-entropy output
Given $z^{(1)} = W^{(1)}x+b^{(1)}$, $h^{(1)}=\text{ReLU}(z^{(1)})$, $z^{(2)}=W^{(2)}h^{(1)}+b^{(2)}$, $\hat y=\text{softmax}(z^{(2)})$, and cross-entropy loss $L=-\sum_k y_k\log\hat y_k$ (Phase 3 Lesson 6):

**Step 1 — output layer gradient (a famously clean result, Phase 4 Lesson 1 previewed this):**
$$
\frac{\partial L}{\partial z^{(2)}} = \hat y - y
$$
This simplification—the softmax-cross-entropy combination's Jacobian collapsing to just "predicted minus true"—is a specific, elegant consequence of how the softmax Jacobian and the cross-entropy gradient's terms cancel; it is *why* this particular pairing (softmax output + cross-entropy loss) is used almost universally for classification, rather than an arbitrary convention.

**Step 2 — propagate to $W^{(2)}, b^{(2)}$:**
$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial z^{(2)}}(h^{(1)})^T, \qquad \frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial z^{(2)}}
$$

**Step 3 — propagate backward through the ReLU layer:**
$$
\frac{\partial L}{\partial h^{(1)}} = (W^{(2)})^T\frac{\partial L}{\partial z^{(2)}}, \qquad
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial h^{(1)}} \odot \mathbb{1}[z^{(1)}>0]
$$
(the ReLU derivative is exactly the indicator of whether the pre-activation was positive — Phase 5 Lesson 1's `relu_derivative`).

**Step 4 — propagate to $W^{(1)}, b^{(1)}$:**
$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(1)}}x^T, \qquad \frac{\partial L}{\partial b^{(1)}} = \frac{\partial L}{\partial z^{(1)}}
$$

### Computational complexity of backpropagation
For a network with $P$ total parameters, one backward pass costs $O(P)$ — the *same order* as one forward pass (each parameter's gradient requires $O(1)$ additional work beyond the matrix multiplications already done forward) — a remarkable efficiency result formalized generally as the fact that reverse-mode autodiff's cost is a small constant multiple of the original function's cost, *regardless* of how many inputs/parameters exist (contrast this with forward-mode autodiff or finite differences, both $O(P)$ times *more expensive* than a single forward pass).

---

## 4. Algorithm — Full Backpropagation (matrix form, generalized to $L$ layers)

```
FORWARD PASS:
  h^(0) = x
  FOR l = 1 to L:
      z^(l) = W^(l) @ h^(l-1) + b^(l)
      h^(l) = phi_l(z^(l))
      CACHE z^(l), h^(l-1)   # needed for backward pass

COMPUTE LOSS: L = loss_fn(h^(L), y)

BACKWARD PASS:
  delta = dL/dh^(L)                          # gradient of loss w.r.t. final output
  (special case: for softmax+cross-entropy, delta = y_hat - y directly, skipping the softmax Jacobian)
  FOR l = L down to 1:
      dz = delta * phi_l'(z^(l))              # elementwise, chain rule through the activation
      dW^(l) = dz @ h^(l-1).T                 # gradient w.r.t. this layer's weights
      db^(l) = sum(dz, over batch dimension)  # gradient w.r.t. this layer's biases
      delta = W^(l).T @ dz                    # propagate gradient DOWN to the previous layer
RETURN all dW^(l), db^(l)
```

---

## 5. Python Implementation

```python
"""backpropagation_core.py — full training loop for the Lesson 1 MLP, using ONLY NumPy"""
import numpy as np


def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLPTrainer:
    def __init__(self, layer_sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2/layer_sizes[i]), (layer_sizes[i], layer_sizes[i+1]))
                  for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def forward(self, X):
        cache = {"h": [X], "z": []}
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            cache["z"].append(z)
            h = softmax(z) if i == len(self.W) - 1 else relu(z)
            cache["h"].append(h)
        return h, cache

    def backward(self, y_true_onehot: np.ndarray, cache: dict) -> tuple[list, list]:
        n = y_true_onehot.shape[0]
        dW, db = [None] * len(self.W), [None] * len(self.b)

        delta = (cache["h"][-1] - y_true_onehot) / n     # Section 3's clean softmax+CE gradient, batch-averaged
        for l in reversed(range(len(self.W))):
            h_prev = cache["h"][l]
            dW[l] = h_prev.T @ delta
            db[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_derivative(cache["z"][l-1])
        return dW, db

    def train_step(self, X, y_onehot, lr: float = 0.1):
        y_hat, cache = self.forward(X)
        dW, db = self.backward(y_onehot, cache)
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]
        loss = -np.mean(np.sum(y_onehot * np.log(y_hat + 1e-12), axis=1))
        return loss


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, n_features, n_classes = 1000, 4, 3
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, n)
    y_onehot = np.eye(n_classes)[y]

    model = MLPTrainer([n_features, 16, 8, n_classes])
    for epoch in range(200):
        loss = model.train_step(X, y_onehot, lr=0.5)
        if epoch % 40 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}")
```

---

## 6. Build From Scratch

**Gradient checking (Phase 3 Lesson 2's technique, applied here to validate the full MLP backward pass):**
```python
import numpy as np

def gradient_check_mlp(model: "MLPTrainer", X: np.ndarray, y_onehot: np.ndarray, epsilon: float = 1e-5):
    y_hat, cache = model.forward(X)
    dW_analytic, db_analytic = model.backward(y_onehot, cache)

    def loss_fn(model):
        y_hat, _ = model.forward(X)
        return -np.mean(np.sum(y_onehot * np.log(y_hat + 1e-12), axis=1))

    # Check just ONE weight matrix's gradient (checking all would be very slow -- O(P) forward passes)
    W = model.W[0]
    dW_numeric = np.zeros_like(W)
    for i in range(min(3, W.shape[0])):          # spot-check a few entries, not the whole matrix
        for j in range(min(3, W.shape[1])):
            original = W[i, j]
            W[i, j] = original + epsilon
            loss_plus = loss_fn(model)
            W[i, j] = original - epsilon
            loss_minus = loss_fn(model)
            W[i, j] = original
            dW_numeric[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    rel_error = np.abs(dW_analytic[0][:3, :3] - dW_numeric[:3, :3]) / (
        np.abs(dW_analytic[0][:3, :3]) + np.abs(dW_numeric[:3, :3]) + 1e-12
    )
    return rel_error.max()   # should be TINY (< 1e-4) if backprop is implemented correctly
```
This is the **mandatory** validation step whenever you write custom backpropagation code (as emphasized in Phase 3 Lesson 2) — silent backprop bugs produce a network that trains "successfully" (loss decreases) but toward a subtly wrong objective, an insidious failure mode that gradient checking catches immediately.

---

## 7. Library Implementation (Comparison)

| From scratch (`MLPTrainer`) | PyTorch |
|---|---|
| Manual forward/backward/update loop | `loss.backward()` + `optimizer.step()` — automatic, handles arbitrary architectures without manual derivation |
| Manual caching of `z`/`h` | Autograd's computation graph handles this automatically, with memory-efficient graph pruning |
| Hand-derived softmax+CE gradient shortcut | `torch.nn.CrossEntropyLoss` implements this exact fused, numerically stable shortcut internally |
| Manual gradient checking | `torch.autograd.gradcheck` — the standard tool for validating any custom `autograd.Function` |

---

## 8. Visual Explanations

**Backward pass as reverse traversal of the forward computation graph:**
```
FORWARD:   x -> [W1,b1] -> z1 -> [ReLU] -> h1 -> [W2,b2] -> z2 -> [softmax] -> y_hat -> [loss]
BACKWARD:  dx <- [W1,b1] <- dz1 <- [ReLU'] <- dh1 <- [W2,b2] <- dz2 <- [softmax+CE shortcut] <- y_hat, y
           (gradients flow in EXACTLY the reverse order of the forward computation)
```

**Gradient checking as a correctness safety net:**
```
Analytic gradient (from backprop) ──┐
                                      ├──▶ compare (should match within ~1e-5) ──▶ PASS / FAIL
Numeric gradient (finite difference)─┘
   (numeric is SLOW but nearly assumption-free -- the trusted "ground truth" for validation)
```

---

## 9. Practical Examples

**Simple:** manually derive and verify, on paper, the gradient of a single linear layer with squared-error loss.
**Medium:** implement the full backward pass for the Section 5 MLP and train it on a synthetic multi-class classification dataset, plotting the loss curve.
**Real-world:** implement gradient checking (Section 6) on your from-scratch MLP before trusting its training results — directly rehearsing the debugging discipline required whenever you implement any custom layer/loss in PyTorch (Lesson 8) that isn't already provided by the framework.

---

## 10. Real Industry Use Cases

- **Every deep learning framework's core engineering**: PyTorch/TensorFlow/JAX's entire value proposition is implementing backpropagation correctly, efficiently, and generally (for arbitrary user-defined architectures) — this lesson is literally what those frameworks automate.
- **Custom research architectures**: whenever a researcher implements a novel layer type not yet in a framework's standard library, they must ensure its backward pass is correctly registered (or rely on the framework's automatic differentiation to derive it, which works for any composition of standard differentiable operations).
- **Numerical stability engineering**: real frameworks fuse operations (like softmax+cross-entropy, Section 3) specifically to exploit these clean gradient simplifications for both speed and numerical stability — understanding *why* directly explains framework design choices you'll encounter in Lesson 8.

---

## 11. Common Mistakes

- Forgetting to average (or correctly sum) gradients across the batch dimension — a very common shape/scaling bug when implementing backprop by hand.
- Re-deriving the softmax Jacobian explicitly instead of using the clean $\hat y - y$ shortcut (Section 3) — not wrong, just needlessly complex and more error-prone.
- Skipping gradient checking when implementing any custom backward pass — the single highest-leverage debugging step for silent (loss-still-decreases) backprop bugs.
- Confusing the *order* of matrix multiplication/transposition in the weight-gradient formulas — a frequent source of shape-mismatch errors that are easy to "fix" by blindly transposing until shapes match, without understanding why, which usually means the fix is actually wrong.

---

## 12. Best Practices (2026)

- Always gradient-check any custom backward-pass implementation against numerical differentiation before trusting training results.
- Rely on framework autograd (PyTorch, Lesson 8) for standard architectures — hand-implementing backpropagation, as done in this lesson, is for understanding, not for production use, where framework-provided autodiff is faster, more general, and far less error-prone.
- Understand which operation fusions (softmax+cross-entropy, layer-norm+residual) your framework provides for numerical stability, and use them rather than composing the individual operations manually.
- When debugging a training run that "isn't learning," gradient checking (or inspecting gradient norms per layer) should be an early diagnostic step, not an afterthought.

---

## 13. Exercises

**Easy:** By hand, derive $\partial L/\partial W$ for a single linear layer $\hat y = Wx+b$ under squared-error loss.
**Medium:** Implement the full Section 5 training loop and train it to convergence on a synthetic dataset, verifying the loss decreases monotonically (or nearly so, given mini-batch noise).
**Hard:** Implement gradient checking (Section 6) for the FULL network (all layers, not just a spot-check), and deliberately introduce a bug (e.g., forget the ReLU derivative in the backward pass) to confirm gradient checking correctly catches it.
**Mathematical:** Derive the softmax+cross-entropy gradient simplification ($\hat y - y$) explicitly, starting from the softmax Jacobian and the cross-entropy loss's own gradient.
**Coding:** Extend the Section 5 `MLPTrainer` to support an arbitrary number of layers and activation function choices (ReLU vs. tanh), verifying correctness via gradient checking for each activation choice.

---

## 14. Mini Project

**Fully train, from scratch, the actuarial-classification MLP architecture designed in Lesson 1's mini project**: implement the complete forward + backward + parameter update loop (Section 5), validate correctness via gradient checking (Section 6) before trusting any results, train on your actual (or synthetic) actuarial dataset, plot the training loss curve, and compare final performance against the XGBoost baseline from Phase 4 Lesson 1 — a complete, honest, from-first-principles comparison between a hand-built neural network and gradient boosting on tabular data.

---

## 15. Interview Preparation

- Derive the backpropagation update for a 2-layer neural network with ReLU activations and cross-entropy loss.
- Why is backpropagation's computational cost only a small constant factor more than a single forward pass, regardless of the number of parameters?
- What is gradient checking, and why is it an essential debugging step for custom backward-pass implementations?
- Explain what's cached during the forward pass and why it's needed during the backward pass.

---

## 16. Summary

Backpropagation is reverse-mode automatic differentiation (Phase 3 Lesson 2) applied systematically to a neural network's layered computation graph: each layer's local gradient is computed and multiplied into an accumulating "delta" signal flowing backward, with the softmax+cross-entropy combination yielding a famously clean $\hat y - y$ simplification that explains its near-universal use for classification. Implementing this by hand — and rigorously gradient-checking the result — is the single most direct way to demystify what every `.backward()` call in PyTorch (Lesson 8) is actually computing, and is essential preparation for understanding why specific architectural choices in CNNs, RNNs, and Transformers (Lessons 3-7) were designed the way they were, largely to keep this exact gradient-flow process numerically well-behaved at depth.

---

## 17. References

- Rumelhart, D., Hinton, G., Williams, R. — "Learning Representations by Back-Propagating Errors" (1986)
- Nielsen, M. — *Neural Networks and Deep Learning* (free online, exceptionally clear backpropagation derivation)
- Karpathy, A. — "Yes You Should Understand Backprop" (essay) and the "micrograd" series (Phase 3 Lesson 2)
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 6 (backpropagation formalized in full generality)
