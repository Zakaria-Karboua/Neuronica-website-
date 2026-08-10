# Phase 5 · Lesson 3 — Convolutional Neural Networks (CNNs)

> Prerequisite: Neural Networks, Backpropagation (Lessons 1–2)

---

## 1. Introduction

### What is a CNN?
A neural network architecture built around the **convolution** operation instead of (or alongside) fully connected layers — specifically designed to exploit the spatial structure of grid-like data (images, and by extension, any locally-structured 1D/2D/3D signal), using far fewer parameters than a fully connected network would require for the same input size.

### Why does it exist?
A fully connected layer processing a modest 224×224 RGB image (a common vision benchmark size) would need $224 \times 224 \times 3 = 150{,}528$ input connections *per neuron* in the first hidden layer — computationally and statistically infeasible at any reasonable width. CNNs exploit two structural priors about images: **locality** (nearby pixels are more related than distant ones) and **translation invariance** (a cat's ear looks like a cat's ear regardless of where it appears in the image) — encoding these priors directly into the architecture rather than hoping a fully connected network learns them from data alone.

### Historical background
The convolution operation for pattern recognition traces to Fukushima's Neocognitron (1980); LeCun's LeNet-5 (1998) demonstrated practical, backpropagation-trained CNNs for digit recognition. The field's decisive turning point was AlexNet (Krizhevsky, Sutskever, Hinton, 2012) winning ImageNet by a wide margin using GPUs and deep convolutional layers — the single event most historians point to as launching the modern deep learning era.

### Real-world motivation
While transformers (Lesson 7, this phase) have partially displaced CNNs in cutting-edge vision research, CNNs remain heavily used in production for their efficiency, and the architectural principles here (local receptive fields, weight sharing, hierarchical feature learning) directly inform how you think about *any* structured-data problem, including some 1D CNN applications to time series (Phase 4 Lesson 6) as an alternative to RNNs/Transformers.

---

## 2. Theory

### The convolution operation
A small learnable **filter/kernel** (e.g., 3×3) slides across the input, computing a weighted sum (dot product) at each position — the same filter's weights are **shared** across every spatial location, directly encoding translation invariance and dramatically reducing parameter count compared to a fully connected layer covering the same receptive field.

### Key architectural components
- **Convolutional layer**: applies multiple filters, each producing a "feature map" detecting a specific pattern (edges, textures, later — object parts).
- **Padding**: adding border pixels (commonly zeros) to control output spatial size — "same" padding preserves input dimensions, "valid" padding shrinks them.
- **Stride**: how many pixels the filter moves per step — larger strides downsample more aggressively.
- **Pooling** (max/average): downsamples feature maps, providing a degree of translation invariance and reducing computation for subsequent layers.
- **Channels**: each layer processes multiple input/output "channels" (e.g., RGB = 3 input channels; each filter simultaneously spans all input channels but produces one output channel).

### Hierarchical feature learning (the core CNN insight)
Stacked convolutional layers learn increasingly abstract features: early layers detect edges/simple textures, middle layers detect parts/patterns, deep layers detect whole objects/semantic concepts — an empirically well-documented phenomenon (visualizable via techniques like activation maximization) that gives CNNs much of their practical power on visual data.

---

## 3. Mathematical Foundations

### The convolution operation, formalized
For a 2D input $X$ and kernel $K$ (size $k \times k$), the output at position $(i,j)$:
$$
(X * K)_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_{i+m,j+n} \cdot K_{m,n}
$$
(Technically this is **cross-correlation**, not true mathematical convolution which flips the kernel — but "convolution" is the near-universal, if technically imprecise, deep learning convention, and since the kernel is learned anyway, the flip is immaterial to the network's expressive power.)

### Output size formula
$$
\text{output size} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$
where $n$ = input size, $p$ = padding, $k$ = kernel size, $s$ = stride — essential for correctly designing an architecture's layer dimensions to match, and a very common source of shape-mismatch bugs when implemented by hand.

### Parameter count comparison (the concrete efficiency argument)
For an input of $C_{in}$ channels producing $C_{out}$ output channels via $k \times k$ filters:
$$
\text{parameters} = C_{in} \times C_{out} \times k \times k + C_{out} \text{ (biases)}
$$
Crucially, this count is **independent of the input's spatial size** (height/width) — unlike a fully connected layer, whose parameter count scales with the *product* of input and output sizes. This is the precise mathematical source of CNNs' parameter efficiency on large images.

### Receptive field growth
The **receptive field** of a unit — the region of the original input that can influence it — grows with network depth. For $L$ stacked $k \times k$ convolutions (stride 1), the receptive field size is:
$$
r_L = 1 + L(k-1)
$$
Pooling/strided convolutions grow the receptive field *multiplicatively* rather than additively, which is precisely why architectures interleave pooling/strided layers — to achieve large receptive fields (seeing "the whole picture") without needing impractically many layers of stride-1 convolutions alone.

### Backpropagation through a convolutional layer (conceptual, extending Lesson 2)
The gradient of the loss with respect to a shared kernel weight is the **sum** of its local gradient contributions across *every spatial position* where that weight was used (since weight sharing means the same weight affects many output positions) — a direct, natural extension of Lesson 2's chain rule, just with an extra summation over spatial positions due to the parameter-sharing structure.

---

## 4. Algorithm — 2D Convolution (naive, fully specified)

```
GIVEN input X (H x W x C_in), kernel K (k x k x C_in x C_out), stride s, padding p:
PAD X with p zeros on each border -> X_padded
output_height = floor((H + 2p - k) / s) + 1
output_width  = floor((W + 2p - k) / s) + 1
INITIALIZE output (output_height x output_width x C_out) to zeros
FOR each output channel c_out:
    FOR each output position (i, j):
        region = X_padded[i*s : i*s+k, j*s : j*s+k, :]     # extract the local k x k x C_in patch
        output[i, j, c_out] = sum(region * K[:,:,:,c_out]) # elementwise multiply + sum = the dot product
RETURN output
```
Complexity: $O(H_{out} \times W_{out} \times k^2 \times C_{in} \times C_{out})$ — naive nested loops are extremely slow in pure Python; real implementations reformulate convolution as a large matrix multiplication (the "im2col" technique) to exploit highly optimized BLAS routines (Phase 3 Lesson 1) or, on GPUs, specialized convolution kernels (cuDNN).

---

## 5. Python Implementation

```python
"""cnn_core.py — naive convolution (educational) + PyTorch equivalent"""
import numpy as np
import torch
import torch.nn as nn


def conv2d_naive(X: np.ndarray, K: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    H, W, C_in = X.shape
    k, _, _, C_out = K.shape
    X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)))
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1
    output = np.zeros((out_h, out_w, C_out))

    for c_out in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                region = X_padded[i*stride:i*stride+k, j*stride:j*stride+k, :]
                output[i, j, c_out] = np.sum(region * K[:, :, :, c_out])
    return output


# --- A real CNN architecture in PyTorch (LeNet-style, for intuition-building) ---
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SimpleCNN()
dummy_input = torch.randn(4, 1, 28, 28)   # batch of 4 grayscale 28x28 images (MNIST-like)
output = model(dummy_input)
print("Output shape:", output.shape)   # (4, 10) -- 4 images, 10 class logits each
```

---

## 6. Build From Scratch

**Max pooling from scratch (to make the downsampling operation fully concrete):**
```python
import numpy as np

def max_pool2d(X: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    H, W, C = X.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w, C))
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, c]
                output[i, j, c] = np.max(region)
    return output
```

**Backward pass through max pooling (only the max element receives gradient — a routing operation):**
```python
def max_pool2d_backward(d_out: np.ndarray, X: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    H, W, C = X.shape
    d_X = np.zeros_like(X)
    out_h, out_w = d_out.shape[:2]
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, c]
                max_idx = np.unravel_index(np.argmax(region), region.shape)   # WHICH position was the max
                d_X[i*stride + max_idx[0], j*stride + max_idx[1], c] += d_out[i, j, c]  # route grad ONLY there
    return d_X
```
This makes explicit a subtlety unique to max pooling's backward pass: unlike a convolution (where every input in the receptive field contributes to the gradient), max pooling's gradient flows **only** to the single input that was the maximum — every other position in that pooling window receives zero gradient for that output position, a direct consequence of the max function's derivative being an indicator on the argmax.

---

## 7. Library Implementation (Comparison)

| From scratch | PyTorch |
|---|---|
| `conv2d_naive` (nested loops, very slow) | `torch.nn.Conv2d` — im2col + BLAS matrix multiplication, or GPU-specific cuDNN kernels; orders of magnitude faster |
| `max_pool2d`/`max_pool2d_backward` | `torch.nn.MaxPool2d` — automatic differentiation handles the argmax-routing backward pass internally |
| Manual architecture wiring | `torch.nn.Sequential`/custom `nn.Module` subclasses — standard, composable architecture definition |

---

## 8. Visual Explanations

**Convolution sliding a 3×3 filter over a padded input:**
```
Input (padded, 5x5, showing one 3x3 window position):    Filter (3x3):     Output (one value):
┌─┬─┬─┬─┬─┐                                               ┌─┬─┬─┐
│0│0│0│0│0│                                                │1│0│-1│
├─┼─┼─┼─┼─┤                                                ├─┼─┼─┤          -> sum(window * filter)
│0│█│█│█│0│  <- 3x3 window being processed                │2│0│-2│              = one output pixel
├─┼─┼─┼─┼─┤                                                ├─┼─┼─┤
│0│█│█│█│0│                                                │1│0│-1│
└─┴─┴─┴─┴─┘                                                └─┴─┴─┘
  (filter SLIDES to the next position for the next output pixel, same weights reused -- weight sharing)
```

**Hierarchical feature learning across CNN depth:**
```
Layer 1 (early): edges, simple gradients
Layer 2-3 (mid):  textures, simple shapes (corners, curves)
Layer 4-5 (deep): object parts (eyes, wheels)
Final layers:     whole-object/semantic concepts (cat, car)
   (receptive field GROWS with depth, enabling increasingly global, abstract feature detection)
```

---

## 9. Practical Examples

**Simple:** implement `conv2d_naive` and verify its output against `torch.nn.functional.conv2d` on a small synthetic input.
**Medium:** build a small LeNet-style CNN (Section 5) and train it on a simple image classification dataset (MNIST or a synthetic equivalent), tracking training/validation loss.
**Real-world:** apply a small 1D CNN (a direct architectural adaptation of this lesson's 2D convolution, sliding along the time axis instead of two spatial axes) to your Brent oil / DZD exchange-rate time series (Phase 4 Lesson 6) as an alternative to ARIMA/XGBoost, comparing performance under the same walk-forward validation discipline.

---

## 10. Real Industry Use Cases

- **Medical imaging** (directly relevant to your cardiology background): CNNs are the historically dominant and still widely deployed architecture for X-ray/MRI/echocardiogram classification and segmentation tasks.
- **Autonomous vehicles**: real-time object detection/segmentation systems (Tesla, Waymo) rely heavily on CNN backbones (sometimes now hybridized with transformer components).
- **Manufacturing/quality control**: visual defect-detection systems in industrial settings are a mature, high-value CNN application.
- **Every modern vision-language model** (Phase 6-7 adjacent): even transformer-dominated architectures like Vision Transformers (ViT) borrow conceptually from CNN principles, and many production systems still use CNN backbones (ResNet, EfficientNet) for their favorable efficiency/accuracy tradeoff.

---

## 11. Common Mistakes

- Miscalculating output spatial dimensions (Section 3's formula) when stacking multiple conv/pooling layers, leading to shape-mismatch errors at the flatten/fully-connected transition.
- Forgetting that max pooling's gradient routes only to the argmax position (Section 6) — a subtlety easy to get wrong when implementing a custom backward pass.
- Using far too few filters/channels for a genuinely complex vision task, leaving the network without enough representational capacity — or the opposite, using enormous channel counts on a small dataset, causing severe overfitting.
- Applying 2D image-CNN intuitions blindly to non-image data without checking that the "locality" and "translation invariance" assumptions actually hold for that data (they often do for time series, but not for arbitrary tabular data with no natural ordering).

---

## 12. Best Practices (2026)

- Use established, pretrained CNN backbones (ResNet, EfficientNet, ConvNeXt) via transfer learning rather than training a CNN from scratch, unless your dataset is genuinely large or your domain is sufficiently unusual that pretrained features transfer poorly.
- Prefer `padding="same"` style layers for straightforward architecture design unless deliberately downsampling spatial dimensions.
- Use batch normalization (or modern alternatives) between convolutional layers to stabilize training — directly addressing Lesson 1's vanishing/exploding gradient concerns in the specific context of deep CNN stacks.
- For 1D sequence/time-series applications, consider 1D CNNs (temporal convolutional networks) as a genuinely competitive, often more parallelizable alternative to RNNs (Lesson 4) for problems with fixed or bounded-length local dependencies.

---

## 13. Exercises

**Easy:** Compute the output spatial size of a 32×32 input after a 3×3 convolution with stride 1 and no padding, then with "same" padding.
**Medium:** Implement `conv2d_naive` (Section 5) and verify it matches `torch.nn.functional.conv2d` on several random small inputs/kernels.
**Hard:** Implement the backward pass (gradient with respect to both the input and the kernel) for a 2D convolutional layer from scratch, and validate it via gradient checking (Phase 5 Lesson 2's technique).
**Mathematical:** Derive the receptive field size formula for $L$ stacked $k\times k$, stride-1 convolutional layers, and verify it by hand for a small concrete example ($L=3$, $k=3$).
**Coding:** Build and train the Section 5 `SimpleCNN` on a real (or synthetic MNIST-like) dataset, and visualize the learned first-layer filters — confirming they resemble edge/gradient detectors, as Section 2's hierarchical feature-learning theory predicts.

---

## 14. Mini Project

Build a **1D CNN time-series forecaster** for your Brent oil / DZD exchange-rate data: design a small stack of 1D convolutional layers (sliding along the time axis) followed by fully connected output layers, train it using proper walk-forward validation (Phase 4 Lesson 6), and compare its performance directly against ARIMA and XGBoost-with-lag-features from that earlier lesson — producing a genuinely comprehensive, three-paradigm (classical statistical, gradient-boosted trees, deep learning) forecasting comparison on the same dataset.

---

## 15. Interview Preparation

- Explain why CNNs use far fewer parameters than fully connected networks for image data, and what structural assumptions make that efficient.
- Derive the output size formula for a convolutional layer given input size, kernel size, stride, and padding.
- What is the receptive field, and how does it grow with network depth?
- Explain the backward pass through a max pooling layer and why the gradient routes only to the maximum element.

---

## 16. Summary

CNNs encode two structural priors about grid-like data — locality and translation invariance — directly into their architecture via weight-shared convolutional filters, achieving dramatic parameter efficiency compared to fully connected networks while enabling hierarchical, increasingly abstract feature learning across depth (edges → textures → parts → objects). The convolution operation's output-size formula, receptive field growth, and max pooling's argmax-routing backward pass are the concrete mechanics underlying every vision architecture from LeNet through modern ResNet/EfficientNet backbones — and the same locality-exploiting principle extends naturally to 1D applications like time-series forecasting, directly complementing Phase 4 Lesson 6's classical and gradient-boosted approaches.

---

## 17. References

- LeCun et al. — "Gradient-Based Learning Applied to Document Recognition" (1998, LeNet-5)
- Krizhevsky, Sutskever, Hinton — "ImageNet Classification with Deep Convolutional Neural Networks" (2012, AlexNet)
- He et al. — "Deep Residual Learning for Image Recognition" (2015, ResNet — also directly relevant to Lesson 7's residual connections)
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 9 (Convolutional Networks)
