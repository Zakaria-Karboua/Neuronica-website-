# Phase 9 · Project 3 — Crop Health Image Classification Service

> Difficulty: ★★☆☆☆ | Primary phases exercised: 5 (Lessons 1–3, 8), 8 (Lessons 1–5)
> Domain tie-in: SANAVIR precision agriculture

---

## Overview

A CNN-based image classification API detecting crop stress/disease from field-captured leaf images — the first deep-learning project in this sequence, introducing GPU-aware serving, PyTorch (Phase 5 Lesson 8), and transfer learning from a pretrained backbone rather than training from scratch.

---

## System Architecture

```
Image upload ──▶ FastAPI (multipart/form-data) ──▶ Preprocessing (resize/normalize)
                                                              │
                                                              ▼
                                              PyTorch CNN (fine-tuned ResNet backbone)
                                                              │
                                                              ▼
                                        {class, confidence, class_probabilities}
```

Training uses transfer learning: a pretrained ResNet (Phase 5 Lesson 3) with its convolutional base frozen, only the final classification head fine-tuned on a labeled crop-image dataset — directly reusing Phase 5 Lesson 1's weight-initialization/regularization concepts and Lesson 3's hierarchical-feature-learning rationale for why transfer learning works (early layers' edge/texture detectors transfer across visual domains).

---

## Folder Structure

```
crop-health-classifier/
├── src/
│   ├── data/
│   │   ├── dataset.py            # torch Dataset/DataLoader, Phase 5 Lesson 8
│   │   └── augmentation.py       # rotation/flip/color-jitter for small-dataset robustness
│   ├── training/
│   │   ├── train.py              # transfer learning loop, MLflow-tracked
│   │   └── evaluate.py           # per-class precision/recall, confusion matrix
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── image_preprocessing.py
│   └── model/architecture.py
├── tests/
│   ├── unit/test_preprocessing.py
│   ├── unit/test_augmentation.py
│   └── integration/test_api_upload.py
├── Dockerfile                     # CUDA-enabled base image for GPU inference
├── k8s/deployment-gpu.yaml        # GPU resource requests, node selector
└── docs/model_card.md
```

---

## Documentation

Model card documents: source dataset and known class imbalance, per-class performance (not just aggregate accuracy — directly Phase 8 Lesson 10's disaggregated-evaluation principle, since misclassifying a rare-but-severe disease matters more than average accuracy suggests), and explicit guidance that this tool flags *candidates for human review*, not an autonomous treatment-decision system.

---

## Testing

Unit tests for image preprocessing (correct resize/normalization, handling of corrupted/non-image uploads returning a clean 400 error rather than crashing); a golden-image regression test (a fixed set of images with known expected predictions, catching silent model-behavior changes across retraining); GPU-vs-CPU inference consistency test (numerical outputs should match within floating-point tolerance, Phase 1 Lesson 1).

---

## Dockerization

A CUDA-enabled base image (`nvidia/cuda` or PyTorch's official GPU image) for the serving container, still following Phase 8 Lesson 2's multi-stage/non-root/HEALTHCHECK discipline; a separate CPU-only image for CI testing (GPU not available in most CI runners) — the model gracefully falls back to CPU inference when no GPU is detected.

---

## CI/CD

CI runs on CPU (model quality gate uses CPU inference); deployment targets GPU-enabled Kubernetes nodes. Quality gate: per-class F1 (not aggregate accuracy) must not regress below the current production model's per-class F1, for *every* class — directly preventing a retraining from improving common-class accuracy while silently degrading a rare-but-critical class.

---

## Deployment

Kubernetes Deployment with a GPU `nodeSelector`/resource request (Phase 8 Lesson 3's resource-allocation concepts extended to GPU scheduling), sized for the SANAVIR use case's expected upload volume (likely bursty — field workers uploading in batches after a survey pass) — HPA scaling on request queue depth rather than CPU utilization, since GPU inference latency doesn't correlate cleanly with CPU usage.

---

## Monitoring

RED metrics + per-class prediction-distribution tracking (a shifting class distribution over time may indicate either genuine seasonal disease patterns or a drifting/miscalibrated model — worth distinguishing) + GPU utilization/memory metrics (Phase 8 Lesson 8's USE method, applied to the GPU resource specifically).

---

## Evaluation

Per-class precision/recall/F1 with confidence intervals; confusion-matrix review specifically for classes with real agronomic consequence if confused (e.g., two diseases requiring different treatments); qualitative review of a sample of low-confidence predictions to assess whether the model's uncertainty is well-calibrated (Phase 4 Lesson 3/Phase 8 Lesson 10).

---

## Scalability

Batch inference endpoint accepting multiple images per request (amortizing GPU-batch efficiency, Phase 6 Lesson 4's batching-for-throughput logic, applied here to vision rather than LLM inference) for the bulk-upload-after-field-survey use case.

---

## Security Considerations

Upload validation (file-type/size limits preventing resource-exhaustion attacks via oversized images); rate limiting per API key; no user-uploaded images retained beyond the inference request unless explicit opt-in consent for model-improvement data collection is given (a genuine data-governance decision worth documenting explicitly).

---

## Definition of Done

- [ ] Transfer-learning model achieves acceptable per-class F1 on a held-out test set
- [ ] GPU and CPU inference produce numerically consistent results
- [ ] Golden-image regression test passes and is included in CI
- [ ] Deployed to a GPU-enabled Kubernetes node pool with working batch endpoint
- [ ] Per-class performance and calibration documented in the model card
