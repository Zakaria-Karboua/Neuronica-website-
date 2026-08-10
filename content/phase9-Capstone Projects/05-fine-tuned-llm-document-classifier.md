# Phase 9 · Project 5 — Fine-Tuned LLM Insurance Document Classifier

> Difficulty: ★★★☆☆ | Primary phases exercised: 6 (Lessons 1, 6–8, 10), 8 (Lessons 1–4, 8, 11)

---

## Overview

A service that classifies incoming insurance documents (claims, policy applications, correspondence) into routing categories using a QLoRA-fine-tuned small open-weight LLM — the first project in this sequence involving LLM fine-tuning, quantization, and the Hugging Face ecosystem in a genuine production context rather than a lesson exercise.

---

## System Architecture

```
Document text ──▶ FastAPI ──▶ Fine-tuned LLM (4-bit quantized base + LoRA adapter)
                                          │
                                          ▼
                              {category, confidence, reasoning}
                                          │
                                          ▼
                          Cost tracker (tokens, Phase 8 Lesson 11)
```

Base model loaded once at startup in 4-bit (NF4, Phase 6 Lesson 8) via `bitsandbytes`; LoRA adapter (Phase 6 Lesson 7) loaded on top, trained on a labeled corpus of document (text, category) pairs via `SFTTrainer` (Phase 6 Lesson 10). The adapter — not the full model — is what gets versioned and promoted through the MLflow registry (Phase 8 Lesson 6), since it's a few megabytes versus the base model's gigabytes.

---

## Folder Structure

```
document-classifier-llm/
├── src/
│   ├── training/
│   │   ├── prepare_sft_dataset.py
│   │   ├── train_qlora.py            # Phase 6 Lessons 7–8, 10
│   │   └── evaluate.py               # accuracy + calibration, Phase 8 Lesson 10
│   ├── api/
│   │   ├── main.py                   # loads base model ONCE + hot-swappable adapter
│   │   ├── schemas.py
│   │   └── cost_tracking.py          # Phase 8 Lesson 11
│   └── prompts/classification_prompt.py   # structured-output schema, Phase 7 Lesson 6
├── tests/
│   ├── unit/test_prompt_construction.py
│   └── integration/test_classification_accuracy.py  # fixed eval set, min-accuracy gate
├── Dockerfile.gpu
├── k8s/deployment-gpu.yaml
└── docs/model_card.md
```

---

## Documentation

Model card documents: base model + LoRA rank/target-modules choice and rationale, quantization precision and its measured accuracy cost (Phase 6 Lesson 8's precision/quality tradeoff, quantified for this specific task), and per-category disaggregated accuracy (some categories are rarer and harder — Phase 8 Lesson 10's disaggregated-evaluation discipline).

---

## Testing

Structured-output schema validation tests (Phase 7 Lesson 6 — the model's classification output must always parse as valid JSON matching the category enum, verified via constrained decoding, not hope); a fixed, held-out evaluation set gating any adapter promotion; a LoRA-merge correctness test (Phase 6 Lesson 7 — merged-weights inference must numerically match separate-adapter inference).

---

## Dockerization

GPU-enabled image bundling `transformers`, `peft`, `bitsandbytes`; base model weights baked into the image or mounted from a shared volume/object store (a real tradeoff — baking in makes the image large but self-contained; mounting keeps images small but adds a startup dependency) — this project documents and justifies whichever choice is made.

---

## CI/CD

CI cannot run GPU-dependent fine-tuning on every commit (too expensive) — the pipeline instead: on every PR, runs prompt/schema tests on CPU with a tiny mock model; fine-tuning runs are triggered manually or on a schedule, with their own MLflow-tracked evaluation gate before adapter promotion (Phase 8 Lesson 4's principles, adapted for the reality of expensive GPU training jobs).

---

## Deployment

Single GPU-backed Kubernetes Deployment; adapter updates are hot-swapped by reloading only the LoRA weights (fast) rather than redeploying the multi-gigabyte base model (slow) — a genuine operational advantage of the LoRA architecture (Phase 6 Lesson 7) directly exploited here.

---

## Monitoring

Token cost per request (Phase 8 Lesson 11), classification confidence distribution, and — critically — **calibration on high-confidence classifications** (Phase 8 Lesson 10's safety-specific calibration check), since a misrouted claim silently marked "high confidence" has real downstream cost.

---

## Evaluation

Accuracy + disaggregated per-category performance + calibration report (Phase 8 Lesson 10) + quantization precision/quality tradeoff study (comparing 4-bit vs. 8-bit vs. full-precision accuracy on the same eval set, Phase 6 Lesson 8).

---

## Scalability

Batching concurrent classification requests (Phase 6 Lesson 4's batching-for-throughput) rather than processing one document at a time; a request queue with a short batching window (directly the dynamic-batching pattern referenced in Phase 5 Lesson 2's exercises).

---

## Security Considerations

Documents may contain sensitive personal/medical information — this is the first project in the sequence with a genuine data-privacy dimension: no document content logged in plaintext (only metadata + classification result), encrypted storage for any retained training data, and Phase 8 Lesson 9's input-handling discipline even though this isn't an agentic tool-calling system (untrusted document text still flows into an LLM's context).

---

## Definition of Done

- [ ] QLoRA fine-tuning pipeline runs end to end, tracked in MLflow
- [ ] Classification output passes structured-output schema validation on 100% of a large test batch
- [ ] Disaggregated per-category accuracy and calibration reported in the model card
- [ ] Adapter hot-swap deployment verified without base-model redeployment
- [ ] Cost-per-request tracked and within a documented target budget
