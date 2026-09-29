# 2026-09-29 — Roadmap coverage review

## Context

- **Scope:** user requested review of extra and missing project coverage,
  specifically asking about quantization.
- **Mode:** REVIEW of curriculum; no learning-target implementation.
- **Learning target:** assess prerequisites, explicit coverage, and scope;
  this was not a learner assessment.
- **Starting commit:** `9772fbfa1fe61493cb8ebabba4bb819eaf890a82`.
- **Environment:** local macOS session per supplied context; no runtime or GPU
  environment revalidation. Earlier H100 results are recorded history.

## Work and evidence

- Read `roadmap.md`, learning instructions, current status, both project notes,
  and the latest relevant timeline-session follow-ups.
- Used `rg --files`, `rg -n`, `sed`, `cat`, and `tail` to inspect coverage;
  `git status --short` initially returned no changes; `git rev-parse HEAD`
  supplied the starting commit.
- Only Projects 1 and 2 have Python implementation files in the inspected
  repository inventory. Review primarily concerns planned curriculum.
- Quantization already appears in TensorRT precision coverage, LLM inference,
  course suggestions, and the TurboQuant capstone. A foundational quantization
  project with explicit experiments and exit criteria is absent.
- Latest recorded Project 2 float64 comparison reports both cuDNN modes fail
  the existing tolerance; this session did not reproduce or reinterpret that
  as proof of model-level failure or an accepted optimization.

## Recommendations (not yet adopted)

- Add a quantization laboratory: reference quantize/dequantize; scale/zero point,
  clipping and granularity; calibration and outliers; weight-only and
  weight/activation paths; PTQ versus QAT; packed kernels and overhead;
  memory, latency, and held-out quality evaluation. Extend later to KV cache.
- Begin numerical precision/error reasoning during Project 2; run the main
  quantization lab after basic GPU/custom-op work and before deployment.
- Strengthen explicit training-memory accounting, activation recomputation,
  accumulation/offload tradeoffs, GPU streams/events and transfer overlap,
  tiled attention/online softmax, and topology-aware communication exercises.
- Add cost subject to latency/quality constraints to serving evaluation.
  Sparsity/pruning/distillation can be optional compression extensions.
- Reduce first-pass scope: one main GPU language plus a small comparison;
  small tensor framework before backend/model expansion; basic runtime before
  speculative/disaggregated serving; one architecture-transfer project first.
- Treat world models, multiple architecture ports, full rollout infrastructure,
  and a general Kubernetes platform as later specializations. Preserve them
  as options for the broad architecture-literate goal.
- Keep compiler optimization and CUDA Graph replay as distinct mechanisms;
  reuse the same workloads and measurement infrastructure across projects.

## Sources checked

- [torchao quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)
  distinguishes weight-only and activation/weight paths and reports separate
  quality and performance measurements.
- [torchao QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html)
  covers fake quantization during training/fine-tuning.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
  explains finite precision and non-associativity.
- [PyTorch memory profiling](https://docs.pytorch.org/tutorials/beginner/mosaic_memory_profiling_tutorial.html)
  illustrates activation-memory differences with checkpointing.
- Sources inform curriculum recommendations, not installed-version support
  claims. No packages installed or hardware compatibility verified.

## Learning and handoff

- **Learner conclusion:** identified quantization as a possible gap; response
  to the review not yet provided. No new mastery evidence.
- **Next concrete learning task:** explain accumulation error and distinguish
  a numerical tolerance from application-level accuracy acceptance in the
  existing Project 2 comparison.
- Review recommendations have not changed `roadmap.md` or learning code.
- Progress documentation changed locally; no commit, push, or remote
  synchronization check performed.
