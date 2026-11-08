# Project 3 revision notes

- **Scope:** kernel writing, validation, and profiling across CUDA C++, Triton, and TileLang.
- **Current state:** affine, bias + ReLU, baseline/warp row sum, and baseline/warp/online softmax passed the existing H100 correctness and sanitizer suites. First synthetic CUDA-event timings are recorded in [results](../results/2026-10-26-cuda-timing/README.md).
- [Elementwise kernels](01-elementwise-kernels/README.md): thread indexing and boundary handling.
- [Reductions](02-reductions/README.md): row sums, stable/online softmax, warp reductions, and synchronization.
- [Validation and measurement](03-validation-and-measurement/README.md): what correctness checks and sanitizers establish.
- [GEMM](04-gemm/README.md): matrix layouts and one-output-per-thread baseline.
- [Softmax comparison — blog working notes](05-softmax-comparison/README.md): kernel inventory, per-shape speedups, and measurement boundaries.
- Add concise lesson notes in descriptive section folders with section indexes as topics are covered. Keep review Q&A separate.
- [Project plan and progress](../../progress/projects/03-gpu-operator-library.md).
- [Project 2 profiling notes](../../project_02_performance_laboratory/notes/README.md).

## Offline handoff

- [Full-sweep profiles and comparison guide](../results/2026-10-27-offline-profiles/README.md): saved evidence for Mac study.
- [Complete results index](../results/README.md).
