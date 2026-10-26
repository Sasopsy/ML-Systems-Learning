# Project 3 revision notes

- **Scope:** kernel writing, validation, and profiling across CUDA C++, Triton, and TileLang.
- **Current state:** affine, bias + ReLU, row sum, and baseline/warp/online softmax source-reviewed. Host correctness harnesses ready; GPU validation pending.
- [Elementwise kernels](01-elementwise-kernels/README.md): thread indexing and boundary handling.
- [Reductions](02-reductions/README.md): row sums, stable/online softmax, warp reductions, and synchronization.
- Add concise lesson notes in descriptive section folders with section indexes as topics are covered. Keep review Q&A separate.
- [Project plan and progress](../../progress/projects/03-gpu-operator-library.md).
- [Project 2 profiling notes](../../project_02_performance_laboratory/notes/README.md).
