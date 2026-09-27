# Project 2 — Quick revision

- **Scope:** whole-model and individual GPU-kernel profiling.
- **Approach:** concepts and experiments at both levels; notes added as topics are covered.

## Lessons

- [GPU timing and synchronization](01-gpu-timing-and-synchronization.md)
- [Training-step setup pitfalls](02-training-step-setup.md)
- [Warmup and timing a block of steps](03-warmup-and-block-timing.md)
- [First profiler capture](04-first-profiler-capture.md)
- [Exporting and reading a timeline](05-reading-a-timeline.md)
- [Timeline reading checklist](06-timeline-reading-checklist.md)
- [Kernel launch arguments](07-kernel-launch-arguments.md)
- [Operation IDs, CUDA contexts, and queue timestamps](08-operation-ids-and-cuda-context.md)
- [Labeling training phases](09-labeling-training-phases.md)
- [Ranking backward kernels](10-ranking-backward-kernels.md)
- [Isolating convolution input gradients](11-isolating-convolution-input-gradients.md)
