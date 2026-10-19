# Project 2 — Quick revision

- Topic folders group the short notes; original numbers preserve lesson order.
- **Review paused after Q20; resume at Q21.** Answers summarize lessons, not proof of mastery.

## Timing and setup

- [Section index](01-timing-and-setup/README.md)

- [GPU timing and synchronization](01-timing-and-setup/01-gpu-timing-and-synchronization.md)
- [Training-step setup pitfalls](01-timing-and-setup/02-training-step-setup.md)
- [Warmup and timing a block of steps](01-timing-and-setup/03-warmup-and-block-timing.md)

## Timeline profiling

- [Section index](02-timeline-profiling/README.md)

- [First profiler capture](02-timeline-profiling/04-first-profiler-capture.md)
- [Exporting and reading a timeline](02-timeline-profiling/05-reading-a-timeline.md)
- [Timeline reading checklist](02-timeline-profiling/06-timeline-reading-checklist.md)
- [Kernel launch arguments](02-timeline-profiling/07-kernel-launch-arguments.md)
- [Operation IDs, CUDA contexts, and queue timestamps](02-timeline-profiling/08-operation-ids-and-cuda-context.md)
- [Labeling training phases](02-timeline-profiling/09-labeling-training-phases.md)
- [Ranking backward kernels](02-timeline-profiling/10-ranking-backward-kernels.md)
- [Isolating a convolution input gradient](02-timeline-profiling/11-isolating-convolution-input-gradients.md)
- [From isolated kernels to model performance](02-timeline-profiling/12-from-kernel-to-model-performance.md)

## GPU hardware and scheduling

- [Section index](03-gpu-hardware-and-scheduling/README.md)

- [Nsight: timeline versus kernel investigation](03-gpu-hardware-and-scheduling/13-nsight-introduction.md)
- [Compute throughput through a dense layer](03-gpu-hardware-and-scheduling/14-compute-throughput-dense-layer-example.md)
- [Constants and instruction reissue](03-gpu-hardware-and-scheduling/15-constants-and-instruction-reissue.md)
- [Latency hiding](03-gpu-hardware-and-scheduling/16-latency-hiding.md)
- [GPU memory: throughput, caches, and wavefronts](03-gpu-hardware-and-scheduling/17-dram-throughput.md)

## Performance comparisons

- [Section index](04-performance-comparisons/README.md)

- [cuDNN search off/on: performance synthesis](04-performance-comparisons/18-off-on-performance-synthesis.md)

## Review questions

- [Section index](05-review/README.md)

- [Profiling review — questions and answers](05-review/19-profiling-review-questions-and-answers.md)

## Offline study

- [Report index, reading order and handoff](../../OFFLINE_PROJECT02.md)
