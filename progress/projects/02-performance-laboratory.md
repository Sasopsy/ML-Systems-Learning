# Project 2: ML Performance Laboratory

Implementation: **not started**. This is the active next project.
Understanding: profiling and measurement have not yet been assessed.

## Agreed direction

Start with a small existing PyTorch CNN and synthetic inputs. The learning target
is measurement and execution, not implementing model layers. The initial baseline
should precede optimization, deliberate bottlenecks, and transformer workloads.

## First milestone

Build a reproducible training-step benchmark. Define forward, loss, backward,
gradient clearing, and optimizer boundaries. Record batch size, input shape, dtype,
model configuration, seed, software/device details, warmup, repetitions, elapsed
time per completed step, and images per second.

Audit asynchronous execution and synchronization when using a GPU. Keep diagnostic
profiling separate from final timing when profiling perturbs execution. Report
what synthetic inputs exclude and whether transfer is included.

Before code: ask the learner to predict what the chosen measurement includes and
what evidence would show that it measures completed work. Independent understanding
can later be checked by changing the timing boundary and explaining the difference.

## Open decisions

- Execution machine: local Mac CPU/Apple GPU or NVIDIA GPU machine.
- Python/PyTorch environment and versions; nothing installed for this project yet.
- Concrete CNN, tensor sizes, and run budget after device selection.

The local default interpreter was checked on 2026-09-25: Python 3.9.6 on macOS
arm64, without PyTorch. Do not generalize this result to other interpreters/machines.

## Later milestones

Separate stage timing; inspect profiler traces; introduce controlled input,
synchronization, allocation, and small-kernel bottlenecks; compare hypotheses;
then add transformer encoder and decoder workloads. Use CUDA/Nsight on suitable
NVIDIA hardware when reaching those parts of the roadmap.

No benchmark measurements or speedup claims exist yet.
