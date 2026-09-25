# Current status

Last updated: 2026-09-25 (Asia/Kolkata).

## Active focus

**Project 2: ML Performance Laboratory.** The user explicitly switched to it after
the scalar-autograd/MLP work. Its benchmark implementation has not started.

Support setup: local Git and durable progress tracking for multiple machines.
The user explicitly deferred GitHub setup. No remote is configured or pushed.

## Resume next

1. Select the Project 2 execution machine: local Mac or NVIDIA GPU machine.
2. In TUTOR mode, define one synthetic CNN training-step baseline and its timing
   boundary. Ask for the learner's prediction about what the measurement includes
   before implementing measurement code.

Before moving to another machine, configure and push a remote when the user is ready.

Initial benchmark acceptance criteria: reproducible configuration and seed, stated
device/software/dtype/batch/input shape, warmup, completed-work timing, repeated
measurements, step latency, throughput, and correctness checks. No optimization yet.
Synthetic pre-created inputs exclude real dataset loading and preprocessing; define
explicitly whether device transfer is inside or outside the timing boundary.

## Project state

| Project | Implementation | Learning evidence |
| --- | --- | --- |
| [1: Mini framework](projects/01-mini-framework.md) | Paused after scalar MLP; full project incomplete | Assisted implementation and passing checks; independent mastery not established |
| [2: Performance laboratory](projects/02-performance-laboratory.md) | Not started; active next project | Initial scope discussed; measurement skills not yet assessed |

## Latest verified local environment

2026-09-25: macOS/Darwin arm64, default `python3` is 3.9.6; no PyTorch module in
that interpreter. Other environments and remote hardware have not been verified.

Project 1 checks passed on this runtime. See the
[setup session](sessions/2026-09-25-git-setup.md) for evidence and limitations.
