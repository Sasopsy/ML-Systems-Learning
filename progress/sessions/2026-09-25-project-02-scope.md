# 2026-09-25 — Project 2 model and kernel profiling scope

## Context

- Project and milestone: Project 2; first reproducible training-step baseline,
  not implemented yet.
- Mode: TUTOR.
- Learning target: reliable completed-work measurement; project scope includes
  both model execution and individual GPU-kernel behavior.
- Starting commit: `fda31210db4c111902584b01ede62deb49bfaa32`.
- Machine: current Linux GPU workspace, Linux 6.11.0-1016-nvidia x86_64;
  NVIDIA H100 80GB HBM3, driver 580.173.02; default Python 3.14.7 (Anaconda).

## Work and evidence

- Read learning instructions, current progress, active project note, latest
  session, and Project 2 roadmap. Recorded the user's explicit model/kernel scope
  in `progress/projects/02-performance-laboratory.md` and `progress/STATUS.md`.
- Read-only commands: `git status --short` (initially clean), `git rev-parse HEAD`,
  `git remote -v`, `uname -srm`,
  `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`,
  and a Python version / `importlib.util.find_spec("torch")` check.
- Observed: GPU reports 81559 MiB; default interpreter has no discoverable torch
  module. `origin` is configured for the GitHub repository. No fetch/push performed;
  synchronization and other Python environments remain unverified.
- Sandbox shell startup failed with a bwrap loopback permission error; read-only
  inspection succeeded through approved execution outside that sandbox.
- Raw evidence: tool output in this task; no separate benchmark artifact exists.
  Progress documents preserve the observations for later machines.
- No learning-target code edited, dependencies installed, or benchmark/tests run.

## Experiment (if applicable)

- Planned workload: small existing CNN with synthetic inputs. Concrete shape,
  dtype, batch, seed, optimizer, and execution budget remain to be chosen.
- Measurement boundary: to be defined before implementation; explicitly account
  for gradient clearing, forward, loss, backward, optimizer, and transfers.
- Learner prediction before assistance: not yet provided.
- Hypothesis, controls, warmup, repetitions, and synchronization: not yet specified.
- No timing results or performance diagnosis exist.

## Learning and handoff

- Learner's own conclusion: not yet provided.
- Understanding: unassessed; agreeing on scope does not establish mastery.
- Plan: connect model traces to isolated kernel experiments and validate any
  eventual kernel improvement in the original model workload.
- Support work can cover environment inspection and progress maintenance; learner
  retains measurement implementation and interpretation in TUTOR mode.
- One concrete next task: predict what a CPU timer around one GPU training step
  measures and propose evidence that all timed GPU work has completed. Later
  transfer check: change the timing boundary and explain the resulting difference.
- Changes remain local and uncommitted; no remote synchronization performed.

## Follow-up — Timing boundaries and pending GPU work

- TUTOR discussion only; no code execution, installations, or timing experiments.
- Learner initially recognized that returning from a GPU training step does not
  guarantee completion, and requested the synchronization mechanism by name.
- Assistant explained asynchronous submission and `torch.cuda.synchronize()`
  before/after a CPU wall-clock timing interval. Learner then described the CPU
  waiting for GPU completion, with assistance.
- Clarified that `perf_counter()` measures elapsed wall-clock time, including
  waiting, and that CPU responsibilities include more than kernel submission.
- Learner challenged a 20 ms earlier-work plus 5 ms step example because the
  timer's starting point matters. The example needed an explicit assumption:
  20 ms of earlier work remains at the starting timestamp, and the new step runs
  after that work without overlap. Under those simplified assumptions, a final
  synchronization can leave roughly 25 ms in the interval. These are illustrative
  numbers, not observed results. Process startup is not itself the timing boundary.
- Understanding: assisted recognition of completion waiting; precise boundary
  reasoning remains under discussion. Do not mark independent mastery.
- Next task: have the learner specify what must have completed at the beginning
  and end of the proposed CNN training-step timing interval.
- Only progress notes changed; no commit or push performed.

## Follow-up — Quick revision notes

- User reports understanding of the synchronization explanation; independent
  application remains untested.
- User explicitly requested per-project notes folders and ongoing short lesson
  summaries. Repository support work only; no learning-target code changed.
- Created `project_01_mini_framework/notes/` with an index, scalar-autograd note,
  and MLP/training note, grounded in the existing code and Project 1 record.
- Created `project_02_performance_laboratory/notes/` with an index and a short GPU
  timing/synchronization note from this conversation. The code block is a teaching
  sketch, not an implemented or validated benchmark.
- Updated `AGENTS.md` to maintain this convention for each project as it starts;
  linked revision notes from `README.md` and project progress records.
- Inspected repository files, roadmap headings, Project 1 sources, and Git status.
  No runtime tests or experiments were run for these documentation changes.
- Next task remains defining the CNN completed-work timing boundary before
  benchmark implementation. Changes are local; no commit or push performed.

## Follow-up — Start the CNN baseline

- User requested continuing Project 2 in TUTOR mode.
- Ran `conda env list`, inspected the Project 2 file list and progress notes, and
  used `conda run -n kvforge python -c ...` to inspect Python/PyTorch versions,
  built CUDA version, CUDA availability, and torchvision module discovery.
- Observed: `kvforge` has Python 3.12.14, PyTorch 2.11.0+cu130, built CUDA 13.0;
  CUDA availability returned True and torchvision is discoverable. Did not import
  torchvision, execute a model, allocate workload tensors, or install anything.
- Proposed a concrete synthetic baseline in the project note: ResNet-18, batch
  32, RGB 64x64 inputs, 10 classes, float32, seed 42, cross-entropy, plain SGD.
- Next task: learner writes untimed setup and a step comprising gradient reset,
  forward, loss, backward, and optimizer update. Review before benchmark timing.
- No benchmark code written by the assistant; no new evidence of independent
  implementation or understanding. Documentation changes remain local/uncommitted.
