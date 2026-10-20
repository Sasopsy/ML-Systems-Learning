# 2026-10-20 — Project 3 Mac/GPU workflow

## Context

- Project and milestone: Project 3 planning; first elementwise operator proposed.
- Mode: TUTOR; authorized progress/notes maintenance only.
- Learning target: parallel work assignment, indexing, and boundary handling; subsequent correctness/performance iteration.
- Starting commit: `fa47ece`; prior runtime-track documentation already uncommitted.
- Environment: local macOS workspace. Hardware/framework versions not inspected; no GPU connection.

## Work and evidence

- Read repository instructions, status, Project 2 progress, latest runtime planning session, and Project 3 roadmap scope.
- Commands: `git status --short`, `git rev-parse --short HEAD`, `rg` file/text searches, `cat` and `sed` document reads.
- Checked official [NVIDIA macOS tooling statement](https://developer.nvidia.com/nvidia-cuda-toolkit-13_0_0-developer-tools-mac-hosts): CUDA target execution/development is unsupported on macOS; some host profiling/debugging tools are available.
- Proposed Mac design/reference/source-review work with early Linux NVIDIA compilation/validation, followed by separate timing and profiling runs.
- Updated `progress/STATUS.md`, Project 2 transition note, new Project 3 progress record, and Project 3 notes index. Detailed workflow remains proposed.
- No learning-target code written. No GPU experiment, install, benchmark, remote sync, or mastery assessment.

## Experiment

- Not applicable: planning only. Proposed first workload is contiguous FP32 `y = a*x + b`; exact sizes and acceptance criteria are not yet set.

## Learning and handoff

- Learner conclusion: requests moving from standalone profiling study to kernel writing with profiling integrated.
- Understanding level: no Project 3 implementation evidence yet.
- Open questions: eventual remote environment availability/versions; first operator contract and implementation.
- Next task: define first contract and have learner derive work assignment/boundary handling.
- Prior and current edits remain local and uncommitted; no push. Project 2 Q21 and unresolved acceptance/transfer work remain saved.
