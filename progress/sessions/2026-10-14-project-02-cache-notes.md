# 2026-10-14 — Project 2 cache revision notes

## Context

- Project/milestone: Project 2, offline Nsight memory analysis.
- Mode: TUTOR; explicitly requested documentation support.
- Learning target: recall L2/LTS, TEX/L1TEX, and per-SM versus GPU-wide cache sharing.
- Starting commit: `e6def5a40598e69dd53b93b667f751337d34ea45`.
- Environment: local macOS workspace; no GPU or framework execution.

## Work and evidence

- Updated `project_02_performance_laboratory/notes/03-gpu-hardware-and-scheduling/17-dram-throughput.md` and `notes/README.md` within that project.
- Condensed existing explanations into a table, short bullets, and a cross-SM weight-reuse example; retained NVIDIA source links.
- Commands: read instructions/status/project/session/template/notes; `git rev-parse HEAD`; documentation edits using apply_patch and Python.
- Evidence: prior verified device attributes in `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv`; not remeasured this session.
- Checks: reviewed revised text and relative index destination. No code tests needed for prose-only edits.

## Experiment

- Not applicable; no new measurement or kernel modification.

## Learning and handoff

- Learner requested notes; no new independent conclusion or transfer assessment.
- Notes summarize explanations, not mastery.
- Next task: learner assesses whether the existing on035 evidence supports DRAM bandwidth saturation and states what low throughput cannot rule out.
- Changes remain local and uncommitted; no push or remote synchronization performed.
