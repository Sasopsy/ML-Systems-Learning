# 2026-10-16 — Variant instruction-count check

## Context

- Project 2, offline variant comparison; TUTOR.
- Target: test a prediction about instruction work versus rate.
- Local macOS; no GPU execution. Starting commit not rechecked.

## Work and evidence

- Python CSV reads of `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv` and `ncu_on_031-raw.csv`; searched corresponding logs for Instruction Statistics.
- Total, shared-load, LDGSTS and branch counts increased; DRAM bytes did not. No direct FFMA count found in inspected exports.
- Added exact counts and limits to note 18; percentage changes calculated in Python.

## Experiment

- No new experiment; analysis of saved captures only.

## Learning and handoff

- Learner predicted more memory transfers and accumulation after instruction-count prompting. Memory-instruction direction confirmed; byte-transfer and accumulation interpretations refined with assistance.
- No independent full diagnosis claimed. Next distinguish total machine work from useful mathematical work; accumulation hypothesis needs opcode evidence.
- Local documentation edits only; no commit, push, or remote synchronization.

## Requested note update

- Added percentage-change column and concise explanations of memory instructions versus HBM bytes, unchanged mathematical work, and higher issue throughput versus useful-work throughput to note 18. Removed the stale pending-comparison wording.
- Verified saved comparison text; no new measurements. Existing next task and uncertainty remain unchanged.

## Requested instruction-counter lookup notes

- Added Details/Raw navigation, exact total/issued/shared-load/LDGSTS/branch counter names, aggregation and warp-instruction units, and the CSV/log verification method to note 18. Added an index link.
- Checked saved text and index destination. Documentation only; no new profiling or current live-UI inspection.
- Next: learner final synthesis and independent transfer check; numerical acceptance remains unresolved.

## Open-note review and eligibility follow-up

- Learner answered occupancy/coverage arithmetic correctly, then correctly computed 102,400 resident-warp cycles and 1.21% of elapsed GPU-wide capacity. Denominator definition and percentage notation refined.
- Correctly distinguished resident and eligible warps. Requested deeper explanation before the equal-average eligible-warp exercise.
- Checked NVIDIA scheduler/stall documentation and added note 16 Q&A: fetch, dependencies, synchronization, functional-unit/queue readiness, branch bookkeeping, sleep; distinguished Not Selected from ineligible.
- No independent transfer certification: this is an open-note guided review. Next resume pending 100-cycle eligibility distribution question. No GPU execution or remote synchronization.

## Requested resident/eligible notes

- Existing note 16 already contained the blocking-condition table. Added conceptual eligibility checks, selection distinction, and shared-tile barrier sketch; linked the section from the notes index.
- Reviewed the updated section; no GPU run or new assessment. Resume the pending eligibility-distribution question next.

## Requested readiness/issue formulas

- Learner correctly solved the 2-eligible-for-50-cycles case after guided clarification: average eligible count 1, issue rate 0.5 instructions/cycle. Corrected numerator units.
- Added formula table and three 100-cycle examples to note 16, with one-issue-per-cycle assumptions, eligible-warp-cycle interpretation, and active-versus-elapsed denominator caveat. Reviewed arithmetic and units.
- No new GPU measurements. Next pending exercise: 3% DRAM throughput with Long Scoreboard stalls and no eligible warps.

## Scoreboard prerequisite explanation

- User asked what scoreboard and Long Scoreboard mean before the pending memory question. Checked NVIDIA definitions and explained conceptual dependency tracking, global-load consumer example, long/short path distinction, and no DRAM-miss inference from the label.
- Added note 16 Q&A. No GPU run or independent assessment; return to pending low-DRAM/Long-Scoreboard exercise afterward.

## Scoreboard measurement, MIO, and registers

- Learner remained uncertain whether scoreboard counts operations or dependencies and how SRAM/registers map to Long/Short. Checked NVIDIA definitions and engineer explanations.
- Clarified tracking mechanism versus profiler observations, MIO expansion, memory technology versus operation class, and three register-producing examples. Note 16 updated. No new experiment; pending latency/bandwidth exercise unchanged.

## Scoreboard measurement and MIO clarification

- Learner remained unclear about tracking versus measuring operations and L1TEX/shared/register boundaries. Checked NVIDIA definitions and explained scoreboard as execution bookkeeping, stall metrics as blocked-state measurements, MIO as memory input/output path, and register readiness determined by the producing operation.
- Note 16 updated. Physical shared/L1TEX organization does not make shared dependency waits Long Scoreboard. No GPU run or independent assessment. Next return to the pending latency-versus-bandwidth question once this distinction is clear.
