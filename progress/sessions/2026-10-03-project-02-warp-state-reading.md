# 2026-10-03 — Project 2 warp-state reading

## Context

- **Project/milestone:** Project 2, offline forward-convolution scheduler study.
- **Mode:** TUTOR.
- **Learning target:** distinguish warp-state measurements and interpret reasons for lack of readiness.
- **Starting commit:** `e6def5a40598e69dd53b93b667f751337d34ea45`; earlier local note/progress changes remain.
- **Environment:** continuing local Mac workspace; hardware/framework versions not re-probed. No GPU execution.

## Work and evidence

- Explained active versus elapsed cycles in the preceding turn using an invented 100-cycle interval (80 active, 50 issuing). No experiment represented by the toy numbers.
- Parsed `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv` using Python standard-library CSV reader; extracted warp-state ratios and PC-sampling counts.
- Largest non-Selected per-issue-active ratios: Wait 0.179721, Dispatch Stall 0.101297, No Instruction 0.089087. Source fields retain their exported units; do not silently treat PC-sampling counts as the same measurements.
- Read session template and ran `git rev-parse HEAD`. No capture or source change.

## Learning and handoff

- **Learner conclusion:** no warp-state interpretation provided yet.
- Earlier correct readiness selection was an immediate prompted check, not independent transfer.
- **Next task:** learner opens on035 Warp State Statistics and reports the two largest non-Selected reason labels, values, and displayed units. Explain meanings only after the reading; do not diagnose causal speedup from ranking alone.
- New session/progress changes remain local; no commit/push or remote synchronization checked.

## Follow-up — First labels read correctly

- Learner identified Stall Wait and Stall Dispatch Stall, matching the saved normalized breakdown; reported inability to see sample counts/percentages.
- Re-extracted ratios with Python CSV reader: Wait 0.179721, Dispatch Stall 0.101297; exported unit is `inst` for these per-issue-active derived fields. Kept raw values separate from percentages and PC-sampling counts; UI screenshot of this graph not yet supplied.
- Consulted NVIDIA definitions: Wait is a fixed-latency execution dependency; Dispatch Stall holds back an instruction ready to issue due to other conflicts/events. Exact dispatch conflict remains unknown.
- Added concise definitions to note 14. No GPU run or causal speedup claim.
- **Understanding:** correct identification of requested labels; conceptual interpretation assisted and transfer pending.
- **Next task:** identify Wait in a dependent accumulator example; then compare sampled states restricted to cycles with no instruction issued.
