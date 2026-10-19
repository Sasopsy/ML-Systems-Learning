# 2026-10-04 — Issued versus executed instruction averages

## Context

- **Project/milestone:** Project 2 offline Warp State Statistics.
- **Mode:** TUTOR; user asked the distinction between two displayed 1.53-cycle ratios.
- **Learning target:** instruction accounting and accumulated warp-cycles, separate from completion latency.
- **Starting commit:** `e6def5a40598e69dd53b93b667f751337d34ea45`.
- **Environment:** continuing local Mac workspace; no GPU/framework probe or execution.

## Work and evidence

- Python standard-library CSV reads of `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv` verified:
  - `smsp__average_warp_latency_per_inst_issued.ratio`: **1.531242**.
  - `smsp__average_warps_active_per_inst_executed.ratio`: **1.532910**.
  - Issued instructions: **18,116,865**; executed: **18,097,152**.
- Both ratios round to 1.53; not exact equality. Explanation uses different issue/execution denominators and accumulated resident-warp cycles.
- Consulted NVIDIA profiling guide and NVIDIA developer response on single-issue Hopper normalization. Verified documented IMC miss/reissue mechanism; no attribution of observed counter difference to that mechanism.
- Read an existing untracked note 15 on constants/reissue; preserved it unchanged. Updated note 14 and progress records only.
- Ran `git rev-parse HEAD` and `git status --short`; earlier local edits/untracked notes remain. No new capture or GPU run.

## Learning and handoff

- **Learner conclusion:** not yet provided for this distinction.
- **Understanding:** explained, not independently demonstrated.
- **Next task:** return to whether independent work from another warp can fill an issue opportunity during a dependency wait, then compare no-issue-cycle stall evidence.
- Changes remain local; no commit, push, or remote synchronization checked.
