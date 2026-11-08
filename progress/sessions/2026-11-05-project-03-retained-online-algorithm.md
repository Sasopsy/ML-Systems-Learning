# 2026-11-05 — Retained online softmax algorithm

## Context

- Project3 softmax comparison; TUTOR explanation with authorized note maintenance.
- User requested our retained-online algorithm after the three PyTorch families.
- Starting commit `b664fcb`; local Mac. Target: retained storage, conditional
  online max/sum updates, and mergeable normalized-sum states.

## Work and evidence

- Added `project_03_gpu_operator_library/notes/05-softmax-comparison/14-retained-online-softmax.md`
  and section index entry. Preserved compact LaTeX, legend and explicit local work.
- Read downloaded `source/softmax.cu` and `source/softmax_sweep_bridge.cu` in the
  Oct31 package; compared the relevant local learning code. Recorded source uses
  `dims` and allows general whole-warp block sizes; benchmark uses128 as before.
- Parsed all30 saved retained-online profile CSVs across Oct27/Oct31 packages:
  block128, one block per row, exact-width template matches each recorded shape.
- Described unchanged source behavior: load/retain first, conditional recurrence,
  two-float shuffle merges, four shared warp pairs, two barriers, reuse logits.
  Empty-pair identity guards precede exponentials; only lane0 warp results used.
- Checked note step numbers, LaTeX brace balance, local links and `git diff --check`.
  No visual renderer, CUDA execution, new timing or kernel implementation changes.

## Learning and handoff

- Explanation supplied; independent understanding not assessed.
- Next: discuss retained Online, then link algorithms to per-shape profile evidence.
- Existing local work and paused GEMM resume point preserved. No commit/push;
  offline source/data links require the separate results downloads.
