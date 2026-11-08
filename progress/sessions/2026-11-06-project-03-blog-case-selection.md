# 2026-11-06 — Select softmax blog case studies

## Context

- Project3 softmax blog, TUTOR analysis/support. User requests first selecting
  interesting shapes, then investigating only their profiles in later steps.
- Working title: Faster Softmax Than PyTorch Eager (for Certain Shapes).
- Local Mac; starting commit `b664fcb`. Retained Online is the custom variant;
  existing width≤16384 plot scope retained.

## Work and evidence

- Read current instructions/progress, note09 timings and note10 plot scope.
  Requested an independent agent review of a compact shortlist; no delegated edits.
- Recommended five cases:37×2048,37×4096,4096×4096,4096×8192,4096×16384.
  Added note15 and index entry. Selection uses connected contrasts and all three
  PyTorch families, not only best speedups.
- Python CSV/statistics checks recomputed20 selected medians from180 raw samples
  in the two source timing packages. Same winner in direct and graph modes for
  all five shapes; nine-round observed ranges are separated within each mode.
  This is within-run evidence, not independent-run statistical significance.
- Verified the10 selected `.ncu-rep` files exist. Did not inspect counters or
  perform new GPU experiments. Graph replay is not eager Python-call latency;
  8192 and16384 come from separate experiment runs.
- Checked local note links and `git diff --check`. No benchmark/code changes.

## Learning and handoff

- Next: state a hypothesis and alternative for37×2048 versus37×4096, then inspect
  those four reports. Later inspect the other selected pairs.
- Recommendations/assisted analysis; independent understanding not assessed.
- Changes remain local/uncommitted. No commit/push; earlier work and GEMM resume
  point preserved. Report/source artifacts require separate results downloads.
