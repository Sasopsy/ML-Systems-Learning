# 2026-10-01 — Launch API review

## Context

- Project2; TUTOR; CUDA launch API and phase attribution.
- Starting commit: dc283650dc09ebb8883e6fb553f9e0b7d77c6fb1.
- User-facing date Asia/Calcutta. Hardware/software not rechecked.

## Work and evidence

- Read learning instructions and latest project/status/session records.
- Read off006/on005 trace CPU annotations with Python JSON; consulted official
  CUDA13.0 execution API documentation. No new performance measurements.
- Added short launch-API revision note to note07.

## Learning and handoff


## Follow-up — Launch APIs and phase selection (2026-10-01, Asia/Calcutta)

- Learner reported off006/on005 phase durations and kernel names, plus launch
  timestamps instead of correlation IDs. Inspected saved traces: all four
  durations exactly match FIRST train_step, not requested second step.
- First off forward/backward 4572.173/6303.919 us; first on
  4513.049/5572.512 us. No phase GPU aggregation or new experiment performed.
- Explained standard cudaLaunchKernel versus extensible cudaLaunchKernelExC
  using official CUDA13.0 runtime documentation; note07 updated. API name alone
  does not identify optional attributes or explain kernel speedup.
- Next learner selects second step in each capture and copies integer correlation
  from GPU kernel args, following its CPU launch. Preserve learner-led analysis.
- No code changes, GPU runs, commit or push this follow-up.


## Follow-up — Delegated phase attribution completed

- User explicitly asked assistant to extract relevant information. Checked all
  six saved model-repeat traces against recorded SHA256 hashes; no GPU reruns.
- Every kernel across all three steps in each capture mapped to exactly one
  runtime/driver launch via correlation, matching External id and enclosing
  CPU op on the launch thread, then exactly one synchronous phase interval.
  This includes autograd worker launches. Summed only category=kernel durations.
- Compared second steps. Representative off006/on005 CPU forward durations
  2954.680/2700.272 us; backward3921.337/3727.392 us.
- GPU forward sum1676.552 ->1316.263 us, saving360.289 us (21.49%);
  backward1795.324 ->1675.332 us, saving119.992 us (6.68%).
- All six second-step captures support same ordering: off forward1676.552–
  1677.604 vs on1310.208–1316.263 us; off backward1792.253–1799.045
  vs on1671.133–1675.332 us. Forward shows larger absolute and relative saving.
- Selected forward launch correlation off4454/on4464, cudaLaunchKernel;
  selected dX off6485 cudaLaunchKernel/on6491 cudaLaunchKernelExC.
  Both dX CPU ops show matching input dimensions, External id1208.
- Full phase counts/durations, exact-name kernel groups and selected events saved
  in progress/sessions/2026-10-01-project-02-phase-kernel-comparison.json.
- GPU duration sums are diagnostic, not elapsed phase or training-step time.
  No claim all savings are on critical path; specific forward changes not yet
  analyzed. Next interpret result then compare forward kernel groups.
- Updated short revision note. Delegated analysis, not independent learner
  completion. No source edits, numerical acceptance changes, commit or push.


## Follow-up — Forward convolution savings located

- Matched second-forward 20 cudnn_convolution calls in off006/on005 by order,
  full input dimensions/strides, and concrete inputs excluding intended benchmark
  flag difference. Kernel External IDs verified by unique launch correlation,
  enclosing CPU operation and thread. Six trace hashes checked; no GPU reruns.
- Top groups: four X[32,64,16,16],W[64,64,3,3] calls save163.551us
  (274.400->110.849); three X[32,512,2,2],W[512,512,3,3] calls save146.174us
  (606.271->460.097). Together309.725us, ~86% of forward saving360.289us.
- Early group switches named tile256x64x8 to128x32x8; late group switches
  implicit_gemm_indexed to implicit_gemm with same named32x32x8 tile. Internal
  efficiency cause not established from names. Both patterns persist across
  all three processes per mode.
- All convolution kernels save361.660us, offset by1.371us increase in other
  forward kernels. One 1x1 convolution regresses6.784->10.496us; search does
  not guarantee every selected call is faster in diagnostic capture.
- Evidence: progress/sessions/2026-10-01-project-02-forward-convolution-comparison.json.
  Tensor values not proven equal after training; this is shape-matched attribution,
  not independent accuracy validation or kernel microarchitecture diagnosis.
- Next discuss tile dimensions as a possible implementation difference, or choose
  early group for controlled kernel investigation; no automatic custom-kernel task.
- Updated quick notes. No code edits, GPU runs, commit or push.


## Follow-up — Learner verifies tile launch resources

- Learner correctly supplied both grids,128-thread blocks, register counts,
  shared-memory bytes and durations, and independently computed32/128 blocks.
- Assistant verified second-forward convolution ordinal2 in off006/on005:
  External id401; launch correlations4552/4562. All supplied fields match.
- Explain that larger grid can use more of132 SMs; lower registers/shared-memory
  requirements can reduce resource constraints, without asserting measured
  occupancy or proven causal decomposition. Single-kernel improvement ~2.64x.
- Added tile/resource recap and compact table to revision note12. No GPU rerun,
  code edit, commit or push. Next distinguish block/thread counts from useful
  work and achieved utilization before deeper kernel analysis.


## Handoff — Next: Nsight Compute kernel investigation

- Completed saved-trace phase and forward-convolution attribution. Learner
  correctly read launch resources, computed block counts and stated that more
  blocks do not imply more useful convolution mathematics. Tile-area reasoning
  explained with assistance; internal cause of speedup remains a hypothesis.
- Next introduce Nsight Compute and inspect availability/permissions before
  designing a bounded isolated-forward-kernel capture. Target the 64-channel
  3x3 workload whose selected tile changes256x64x8 ->128x32x8; distinguish
  launch metadata from measured occupancy, activity and memory behavior.
- No Nsight Compute run or custom kernel implemented yet. Numerical acceptance
  remains unresolved; full project and independent exit test incomplete.
- User authorized add/commit/push. Evidence JSON parses and local note links
  resolve; no new GPU execution for handoff. Remote fetch succeeded; push to
  be verified after commit. Raw traces/tensors remain ignored local artifacts.
