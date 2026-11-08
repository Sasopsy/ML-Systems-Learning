# 2026-11-02 — Large-width softmax results verification

## Context

- Project3, REVIEW of downloaded results; authorized progress maintenance.
- User requested checking new results after discussing specialization fairness.
- Local macOS; starting commit `b664fcb`, existing uncommitted roadmap/GEMM work preserved.
- Learning target: distinguish dispatch, algorithms, measured performance and causal hypotheses. Independent understanding not assessed.

## Work and evidence

- Read learning instructions, current status/project, prior handoff and new result indexes.
- Inspected `project_03_gpu_operator_library/results/2026-10-31-softmax-128k/` and the corresponding `2026-10-31-offline-profiles-128k/` package.
- Python manifest verification:178 files,334818377 bytes,24 reports; zero missing/size/hash errors. The first helper used unavailable `hashlib.file_digest`; reran successfully with streaming SHA256 on the local Python version.
- Python CSV check: recomputed all72 summary medians from648 raw samples (nine samples each). All24 capture logs contain PASS and exit0; metadata records36 initial reference checks. Four retained-kernel memchecks pass for three rows at each of16384/32768/65536/131072 dims.
- Kernel names: PyTorch `cunn_SoftMaxForward<4,...>` and ours `softmax_online_registers<128,dims>` across all12 shapes. Timing additionally includes non-retained `softmax_online<128>`.
- Source snapshot preserves exact-width specialization, renames Cols to dims and permits generic block sizes; benchmark dispatch still uses N128. Local tracked source/bridge is older. No source replacement, pull or remote operation performed.
- Compiler reports168 registers/thread and no spills at16384;255 registers/thread and spills at32768+. Profiled local spilling requests support a register-pressure hypothesis, not an isolated causal runtime estimate.

## Experiment evidence (recorded remote run, not rerun here)

- FP32 contiguous inputs; rows37/256/4096 × dims16384/32768/65536/131072.
- H10080GBHBM3, PyTorch2.11.0+cu130; pinned revision in package metadata.
- Fifty warmups, nine rounds of500 calls, rotated variants, CUDA events; preallocated outputs. Direct and graph modes; reused buffers and uncontrolled clocks/caches. Profile replay is separate.
- Graph medians at4096 rows, microseconds (PyTorch / online / retained):16384:214.255/317.428/261.213;32768:654.465/624.104/1646.811;65536:1400.800/1241.854/4481.342;131072:2805.656/2534.497/12867.443.
- Retained loses all12 extension shapes in both modes. Online has lower medians than PyTorch for the three largest widths at4096 rows, but no paired online profiles are included. Do not interpret the pair index's losses as losses for every custom variant.
- Runtime-width/bucket-specialization comparison remains unperformed; this is a width extension, not that proposed control experiment.

## Learning and handoff

- Learner conclusion not yet provided; no mastery assessment or new GPU measurements.
- Next: explain PyTorch warp, register-buffered, and general kernel algorithms before evaluating per-shape performance hypotheses. Preserve GEMM resume point.
- Results are downloaded/ignored, not Git-tracked. Progress changes remain local; no commit/push.

## Follow-up — blog working notes and complete tables

- User requested a dedicated short-notes subsection and a first-step table of all
  compared shapes, kernel names, timings and speedups, highlighting custom wins.
- Added `project_03_gpu_operator_library/notes/05-softmax-comparison/README.md`
  and `09-results-and-kernel-inventory.md`; linked the Project3 notes index.
- Used the current downloaded combined summary with experiment provenance. Checked
  all180 medians against source summaries and recomputed them from1620 samples
  (972 earlier plus648 extension); older narrative quotations were not substituted.
- Validated30 captured kernel-pair names,60 rendered rows,120 custom ratios and
  bold win markers against CSV; checked25 local note/index links. `git diff --check`
  passed. No CUDA compile/run or new timing measurement.
- Saved graph and direct tables separately. Graph results use eager-selected kernels
  under CUDA Graph replay, not ordinary eager invocation latency. Direct timing
  includes potentially different host-submission gaps for PyTorch and ctypes.
- Graph observed wins: online7/30 and retained9/30; direct11/30 each. Tables carry
  workload, exact-width specialization, synthetic-input, two-run, clocks/caches,
  timing-distribution and profile-coverage limitations.
- This is source-backed results compilation, not a completed causal explanation,
  independent learner mastery, or published blog. Next: algorithm walkthroughs,
  then profiling hypotheses. Changes remain local; no commit or push.

## Follow-up — PyTorch/Online plots through width16384

- User requested graphs by row count, comparing PyTorch and Online through16384.
  Used non-retained Online and the established graph-replay timing mode; stated
  those choices explicitly. Retained variant is excluded.
- Added `benchmarks/plot_softmax_comparison.py`, note10 in the comparison section,
  and three PNG plus three SVG plots under that section's `figures/` directory.
- Script selects42 medians, checks complete shape/variant coverage and experiment
  provenance, and uses matching logarithmic axes plus measured-value annotations.
- Local/system and bundled Python lacked Matplotlib. Created a temporary virtual
  environment; the sandboxed pip download failed on network access, then approved
  escalation installed Matplotlib3.9.4 there. No global packages were changed.
- Ran the plot script successfully with a temporary Matplotlib cache. A fontconfig
  cache warning appeared, but all six exports completed. Visually inspected all
  three PNGs: labels, legends, axes and footnotes are readable and uncut.
- Checked PNG signatures, parsed SVG XML, verified note links and ran
  `git diff --check`. No CUDA compilation, new GPU timings, publication or Git push.
- Observed for this selected mode/variant: Online has lower medians at widths1024
  and2048 for37/256 rows, while PyTorch wins all selected widths at4096 rows.
  Algorithmic explanations remain the next lesson, not established by these plots.

## Follow-up — retained Online plots

- User requested retained versus PyTorch graphs with the preceding scope.
- Extended the plot script with `--variant online_registers`, retaining the
  non-retained default and its outputs. Generated three additional PNG/SVG pairs
  for rows37/256/4096 and widths128–16384 from graph medians;42 measurements
  validated. Kept the same axes, timing notes, provenance and colors.
- Visually checked all three new PNGs; labels and curves are legible. Note10 now
  contains both plot sets and reproduction commands. Checked SVG/PNG outputs and
  local links; `git diff --check` passed. No kernel change, GPU run or push.

## Follow-up — linear axes requested

- User preferred linear scales to show absolute runtime differences. Updated both
  axes to linear and runtime to start at zero; numeric width spacing retained.
- Regenerated all six PNG/SVG pairs, with row-specific y limits25/40/400µs shared
  across retained/non-retained plots. Small-width values remain plotted; text
  annotations start at2048 to avoid crowding. Source measurements unchanged.
- Updated note10, visually inspected both plot sets, checked SVG labels and PNG
  presence, and ran `git diff --check`. Earlier log-scale entries describe the
  superseded versions. No new GPU execution, commit or push.

## Follow-up — PyTorch warp softmax algorithm note

- User requested an algorithm explanation here and in the blog subsection with a
  PyTorch reference. Added note11 and linked the section index.
- Read the bundled `PersistentSoftmax.cuh` forward kernel, XOR reduction helper,
  and dispatch plus `SoftMax.cu` selection. Opened pinned upstream SoftMax.cu and
  raw PersistentSoftmax.cuh; GitHub's header-page fetch initially returned a cache
  miss, while raw source succeeded. Linked the exact recorded revision.
- Explained finite/unmasked FP32 stable softmax: retain inputs, private/warp max,
  retain exponentials and private sum, warp sum, normalize/store. At width1024,
  one warp per row/four rows per block; at128, two separate rows per warp/eight per
  block. Distinguished capacity specialization, bounds handling, register intent
  versus actual allocation, and array passes versus global-memory passes.
- Added source-level comparison with retained Online and hypotheses for later
  profiling; no causal performance attribution or mastery claim. Local links and
  `git diff --check` pass. No kernel edits, CUDA execution, commit or push.
- Next: discuss this algorithm before Reg/General and per-shape profile analysis.

## Follow-up — detailed template-parameter explanation

- User requested more detail on template parameters. Expanded note11 with the
  six-argument declaration and observed specialization, types versus compile-time
  values, input/output versus accumulator storage, capacity buckets, derived
  constants, log-softmax/masking choices, runtime argument table and launch flow.
- Cross-checked local forward/dispatch source, including the half-to-float path.
  Explained width900 with capacity1024, logical padding, two-row warp batching,
  and why compile-time values do not imply exact-width-only input support.
- Verified illustrative lane valid counts (four lanes with29,28 with28 =900) and
  ceil(rows/4) grid sizes, local links and `git diff --check`. No GPU tests or
  changes to learning-target code. User understanding not assessed.
- Next remains algorithm discussion, then the other PyTorch families and causal
  profiling hypotheses. Changes local; no commit/push.
