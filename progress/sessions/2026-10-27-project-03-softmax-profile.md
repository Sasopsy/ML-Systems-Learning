# 2026-10-27 — Project 3 softmax profile

## Context

- Project 3; TUTOR with explicitly delegated profiling support. Starting kernel commit `9882f73`; earlier validation/timing support and progress changes already local.
- Learning target: distinguish instruction work, bandwidth, and exposed dependency latency using measurements. Learner prediction not yet provided; earlier assistant hypothesis was extra merge work/dependencies versus saved memory reads.
- H100 80GB HBM3, CUDA 13.0.88, driver 580.173.02; Nsight Compute 2025.3.1.0.

## Work and evidence

- Added `project_03_gpu_operator_library/benchmarks/profile_softmax.cu` to select one of three unchanged kernels, warm it 50 times, bracket one launch with profiler start/stop and validate output against the existing double reference.
- Built with nvcc C++17, O3, lineinfo, sm_90; captured full reports sequentially with 240-second subprocess limits. All three processes exited 0, 38 replay passes each, zero bad elements/rows after profiling.
- Commands, reports with embedded source, text/raw exports, selected comparison and interpretation saved under `project_03_gpu_operator_library/results/2026-10-27-softmax-ncu/`.
- Read NVIDIA's official metric definitions for scoreboard/pipeline stalls. Exported CUDA/SASS for warp and online; static repeated exponentials visible, but CLI per-source stall columns zero despite nonzero aggregate samples. No exact line attribution claimed.

## Experiment

- Same synthetic FP32 4096×1024 input and 128-thread row mapping as benchmark. One selected launch; no allocation/copies inside selected kernel, no kernel arithmetic edits.
- Online executes 52.3% more warp instructions than warp; global-load instructions decrease one-third; long-scoreboard ratio increases from 2.69 to 6.26 cycles/issued instruction. No spills; register count decreases, occupancy remains similar.
- Diagnosis: extra instruction work plus exposed load-dependency latency; no evidence for a saturated special-function pipeline or memory bandwidth ceiling. Correlation does not quantify individual causes.
- Uncontrolled clocks/caches intentionally preserve warm-cache intent without clock changes. Replays perturb conditions; six CTC traffic counters unavailable. Detailed limitations recorded with reports. Unprofiled timing remains separate evidence.

## Learning and handoff

- Learner conclusion not yet provided. Assisted profiling success is not independent mastery.
- Added a short revision note: fewer loads versus DRAM traffic, load-latency stalls versus bandwidth, and stall fractions versus wall time.
- Next task: inspect warp/online reports side by side, then propose one targeted optimization and predict which counter should change before measuring it.
- All harnesses, results and records are local/uncommitted; no push or rental shutdown performed.

## Follow-up — requested Project 2 revision note

- User acknowledged the normalized load-dependency stall explanation and requested saving it in Project 2.
- Extended `project_02_performance_laboratory/notes/03-gpu-hardware-and-scheduling/16-latency-hiding.md` with a short load/consumer example, conceptual ratio, 626/100 toy calculation, exact metric name, and per-load/wall-time/bandwidth caveats. Added a top-level index link and updated Project 2/status records.
- No new experiment or independent-understanding assessment. Project 2 review stays paused at Q21; next active task remains interpreting Project 3 reports and choosing one controlled optimization. Changes local/uncommitted.

## Follow-up — quick PyTorch baseline

- User requested a quick PyTorch run before further stall investigation. Added a standalone Python support benchmark, no learner kernel edits.
- Executed `conda run --no-capture-output -n kvforge python project_03_gpu_operator_library/benchmarks/time_torch_softmax.py`; exit 0, 36 timing samples, all reference/invariant checks passed. PyTorch 2.11.0+cu130 on H100.
- Identical synthetic FP32 inputs, 50 warmups, nine rounds × 500 calls, alternating normal/preallocated modes. Reference uses NumPy FP64 max-subtracted exponentials; original tolerances retained.
- 4096×1024 median: 12.685 µs ordinary eager softmax, 10.175 µs preallocated output. 37×130: 5.337/4.663 µs. Raw samples and methodology saved under `project_03_gpu_operator_library/results/2026-10-27-pytorch-softmax/`.
- Direct Python submission can introduce GPU idle gaps inside event intervals; allocator/cache behavior differs by mode. No claim about pure kernel speed, chosen PyTorch implementation or causal bottleneck. No profiling in this run.
- Next: resume controlled investigation of online's load-dependency stalls. Learner interpretation pending; support code/results/progress local and uncommitted.

## Follow-up — divergence hypothesis

- User proposed substantial warp divergence. Read existing raw metrics/details and exported source-correlated divergent branch counts; no new GPU run.
- Warp/online branch efficiency 100%/99.01%; active lanes 32/31.38; active predicated-on lanes 31.12/29.91. Online 8,192 divergent branch targets all map to two machine branches at the merge empty-summary guard (4,096 each).
- Divergence exists; widespread lane inactivity is not observed. Additional control instructions remain a separate cost. Do not convert branch uniformity into a runtime fraction or generalize to the unimplemented conditional-update optimization.
- Saved branch source export, report follow-up and concise revision note. No independent-understanding assessment; next task remains choosing one controlled optimization. Changes local/uncommitted.

## Follow-up — Project 2 divergence note

- User requested saving the four metric definitions and Nsight navigation in Project 2. Added note 20 under GPU hardware/scheduling and linked both indexes.
- Covers branch efficiency, active versus predicated-on lanes, divergent branch targets, Raw metric lookup and CUDA/SASS attribution. Existing online-softmax numbers are a labeled example with a relative link to full evidence.
- No new experiment or mastery claim. Verified local links and whitespace. Project 2 review remains paused at Q21; active Project 3 next task unchanged. Documentation remains local/uncommitted.

## Follow-up — specialized accumulation implementation and execution

- User explicitly delegated only the per-element update change and initially prohibited running. Edited first-value initialization and conditional one-exponential updates; generic merges unchanged. No execution in that edit turn.
- User subsequently authorized compilation/execution. Built softmax and timing harnesses with original flags; ran all 30 correctness cases, the existing full timing harness, and memcheck/racecheck/synccheck/initcheck on softmax. All exited 0 with no correctness failures or sanitizer errors/hazards.
- Evidence and tested-source SHA256 saved under `project_03_gpu_operator_library/results/2026-10-27-specialized-online/`; previous results preserved.
- Nine rounds × 500 calls after 50 warmups: online large median 17.125504 µs, mean 17.108430 µs; small median 3.191424 µs. Large current warp 12.543040 µs. Earlier online 17.771969 µs, so preliminary separate-run reduction ~3.6%; unchanged controls also shifted, preventing a strong causal speedup claim.
- No new profile or per-call p99. Next: profile changed online for instruction/divergence/load-stall comparison. User-delegated implementation and assisted validation; no independent mastery assessment. Changes local/uncommitted.

## Follow-up — exact PyTorch implementation

- User requested identifying the actual PyTorch path. Captured one normal and one preallocated eager call after 50 warmups with torch.profiler; both launch the same persistent warp kernel. Script and saved traces/dispatch metadata recorded under Project 3 benchmarks/results.
- Read installed PersistentSoftmax.cuh and matching official SoftMax.cu at installed revision 70d99e998b4955e0049d13a98d77ae1b14db1f45. Verified one warp per row, four rows/block, 32 saved values/thread, separate max and exponential-sum stages with retained exponentials and XOR-shuffle reductions.
- Trace reports 48 registers/thread, zero shared memory, grid 1024 and block (32,4). No independent PyTorch SASS/spill counters or new benchmark collected. Mapping/algorithm differs from learner online implementation; source observations do not isolate causes of speed differences.
- Next: retain values as one controlled change before separately testing row mapping. No learner kernel edits; notes and artifacts local/uncommitted.

## Follow-up — common-width/dtype dispatch survey

- User requested other common values. Added and ran `benchmarks/survey_torch_softmax.py` through kvforge; initial sweep extended near the register/shared boundary after inspecting results.
- Final survey has 57 cases: 17 widths, three dtypes, 256 rows; six extra FP32 cases at rows 1/37/4096 and widths 1024/4096. Five warmups and one captured eager call each. Finite-output assertions passed; no full accuracy reference or timing claims.
- Saved traces, exact names/launch dimensions/resources, revision metadata and family summary under `results/2026-10-27-pytorch-dispatch-survey/` within Project 3. Verified register/shared/general implementations against official pinned source.
- Four families observed. FP16/BF16 width16384 uses shared input buffering; FP32 uses general fallback. No family change with row count in tested 1024/4096 cases. Private value buffers, dtype storage and row width affect dispatch; these are observations for the installed build/H100 only.
- Learner kernels unchanged. Next controlled retained-value exercise remains pending; files local/uncommitted and no independent mastery assessment.

## Follow-up — delegated register-buffered online kernel

- User explicitly requested implementing a new kernel using the original benchmark format. BUILD scope: new fixed-width-1024 variant, tests, timing integration, profiling selector and records. Existing kernels preserved.
- Implemented `softmax_online_registers<128>`: eight retained input floats/thread, first-value initialization and same conditional online recurrence, unchanged shuffle/shared pair merge, normalize retained originals after final pair publication. Compile-time width/unrolled indexing supports register allocation; not a dynamic-width implementation.
- Expanded host suite with eight width-1024 fixtures using existing independent reference/patterns/tolerances. All 62 scheduled runs passed. Compiled correctness/timing/profile harnesses with nvcc O3/C++17/lineinfo/sm_90/ptxas verbose; all exited 0. Ran four sanitizer tools on expanded suite: zero errors/hazards. Benchmark validations also passed.
- Nine rounds × 500 direct launches after 50 warmups: existing online median 16.873280 µs, new 13.823104 µs (mean 13.840555); warp 12.215808. Same-run improvement ~18.1%; fixed-width specialization/unrolling accompany buffering, so exact mechanism is unproven.
- Compiler: new 30 registers/thread, old 31; zero spills/stack bytes, 32 bytes shared each. No new runtime occupancy or Nsight metric capture. Files/logs/source hashes under Project 3 `results/2026-10-27-register-online/`.
- Added concise retained-input notes; learner conclusion not yet provided. Assisted code/results are not independent mastery. Next: paired Nsight capture to compare load traffic and stalls. Changes local/uncommitted; no push.

## Follow-up — register variant occupancy

- User requested occupancy. Captured LaunchStats/Occupancy for warmed-up online_registers at 4096×1024; one-pass report, exit 0 and reference check passed. Same profiling harness and uncontrolled clock/cache settings.
- Theoretical occupancy100%, achieved80.67%, active warps/SM51.63; 16 blocks/SM residency limit, four warps/block. Register count30. Not a ready-warp or compute-throughput metric; no specific cause assigned to the theoretical/achieved gap.
- Saved report/text/log under register-online results and updated progress. Next paired load/stall investigation still pending. Local/uncommitted.

## Follow-up — PyTorch occupancy comparison

- User requested comparison to PyTorch. Added profile_torch_softmax.py; one selected preallocated FP32 4096×1024 call after50 warmups, same Nsight sections/settings. One pass, exit0; FP64 framework-reference and probability checks passed.
- PyTorch theoretical62.5%, achieved39.67%,25.39 active warps/SM,48 registers/thread,10 register-limited blocks/SM; ours100%/80.67%,51.63 warps,30 registers,16 blocks. PyTorch grid1024 versus ours4096; both128 threads/block but four versus one row/block.
- Saved report/details/capture log in register-online results and revised occupancy note. No new benchmark, no full compute/memory counters. Prior faster PyTorch timing despite lower occupancy illustrates residency is not throughput; cause remains unisolated.
- Next paired load/instruction/stall comparison pending. Assisted interpretation, learner conclusion not yet provided; files local/uncommitted.

## Follow-up — full PyTorch versus retained-online counters

- User clarified instructions/stalls request. Collected full Nsight reports for both current kernels with matching shape, input, warmup and clock/cache settings; all captures/validations passed. Saved reports, full raw/detail exports and selected comparison CSV under register-online results.
- Ours9.517M versus PyTorch4.899M warp instructions, identical131072 global loads/stores, shared20480 loads+20480 stores versus0. Online conditional variant branch efficiency82.4%, active lanes27.01; PyTorch100%,32. Corrected applicability of earlier99.01% generic-online result.
- Long-scoreboard ratios1.300/1.364, barrier1.706/0, fixed wait2.429/2.235. Ours more residency/issue activity but more work; PyTorch finishes sooner even with more no-eligible cycles. Not-selected isn't wasted scheduler time. No causal percentage assigned to individual counters.
- Cache hit/DRAM traffic differences and uncontrolled clocks/replay limit interpretation. No new learner code or timing benchmark. Next one controlled change to algorithm or mapping; local/uncommitted.

## Follow-up — three source-level questions

- User requested theoretical instruction explanation, branch-efficiency cause and shared-memory use. Read source and exported branch-attributed CUDA/SASS from existing full report; no rerun/edit.
-111092 of119284 divergent targets map to the conditional per-element recurrence;8192 to merge empty-state guard. Explained per-lane maxima and differing branch choices, without treating target counts as lost-time fractions.
- Verified64-bit shared pair transfers and two block barriers. Four partial stores plus final store; one partial-load warp instruction plus four final-load warp instructions:5×4096=20480 each.
- Explained extra rescaling/pair-merge arithmetic and row mapping relative to retained-exponential PyTorch. Saved evidence and note. Next controlled algorithm/mapping experiment pending; assisted explanation, no learner conclusion or mastery claim, local/uncommitted.

## Follow-up — user-requested matched sweep

- User requested PyTorch/initial online/register-online sweep; clarified via async reply that initial means current conditional `softmax_online`. Used that baseline, no original generic-merge variant included.
- Generalized register kernel compile-time width with default1024 preserved; widths must be positive multiples of128. Added minimal shared CUDA bridge and same-process Python sweep; no per-element arithmetic/mapping change. Source hashes saved.
- Compiled CUDA shared library with O3/C++17/lineinfo/sm_90/PIC and verbose ptxas. Zero spill/stack bytes across six width specializations, registers22/24/30/40/56/80.
- Ran18 shapes ×3variants ×2modes ×9rounds ×500calls. Same fixture, independent NumPy FP64 reference and fixed criteria; all54 initial and every post-batch validation passed. Process exited0;972 samples/108 summaries saved under Project3 results/2026-10-27-softmax-sweep.
- Direct uses Python/ctypes/eager launches; graph captures500 launches per replay and excludes setup.50warmups; graph additionally replayed once before timing. Output preallocated for every variant, rotated order, warm reused buffers, uncontrolled clocks/caches. No per-call p99 or new-width sanitizer capture.
- Observed crossover at4096 rows: PyTorch wins128/512/1024; register-online wins2048/4096/8192 in both modes. This does not isolate why: specialization, mapping and PyTorch dispatch all vary with width.
- Updated kernel-contract note, benchmark documentation and progress. Next: matched profile at width2048. Learner conclusion pending; assisted work, no mastery claim; all files local/uncommitted/unpushed.

## Follow-up — Project 2 CUDA Graph revision note

- User requested saving the explanation. Added `project_02_performance_laboratory/notes/01-timing-and-setup/21-cuda-graph-replay-timing.md`, with short bullets, direct/replay comparison, procedure and illustrative Python timing code; updated section/root indexes.
- Captures covered concepts only: repeated500-call graph, persistent buffers, capture-stream requirement for custom kernels, warmup/event boundaries, batch-average interpretation and absence of automatic kernel fusion.
- Checked local note links and whitespace. Documentation only; example was not separately executed this turn. Prior sweep is the actual execution evidence. No new mastery claim, commit or push. Next active task remains investigating the width2048 crossover.

## Follow-up — full offline comparison handoff

- User requested updating test/new-kernel records and profiling all cases where retained online beats PyTorch, for offline Mac study. Selected11 shapes from saved sweep:9 wins in both modes plus2 direct-only cases (37×128,256×128); no cherry-picked omission of the graph reversals.
- Added `profile_sweep_case.py` with independent NumPyFP64 pre/post checks, NaN output initialization,50 warmups and one profiler-selected launch. Added `capture_winning_cases.py` for bounded240s per-kernel captures, full sections and text/raw exports. Added `build_offline_profile_index.py` for SASS exports, pair pages and manifest.
- Captured22 reports, each process exited0 and passed fixed reference/invariant checks. H100,CUDA13.0.88,NCU2025.3.1.0,PyTorch2.11.0+cu130. Uncontrolled caches/clocks, replay captures; six CTC counters unavailable. PyTorch binary lacks CUDA source mapping; bundled pinned upstream source/license separately. Our source embedded and snapshotted.
- All22 reports successfully imported; wrote per-case instruction/traffic/stall/occupancy tables plus raw/full details/SASS. Included original sweep summaries/raw samples/metadata/build logs and narrower validation/sanitizer evidence, with scope stated explicitly.
- Self-contained offline folder contains148 manifested files totaling114309514 bytes (manifest excluded). Verified every file size/SHA256 and local handoff links. Report/source snapshots are for current retained kernel; earlier reports remain historical.
- Updated root OFFLINE_PROJECT03.md, results index, status and project record. Next offline task: compare4096×2048 pair and form a testable explanation; no further GPU work required merely to study artifacts.
- No learner mastery assessment, Git commit/push, download verification on Mac or rental shutdown performed. All changes local/uncommitted/unpushed; verify transfer before shutting down.

## Follow-up — include PyTorch-faster cases

- User requested both sides of the sweep. Extended capture helper with --all-cases and verified-existing-report resume behavior; source snapshots checked against current kernel/bridge before reuse.
- Captured14 additional full reports:37×4096,37×8192,256×4096,256×8192,4096×128,4096×512,4096×1024, each for ours and PyTorch. All exited0 and passed independent FP64 pre/post validation. Existing22 reports retained, total36 across18 shapes.
- Regenerated comparison pages/raw selected metrics/SASS/checksums; corrected win labels for seven PyTorch-both-mode cases, nine ours-both-mode cases and two mode-dependent reversals. Updated root/results/note handoff indexes; no new benchmark or kernel changes.
- Same profiler/clock/cache/source-line limitations apply. Next offline task compares4096×1024 with4096×2048. Changes local/uncommitted/unpushed; no rental shutdown or verification of transfer to Mac.

## Follow-up — authorized Git handoff

- User authorized committing and pushing the code, benchmark scripts, revision notes, progress records and small validation logs.
- Project 3 `results/` is now ignored by Git, alongside Project 2 results and generated binaries/profiler reports. All local result files are preserved for separate download (about 200 MiB for the complete Project 3 results folder).
- Result links in these documents require restoring the downloaded folder at its original repository-relative location; a Git clone alone will not contain reports.
- This checkpoint prepares the authorized commit/push; confirm remote synchronization from Git after the push. Historical local-only entries below describe earlier checkpoints.
- Next: download results and open the 4096×1024 and 4096×2048 paired reports on the Mac before shutting down the rental. Download verification and shutdown have not been performed.
