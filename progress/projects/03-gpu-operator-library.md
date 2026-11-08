# Project 3: GPU Operator Library

## Softmax comparison checkpoint — 2026-11-08

- Saved the retained-online redundancy discussion in note 14: Warp also retains
  inputs and reuses its exponentials; our online recurrence adds rescaling and
  recomputes exponentials for output. Combined reduction is still a tradeoff.
- Proposed a same-mapping, 128-thread retained conventional-softmax control;
  no implementation or measurement. Existing wins do not isolate the recurrence.
- User authorized commit/push of accumulated work. Next: synthesize the five
  selected blog cases. GEMM resume preserved; understanding not reassessed.

## Wide-row softmax crossover analyzed — 2026-11-08

- Added note18 for4096×8192/16384: ours retains64→128 values/thread, registers80→168 and resident-block limit6→3. PyTorch switches Reg→General with32 registers, two1024-thread blocks/SM and vector4 memory operations. No spills at16384.
- General executes3× input load sectors but only~2.4% more DRAM reads than ours; L1/L2 hits absorb rereading. Vector accesses yield3/4 as many load instructions and1/4 stores. Our instruction advantage shrinks36.9→13.8% while issue falls48.52→32.72%; General issues46.44% at16384.
- Read four profiles and source/SASS with independent resource and memory audits. Saved exact metrics/units and benchmark-versus-profile timing distinction. No new GPU run, kernel changes, commit or push.
- Next: discuss this crossover, then synthesize the five selected cases into the blog narrative. Independent understanding not assessed; GEMM resume preserved.

## Row-count softmax crossover analyzed — 2026-11-08

- Added note17: the same Reg/retained Online specializations reverse winner as rows grow37→4096. Resident-row ceilings2 versus9 let ours gain inter-row latency hiding; issue16.66→58.57% while retaining ~30% fewer warp instructions/row. Reg still has higher occupancy/readiness/issue at the large shape.
- Read four raw/detail captures and sources/SASS with independent scheduler and memory audits. Larger input64MiB exceeds reported50MiB L2; read hits collapse and both stream similar DRAM traffic. Ours sustains2.67TB/s versus2.23TB/s, an outcome rather than an isolated cause. No spills; equal global sectors.
- Next: discuss this explanation, then4096×8192 versus4096×16384. Benchmark and profiler timings remain separate; no new GPU run, kernel edits, commit or push. Independent understanding not assessed; GEMM resume preserved.

## First softmax profile crossover analyzed — 2026-11-06

- Follow-up audit rechecked raw metrics and static SASS sites; expanded note16 with explicit algorithm-stage comparisons and equal global-load instruction counts redistributed over different numbers of warps. Kept per-SM placement as bounds and static instruction sites distinct from dynamic counts.
- Added note16 for37×2048/37×4096: profiles support work distribution/per-thread work at2048 and stronger within-row parallelism/latency hiding at4096. Torch changes10→37 blocks and1→32 warps per row; ours stays37 blocks/four warps per row.
- At2048 ours wins despite more instructions and lower active-cycle issue%; at4096 Torch wins despite more instructions, with issue59.72% versus16.66% and eligible2.10 versus0.17. Same input/store sectors within each shape, no spills, no DRAM saturation evidence.
- Read four raw/detail/SASS exports and source; retained benchmark medians separately from profiler durations. Findings are supported explanations, not isolated causal percentages. User's same-H100/environment/repeated-run clarification accepted; no rerun prerequisite.
- Next: discuss this explanation, then compare37×4096 with4096×4096. No GPU run or kernel changes; understanding not independently assessed. Changes local/uncommitted.

## Softmax blog case shortlist — 2026-11-06

- Proposed five retained-Online/PyTorch cases: 37×2048, 37×4096, 4096×4096, 4096×8192, 4096×16384. They connect width/family changes and a fixed-width row-count reversal, covering all three PyTorch families within the width16384 scope.
- Added note15 with working title, matched graph medians, direct cross-check, case questions and report links. Recomputed20 selected medians from180 raw batch samples; winner agrees across modes and sample ranges are separated within each saved run.
- Next: inspect the paired reports for37×2048 and37×4096 after stating hypotheses. No profiler-counter analysis or new GPU run during selection. Cross-run8192→16384 and graph-versus-eager timing scope remain explicit. Changes local/uncommitted; GEMM resume preserved.

## Retained online softmax algorithm — 2026-11-05

- Added note14 in the requested compact LaTeX/legend format: retained inputs, explicit conditional online updates, warp/block pair merges, empty-pair handling and output reuse.
- Read the recorded source/bridge and local learning source; verified all30 saved retained-online captures use block128, grid equal to rows and exact-width templates. Preserved the distinction between recorded `dims` and local `Cols` variants.
- Next: discuss retained Online, then compare algorithms with profiles for the plotted shapes. Explanation assisted; independent understanding not assessed. No new GPU run, kernel edit, commit or push; GEMM resume preserved.

## PyTorch general softmax algorithm — 2026-11-04

- Added note13: compact LaTeX algorithm/legend, explicit local max/sum, three vectorized input passes and the shared block-reduction helper from note12. Distinguished vector width4 from Reg's retained-slot count.
- Verified all12 extended PyTorch captures use General template4, block1024, grid equal to rows. Inspected source alignment/prefix/tail/output branches; algorithm scopes the aligned FP32 main path. No new GPU measurements or causal performance claim.
- Next: discuss General, then connect the three PyTorch algorithms to per-shape profiles. Independent understanding not assessed; changes local/uncommitted. GEMM resume point preserved.

## PyTorch register softmax algorithm — 2026-11-04

- Added note12 in the requested compact LaTeX/legend format: explicit local max/sum, retained logits, block-reduction scratch/barriers and output epilogue.
- Verified all six recorded Reg captures: 1024 threads/block, one block/row, template slot count4/8 for width4096/8192. Inspected local source and pinned upstream block-reduction helper; no new GPU run or causal performance conclusion.
- Next: discuss Reg's block-wide reduction and retained-input behavior, then the General kernel. Explanation assisted; independent understanding not assessed. Changes local/uncommitted; GEMM resume point preserved.

## Explicit local softmax reductions — 2026-11-04

- Added and then condensed capacity/log2, interleaving/coalescing and local-to-warp reduction explanations to three short bullets with examples, per user feedback. Independent understanding not reassessed.
- Expanded note11 with local maximum and local sum initialization/loops before each warp reduction; added `lmax`/`lsum` to the legend and renumbered the algorithm.
- User identified that the compact notation hid the local reduction; acknowledged the explanation. Independent transfer remains unassessed.
- Documentation only; checked step numbering, LaTeX brace balance, local links and whitespace. Next: continue the warp-softmax walkthrough before other kernel families. Changes local/uncommitted.

## Paper-style warp softmax algorithm — 2026-11-03

- Latest format correction: compact LaTeX algorithm with short mnemonic names, a legend, per-thread vector notation and one inline reduction helper. Replaced the crowded long-name version; mathematical work unchanged, independent row work regrouped for readability.
- User explicitly requested a single continuous algorithm box like the FlashAttention papers. Note11 now includes host dispatch, parameters, private storage, all device stages and both XOR reductions in that one box; explanatory notes follow below. Private buffer lifetime is logits → exponentials → output.
- Kept measured-width instantiations, capacity padding, mode/type variations and pinned sources. Recorded this presentation preference in learning instructions.
- CPU-model sanity checks against stable softmax passed for seven cases, including width17, width900, two-row batching and partial blocks. Local note links and `git diff --check` passed. These checks do not validate CUDA compilation, races or GPU performance.
- Next: discuss this algorithm before advancing to the Reg/General kernels. No mastery assessment, GPU run, commit or push; changes remain local. GEMM resume point remains saved.

## CUDA explanation depth adjusted — 2026-11-03

- Follow-up: the user also rejected the stripped-down version. Note11 now gives kernel-level reasoning that connects mapping, private storage, compile-time choices and each computation stage, without re-teaching CUDA basics.
- User clarified they know basic CUDA and found the expanded walkthrough excessive. Condensed note11 to kernel-specific mapping, computation and template choices; retained source references and relevant variations. Recorded the preference in learning instructions.
- Next remains discussion of this kernel, then other families/profiles. No kernel edits, measurements or mastery assessment; local documentation changes only.

## Warp softmax explanation restructured — 2026-11-03

- User found the separate template glossary and algorithm explanation confusing. Rewrote note11 as one [37,1024] input moving through launch, per-lane storage, maximum, exponentials/sum, and output; introduce parameter names when their purpose appears.
- Preserved six-parameter coverage, width900 padding, width128 batching, dtype/mode variations and pinned references. Recorded preference for concrete integrated explanations in the learning instructions.
- Documentation only; no new numerical/performance claims or mastery assessment. Next: discuss the revised walkthrough before advancing to other kernels. Changes local/uncommitted.

## Warp softmax template parameters — 2026-11-02

- Expanded note11 with all six type/value template parameters, accumulator-versus-storage precision, capacity bucket dispatch, derived private-array dimensions, compile-time mode flags and runtime arguments.
- Added the width900/capacity1024 worked example, clarified semantic masking versus bounds guards, and separated hardware warp size from subwarp grouping. FP32 unmasked benchmark scope retained; no new kernel implementation or measurements.
- Checked illustrative padding/grid arithmetic and note links. Next: discuss this mapping before later kernel families and profiling hypotheses. Understanding remains unassessed; changes local/uncommitted.

## PyTorch warp softmax algorithm — 2026-11-02

- Added blog note11 with pinned upstream/bundled source references, stable-softmax algorithm, width1024 lane mapping, private input-to-exponential reuse, XOR all-reductions and width128 two-row batching.
- Checked the bundled forward/helper/dispatch source and pinned upstream raw source. Described runtime width versus compile-time capacity and source-level differences from retained Online. No CUDA/source implementation edits or new GPU measurements.
- Next: learner discussion of this algorithm, then Reg/General walkthroughs and profile-based hypotheses. Source explanation is assisted; independent understanding not assessed. Changes local/uncommitted.

## Linear softmax plots — 2026-11-02

- At user request, regenerated both retained and non-retained plot sets with linear x/y axes and zero-based runtime. Row-specific y ranges25/40/400µs match across variants; small-width markers stay at their numeric positions, with overlapping value labels omitted below2048.
- Updated note10 and the plot script. Data/timing boundaries unchanged; no new GPU measurements. Next: algorithm walkthrough and profile interpretation.

## Retained Online comparison plots — 2026-11-02

- Added the requested retained-versus-PyTorch plots for rows37/256/4096 through width16384, using graph medians and matching log axes. Note10 now contains both plot sets; script selects the variant explicitly. Six new PNG/SVG files generated and visually checked; no new GPU run.
- Next remains the algorithm walkthrough followed by profile interpretation. Changes local/uncommitted.

## Softmax runtime plots — 2026-11-02

- Created note10 and three PNG/SVG plots for rows37/256/4096, width128–16384: PyTorch versus non-retained Online, using graph-replay medians. Common log axes, point labels, two-run provenance and timing scope are visible.
- Added a reproducible Matplotlib plotting script; validated42 input measurements, visually checked all three plots, and verified outputs/links. No new GPU run or kernel edits.
- Next: explain the algorithms behind the plotted families before attributing their performance. Saved tables still include retained online and direct mode. Changes local/uncommitted.

## Softmax blog comparison tables — 2026-11-02

- User requested a blog-oriented notes subsection, starting with the full kernel/shape/speedup inventory. Created `notes/05-softmax-comparison/` and note09 with all30 shapes × three variants, separate graph/direct tables, actual PyTorch names, and bold custom medians below PyTorch.
- Ratios use same-run/mode PyTorch median divided by custom median. Recomputed all180 source medians from1620 raw samples; checked30 recorded kernel pairs and all60 rendered table rows/120 ratios/highlights. Graph wins: online7/30, retained9/30; direct wins11/30 each. These are observed medians, not significance claims.
- Notes distinguish eager dispatch under graph replay from direct Python/ctypes submission, preserve two-run provenance, and keep exact-width specialization and unprofiled-online limitations explicit. No algorithmic cause established yet.
- Next: walk through the algorithms of PyTorch Warp/Reg/General and our Online/Retained variants before interpreting per-shape profile differences. No GPU execution or publication; local/uncommitted.

## Large-width softmax results checked — 2026-11-02

- Inspected the downloaded 2026-10-31 extension: rows37/256/4096 × dims16384/32768/65536/131072, FP32, N128. Timing includes PyTorch, conditional online, and retained online; profiles include only PyTorch and retained online.
- Verified all178 manifest files (334818377 bytes), including24 reports; zero missing/size/hash errors. Recomputed all72 medians from648 samples; all24 capture logs contain PASS and exit0. Four memchecks passed at three rows, one per new width; no broader sanitizer claim.
- PyTorch uses `cunn_SoftMaxForward<4,...>` in all12 new shapes. Retained kernel still specializes exact dims; the proposed runtime-width capacity experiment has not been performed in this package.
- Retained variant loses all12 shapes in both timing modes. Static spills begin at32768 dims (255 registers/thread), supported by measured local spilling requests. At4096 rows, non-retained online has lower observed medians than PyTorch at32768/65536/131072; it has no paired profiles here. No sole-cause performance claim.
- Bundled source differs from the local tracked source (generic N and larger-width dispatch); reviewed the snapshot without replacing code or pulling Git. No GPU run, commit or push.
- Active next step: explain the three PyTorch kernel families now represented, then compare algorithms and profiling evidence. GEMM indexing remains saved for later resumption. [Verification session](../sessions/2026-11-02-project-03-large-softmax-results.md).

## Naive GEMM foundations — 2026-10-28

- User selected GEMM as next lesson. First contract: contiguous row-major A[37,130], B[130,65], C[37,65], FP32 inputs/accumulation/output; alpha1/beta0, no transpose or fused epilogue.
- Mapping: one output/thread, 16×16-thread blocks with x across columns and y across rows. Learner to derive grid, guard and flat indexing before implementation.
- Note 08/section added. No code, tests, or timing; subsequent low-precision MMA/WGMMA contract preserved. Softmax follow-ups remain open.

## Mac results available and verified — 2026-10-28

- Pulled origin/main by fast-forward to b664fcb. Separately downloaded ignored results were already present and preserved.
- Full-sweep package verification passed for all 227 manifest files, including 36 `.ncu-rep` reports. Inspected recorded 62-run correctness and four sanitizer summaries; no new execution or broader sanitizer claim.
- Read current experiment history, sweep methodology and 4096×1024/2048 pair tables. Next: learner interprets the width-dependent crossover using saved reports, separating unprofiled timing from replay diagnostics.
- Verification/progress updates local and uncommitted. No GPU contact or shutdown action.

## Git handoff and separate results download — 2026-10-27

- User authorized committing and pushing the code, benchmark scripts, revision notes, progress records and small validation logs.
- Project 3 `results/` is now ignored by Git, alongside Project 2 results and generated binaries/profiler reports. All local result files are preserved for separate download (about 200 MiB for the complete Project 3 results folder).
- Result links in these documents require restoring the downloaded folder at its original repository-relative location; a Git clone alone will not contain reports.
- This checkpoint prepares the authorized commit/push; confirm remote synchronization from Git after the push. Historical local-only entries below describe earlier checkpoints.
- Next: download results and open the 4096×1024 and 4096×2048 paired reports on the Mac before shutting down the rental. Download verification and shutdown have not been performed.

## Offline profiles expanded to both winners — 2026-10-27

- User requested profiling the PyTorch-faster shapes too. Added14 reports for seven missing shapes; preserved22 prior reports. Complete package now has36 full reports across all18 sweep shapes, both kernels per shape.
- All added captures exited0 and passed independent FP64 pre/post reference and probability checks. Index labels nine ours-both-mode wins, seven PyTorch-both-mode wins and two direct-only ours wins. Diagnostic replay durations are separate from selection timings.
- Updated source snapshots/capture script, per-case comparisons, raw/details/SASS exports, checksum manifest and OFFLINE_PROJECT03.md. Same folder: `project_03_gpu_operator_library/results/2026-10-27-offline-profiles/README.md`.
- Next offline task: compare4096×1024 (PyTorch wins) with4096×2048 (ours wins). Changes local/uncommitted/unpushed; no rental shutdown or Mac download verification performed.

## Offline GPU handoff prepared — 2026-10-27

- User requested updated tests/kernel records and paired profiles for cases beating PyTorch. Captured all11 shapes that won in either sweep mode:22 full Nsight reports, each independently validated before/after. Nine cases won both modes;37×128 and256×128 only direct mode, explicitly labeled.
- Self-contained114.3MB folder: `project_03_gpu_operator_library/results/2026-10-27-offline-profiles/README.md`. Includes per-case metric comparisons, details/raw/SASS exports, source snapshots, pinned PyTorch source/license, sweep/validation evidence and verified148-entry SHA256 manifest. All reports imported successfully to text/CSV/SASS.
- PyTorch binary lacks CUDA line mapping; separate pinned source provided, not falsely presented as embedded mapping. Replay caches/clocks uncontrolled and six CTC counters unavailable; logs preserve warnings.
- Added `OFFLINE_PROJECT03.md`, results index, reusable capture and index scripts. Updated test scope:62-case retained-width1024 suite and four sanitizers;54 sweep comparisons/972 timings;22 profile validations. No broader sanitizer coverage claimed.
- Next offline task: compare4096×2048 pair to explain crossover before choosing another change. No GPU required to read saved reports. Project3 still active; assisted work, not independent mastery.
- Changes local/uncommitted/unpushed. No rental shutdown performed; verify download before shutting down.

## Matched softmax shape sweep — 2026-10-27

- User confirmed current conditional online baseline. Compared PyTorch preallocated, online and register-online across18 FP32 shapes (rows37/256/4096; widths128/512/1024/2048/4096/8192), direct and graph timing modes. All54 initial reference comparisons and after-batch validations passed;972 samples saved.
- Added stream-aware CUDA bridge and Python sweep, generalized retained kernel to compile-time widths divisible by128 (default1024 preserved). Static compiler reports no spills,22–80 registers/thread across widths.
- At4096 rows, PyTorch fastest through1024; retained online fastest at2048/4096/8192 in both modes. Direct medians at8192: PyTorch147.598µs,online162.498µs,retained103.338µs. No universal speedup or causal bottleneck claim.
- Results/methodology: `project_03_gpu_operator_library/results/2026-10-27-softmax-sweep/README.md`. Same50 warmups,9×500 batch format; graph mode separates per-operation host submission. No per-call p99 or new-width sanitizer suite.
- Next: profile width2048 crossover before another optimization. Assisted implementation/measurement; local/uncommitted/unpushed.

## Online kernel source attribution — 2026-10-27

- Investigated user's three questions using existing report/source; no GPU rerun or kernel edit. Divergent-target attribution:111092 at specialized value≤running-max update,8192 at merge empty guard.
- Confirmed shared pair exchange in SASS:64-bit loads/stores plus two block barriers; five shared warp loads and five stores per block explain20480 each across4096 blocks.
- Extra instruction work explained by online rescaling/pair merges, output exponential computation, divergence and four-warps/row communication, versus PyTorch scalar reductions/retained exponentials/one warp per row. No exact causal runtime percentages claimed.
- Details/source export saved with register-online results and revision note updated. Next controlled algorithm/mapping experiment pending; local/uncommitted.

## Full PyTorch instruction/stall comparison — 2026-10-27

- User clarified request; captured full reports for register-buffered online and PyTorch at 4096×1024. Both validation checks passed. Ours/PyTorch: 9.517M/4.899M executed warp instructions, identical131072 global loads, shared-load/store20480 each versus0.
- New conditional online branch efficiency82.4% versus100%, active lanes27.01/32; barrier stall ratio1.706/0. Long-scoreboard ratios similar1.300/1.364. Earlier generic-online low-divergence conclusion does not apply to this changed kernel.
- Reports/CSV/interpretation saved under register-online results. Stall ratios are not runtime percentages; warm-cache replay differences limit causal traffic comparisons. No source edits or new benchmark.
- Next: select one controlled algorithm/mapping change to reduce instruction and cross-warp work. Local/uncommitted; no mastery claim.

## PyTorch versus retained-online occupancy — 2026-10-27

- Matching Nsight LaunchStats/Occupancy capture for PyTorch FP32 4096×1024 passed post-profile validation. Theoretical/achieved occupancy: PyTorch62.50%/39.67%, ours100%/80.67%. Registers48/30, register block limits10/16; grids1024/4096, both128 threads/block.
- PyTorch uses four rows/block versus one, reducing launched blocks; lower occupancy does not imply slower execution. Earlier separate-run timings favored PyTorch; no new speed measurement or causal isolation performed.
- Report and caveats appended to `project_03_gpu_operator_library/results/2026-10-27-register-online/README.md`. Added reproducible PyTorch capture harness. Next: paired load/instruction/stall analysis remains pending; changes local/uncommitted.

## Register-buffered kernel occupancy — 2026-10-27

- Measured at 4096×1024 using Nsight LaunchStats/Occupancy: theoretical 100% (64 warps/SM), achieved 80.67% (51.63 active warps/SM). Register/warp block limits both 16; 30 registers/thread.
- One-pass capture and post-profile correctness passed. Report saved in `project_03_gpu_operator_library/results/2026-10-27-register-online/occupancy.ncu-rep`. Occupancy is residency, not compute utilization. Next load/stall comparison remains pending; changes local/uncommitted.

## Register-buffered online variant implemented and measured — 2026-10-27

- User delegated new kernel and original benchmark format. Added fixed-width-1024 `softmax_online_registers<128>`, retaining eight inputs/thread with unchanged online recurrence and warp/block merging. Existing variants preserved; timing/profiling dispatch extended.
- Added eight targeted fixtures: all 62 correctness runs passed; four sanitizer tools passed with zero errors/hazards. Compile reports: new variant 30 registers/thread, existing online 31; neither spills.
- Same-run 4096×1024 online versus retained medians: 16.873 versus 13.823 µs (~18.1% reduction); warp 12.216 µs still faster. Compilation specializes/unrolls width too; no sole-cause load-latency claim.
- Results: `project_03_gpu_operator_library/results/2026-10-27-register-online/README.md`. Same 50 warmups/nine rounds/500 calls benchmark, unchanged correctness tolerances. Added quick lesson note.
- Next: paired profiling of existing/new online for load traffic and stalls. Assisted implementation, no independent mastery assessment. Changes local/uncommitted/unpushed.

## PyTorch dispatch survey — 2026-10-27

- Captured 57 eager CUDA softmax cases: 17 widths × three dtypes plus six row-count checks. All completed with finite output; dispatch inspection only, not benchmarks/full reference validation.
- Observed warp, register-buffered, shared-memory-buffered and general block kernel families. Width 16384 selects shared memory for FP16/BF16 but general fallback for FP32. Row count did not change families in tested 1024/4096-width cases.
- Reproducible script and per-case traces/metadata: `project_03_gpu_operator_library/results/2026-10-27-pytorch-dispatch-survey/README.md`. Version/H100/layout-specific results; kernels unchanged, files local/uncommitted.
- Next: implement one retained-value experiment while preserving mapping before separately testing PyTorch's one-warp-per-row strategy.

## Exact PyTorch softmax dispatch identified — 2026-10-27

- Captured both output modes at FP32 4096×1024: same `softmax_warp_forward<float,float,float,10,false,false>`, grid 1024, block (32,4), 48 registers/thread, no shared memory in trace.
- Matched installed PyTorch 2.11.0+cu130 revision and PersistentSoftmax.cuh: one warp/row, 32 saved values/thread, max then retained exponentials then sum/normalization, warp-only reductions. This selected path is not online normalization.
- Traces, metadata and pinned source links: `project_03_gpu_operator_library/results/2026-10-27-pytorch-implementation/README.md`. Added reproducible dispatch-capture script. No new timing or independent spill validation.
- Next: one controlled retained-value experiment before separately changing row mapping. Assisted source investigation; changes local/uncommitted.

## Specialized online update validated — 2026-10-27

- At user's request, implemented only private per-element specialization, then compiled and ran after separate authorization. All 30 softmax correctness cases and all four sanitizer tools passed with unchanged tolerances; timed workloads also passed validation.
- Updated online medians: 3.191 µs at 37×130; 17.126 µs at 4096×1024 (mean 17.108 µs). Earlier large median 17.772 µs; ~3.6% lower across separate runs, not a controlled causal estimate. Current warp median 12.543 µs remains faster.
- Results/build logs/source hash: `project_03_gpu_operator_library/results/2026-10-27-specialized-online/README.md`. No new Nsight capture. Next: profile specialized online to compare instruction work, divergence and load-dependency stalls. Changes local/uncommitted; implementation assisted, not independent mastery.

## Online softmax divergence checked — 2026-10-27

- Reused saved Nsight reports: online branch efficiency 99.01% versus warp 100%; active threads/instruction 31.38 versus 32. Divergence is present, but aggregate lane participation remains high.
- Source export attributes 8,192 divergent branch targets to two machine branches mapped to `merge_online`'s `if (b.s == 0)` guard. Not evidence of widespread data-dependent divergence in every update.
- Added source export and interpretation to `project_03_gpu_operator_library/results/2026-10-27-softmax-ncu/README.md`. No new kernel execution or code changes; local/uncommitted. Next: choose one controlled optimization, preserving correctness.

## Quick PyTorch softmax baseline — 2026-10-27

- User requested a quick framework timing before further load-stall investigation. Added `benchmarks/time_torch_softmax.py` under Project 3; eager FP32, matching inputs, 50 warmups, nine rounds × 500 calls and unchanged numerical criteria. All checks passed.
- H100/PyTorch 2.11.0+cu130: at 4096×1024, default output median 12.685 µs; preallocated output 10.175 µs. At 37×130: 5.337/4.663 µs. CUDA events include submission gaps; Python and C++ host paths differ, so this is not a pure kernel-latency ranking.
- Results: `project_03_gpu_operator_library/results/2026-10-27-pytorch-softmax/README.md`. No PyTorch kernel identity or bottleneck captured. Changes local/uncommitted.
- Next: return to controlled load-dependency investigation; framework speed alone does not explain the mechanism.

## Softmax Nsight Compute diagnosis — 2026-10-27

- Captured baseline/warp/online at FP32 4096×1024 after 50 warmups; all captures and post-profile correctness checks passed. New support harness only; kernels unchanged.
- Online versus warp: 11.153M versus 7.324M executed warp instructions (+52.3%), one-third fewer global-load instructions, long-scoreboard cycles/issued instruction 6.26 versus 2.69. Registers 26 versus 30; no spills; achieved occupancy 79.4% versus 80.5%.
- Evidence supports extra instruction work and exposed load-dependency latency, not a saturated exponential unit or DRAM bandwidth ceiling. Stall fractions are not wall-time fractions; precise causal contributions remain unproven.
- Reports with embedded source, text/CSV exports, capture warnings and interpretation: `project_03_gpu_operator_library/results/2026-10-27-softmax-ncu/README.md`. Each report used 38 replay passes, uncontrolled caches/clocks; profiling durations are diagnostic only.
- Next: review the two reports side by side, then choose one controlled optimization to test. Harness/results/notes remain local, uncommitted and unpushed.

## First CUDA timings — 2026-10-26

- Added delegated standalone `project_03_gpu_operator_library/benchmarks/time_cuda.cu`; reuses existing kernels unchanged. All seven variants timed across 12 variant/shape combinations; correctness passed on every workload before and after timed batches.
- H100, 50 warmup launches/variant, 9 rounds × 500 direct launches, CUDA-event batch averages; variant order rotates. Allocation/transfers/validation excluded; submission gaps and warm-cache reuse included. No profiler or sanitizer during timing.
- At 4096×1024: row sum baseline/warp median 5.405/5.141 µs; softmax baseline/warp/online 13.108/12.270/17.772 µs. Online slower here; cause unproven. Small row-sum timing ranges overlap.
- Raw rounds, summary and reproduction details: `project_03_gpu_operator_library/results/2026-10-26-cuda-timing/README.md`. Single-process evidence, no universal speed or framework comparison claim.
- Next: interpret why online softmax can lose despite fewer logical traversals, then choose one controlled follow-up measurement. Harness, results and progress remain local/uncommitted.

## H100 correctness and sanitizer validation — 2026-10-26

- Pulled `9882f73` and compiled the unchanged CUDA kernels with CUDA 13.0.88 (`-std=c++17 -O3 -lineinfo -arch=sm_90`). GPU: H100 80GB HBM3; driver 580.173.02.
- Passed: CPU softmax reference/validator tests, affine (1,000 outputs), bias + ReLU (4,810 outputs), baseline/warp row sum (20 variant/case runs), baseline/warp/online softmax (30 variant/case runs). Existing references and tolerances unchanged.
- All four CUDA executables passed memcheck, racecheck, synccheck and initcheck: 16 sanitizer runs, zero reported errors/hazards. This validates tested workloads, not all possible inputs or independent mastery.
- Evidence: `progress/evidence/2026-10-26-project-03-validation/`; session: `progress/sessions/2026-10-26-project-03-execution-preflight.md`. No kernel source changes or performance measurements.
- Next: establish a matched CUDA-event timing harness for baseline/warp/online softmax, separating warmup, allocations and transfers from repeated kernel execution. Validation notes/evidence remain local and uncommitted.

## Commit/push checkpoint — 2026-10-26

- User requested saving accumulated Project 3 work to Git. Kernel/harness/notes/roadmap scope includes initial elementwise operators, both row sums, and all three softmax variants. GPU compilation, correctness and performance remain untested.
- Recommended next offline task: retained-values softmax with unchanged thread mapping, followed by GPU comparisons and backward. Subsequent direction: language ports, naive→shared/register-tiled→MMA→WGMMA GEMM and attention; RMSNorm can be a smaller intervening exercise. These are plans, not completed lessons.
- Prepared progress for handoff; final Git command results establish commit and push outcome.

## Three-variant softmax host suite — 2026-10-26

- Added user-requested online launch branch and label; loop count derives from three labels. Existing ten cases, references, tolerances and per-launch NaN/error checks retained; 30 variant/case runs scheduled.
- Source dispatch reviewed and `git diff --check` passed. No CUDA compilation or GPU test; kernel arithmetic unchanged. Next: deferred GPU correctness and synchronization validation, then benchmark matched variants.

## Online cleanup source-reviewed — 2026-10-26

- Confirmed duplicate load stage removed, explicit empty padding retained, and two publication barriers correctly placed. Online implementation is source-reviewed after guided corrections; runtime correctness and performance remain untested.
- Next: add online variant to the shared softmax host suite without changing cases/reference/tolerances. No source edits during this review.

## Online kernel shared exchange corrected — 2026-10-26

- Source now publishes each warp summary before reads, merges warp summaries with a defined offset loop/full participation, publishes the final pair and broadcasts it before output. No blocking source issue found for intended finite-input workloads.
- Duplicate warp-0 load/padding stage and middle block barrier are redundant. Current later padding works because earlier stage already assigned empties; cleanup must retain explicit empty assignment rather than merge-with-empty.
- Next: learner simplifies duplicate load stage, then extend host harness to online variant. No compilation/device tests, performance evidence, or assistant kernel edits.

## Online kernel accumulation/warp review — 2026-10-26

- Private singleton accumulation is correct. First shuffle loop fetches both fields before assignment and produces a correct lane-0 summary; omitted valid-source guard causes unused upper lanes to self-merge but does not corrupt lane 0's reduction tree.
- Later stages read unwritten shared storage, use undefined `d` and guarded full-mask shuffles, and omit final shared publication/all-thread broadcast. Next targeted correction: lane-0 per-warp shared stores and block barrier before final-warp reads.
- Source inspection only; no assistant code edits or execution evidence.

## Online helper correction verified — 2026-10-26

- Saved helper now uses exp(a.m-result.m) and exp(b.m-result.m). No remaining source issue found for valid finite-input summaries, including empty identity cases.
- Next: learner implements online TODOs 1–2; helper follows guided correction, not independent transfer. No source edits or execution during review.

## Online helper review — 2026-10-26

- Empty-summary handling is correct. Combined maximum is correct; both exponential factors have reversed signs. Explained that shifting to a larger maximum must shrink prior contributions.
- Next: fix rescaling to old maximum minus combined maximum. No implementation edits, compilation, or GPU testing during review.

## Online softmax scaffolding — 2026-10-25

- User requested helper/kernel skeletons. Added two-float state, helper empty/rescale TODOs, and online kernel shared storage/identifiers/empty state with staged reduction/publication/output TODOs.
- Helper returns explicit NaN placeholder until implemented. Host still runs only completed baseline/warp variants; no kernel body supplied or existing arithmetic changed.
- Next: implement merge helper and online TODOs 1–2. No CUDA build/device test; scaffold is assisted support.

## Online recurrence and pair merge — 2026-10-25

- Learner correctly answered rescaling, unchanged-max contribution, example pair merge, empty-empty NaN, and final normalization. Assistant supplied general formulas, empty-group handling and full-shuffle participation requirements.
- Note 06 now contains these explanations. Next: learner writes pair-merge helper, then uses it in thread/warp/block online normalization. No code or tests this follow-up; independent implementation/transfer unassessed.

## Online softmax foundations — 2026-10-24

- User selected online normalization as next optimization lesson. Initial target: maintain maximum and shifted exponential sum together; derive rescaling before parallel pair reduction and implementation.
- Compared current three logical input traversals to planned two; distinguished from DRAM traffic and universal speedup. Original paper checked; note 06 added.
- Next learner task: rescale the partial sum for [2,3] when 5 arrives. No implementation supplied or online performance evidence; GPU batch for existing variants remains deferred.

## Both-variant host harnesses — 2026-10-24

- Delegated host scaffolding now validates baseline and warp kernels independently against common references, with per-launch NaN sentinels and labeled pass/fail diagnostics. Row sum: ten widths including original 130 and warp/block boundaries; softmax: unchanged ten-case suite/tolerances. Each executable schedules 20 variant/case runs at N=128.
- Kernel bodies preserved; only stale launch-status comments updated. CPU softmax reference/validator tests passed with clang++ C++17 warnings-as-errors. Verified closed-form row-sum references and bounds guaranteeing exact FP32 integer sums for all 370 fixture rows.
- No nvcc on PATH; neither CUDA harness compiled/executed. Next: deferred GPU compilation/correctness and memory/synchronization checks, then measured performance. Host work is assisted support, not learner mastery evidence.

## Warp softmax source review complete — 2026-10-24

- Verified both fixes in saved source: private `float row_max` declaration and sum partials stored to `warp_partials[warp]`. Reviewed full reduction, identity padding, publication/reuse barriers, and normalized output; no remaining source correctness issue found for the declared contract.
- Guided implementation is complete; independent transfer and device correctness are untested. Stale skeleton/TODO comments remain cleanup.
- Next: baseline/warp host validation on matching inputs for both row sum and softmax, then deferred GPU compilation/correctness before timing. Current hosts still launch only baselines; no GPU activity or speedup claim.

## Warp softmax first review — 2026-10-24

- Private and warp maximum reductions are correct. Later stages are also filled; source review found undeclared `row_max` and a sum-partial write race from every warp leader targeting shared slot 0. Other slots would retain old maxima.
- Max publication, read-before-reuse barrier, and final denominator publication barrier are correctly placed. Next: correct the declaration and per-warp sum destination before final review. No assistant code edits or device testing.

## Warp softmax skeleton — 2026-10-24

- Added user-requested `softmax_warp` scaffolding in `kernels/cuda/softmax.cu`. Shared warp partials double as the final result slot; TODOs cover max reduction, broadcast/read-before-reuse barriers, sum reduction and normalization.
- Baseline and host harness unchanged; variant is incomplete and not launched. Exponentials will still be recomputed for output to isolate the reduction change.
- Next: learner implements TODOs 1–2. No CUDA compilation, device validation, or performance measurement.

## Warp row-sum source review complete — 2026-10-24

- Saved implementation now loads warp totals with zero padding before warp 0's final shuffle; participation, shared publication/barrier, and output ownership are correct under the declared contract. Guided implementation, not independent transfer evidence.
- Redundant initial shared zeroing and stale skeleton/TODO comments remain cosmetic cleanup. Host still launches baseline; no CUDA compile/run or performance evidence.
- Next: softmax warp variant with all-thread maximum/denominator broadcasts and shared-buffer lifetime barriers.

## Warp row-sum follow-up — 2026-10-24

- Corrected with guidance: shuffle participation, shared-store ownership, final warp selection and output ownership. Block barrier is correctly unconditional.
- Pending: load shared warp partials into warp 0's register values, with zero identities for lanes beyond `num_warps`, before the second shuffle loop. Current version still re-reduces old values.
- No kernel edits or device validation; next targeted task is the missing second-stage input assignment.

## First warp implementation review — 2026-10-24

- Strided private accumulation is correct. Full-mask shuffles inside `lane < d` are invalid participation; next targeted correction is making the first shuffle loop unconditional across each warp.
- Later stages also need correction: lane-0-only shared stores, warp-0 shared-partial loading with identities, and a single final output writer. No assistant kernel edits or GPU test; implementation remains incomplete.

## Warp row-sum skeleton — 2026-10-24

- User requested scaffolding; assistant added `row_sum_warp` with complete-warp template constraints, one shared slot per warp, identifiers, and staged TODOs. Existing baseline and host launch unchanged.
- Next: learner implements TODOs 1–2, then review before cross-warp merging. Skeleton is assisted support, not implementation mastery. No CUDA compilation/device execution.

## Warp-reduction synchronization reasoning — 2026-10-24

- Learner answered neutral values (0 / negative infinity), all-thread softmax maximum use, and the shared maximum overwrite hazard correctly after guided explanation. Assistant clarified block-wide synchronization versus shuffle scope.
- Added [note 05](../../project_03_gpu_operator_library/notes/02-reductions/05-warp-reductions.md). Implementation and independent transfer remain unassessed.
- Next: learner adds a separate warp row-sum kernel alongside the baseline, beginning with private accumulation and within-warp reduction. No kernel edits or GPU activity this follow-up.

## Warp-reduction optimization — 2026-10-23

- User requests optimizing both row sum and softmax with warp reductions. Preserve shared-tree versions for comparison; keep first variant at 128 threads/block.
- Teach register exchange via `__shfl_down_sync`, full-warp mask participation, lane/warp identifiers, and a two-level reduction with four shared warp results.
- Row sum needs only the final writer; softmax needs maximum/denominator visible to all warps and safe reuse of shared storage. Shuffle synchronization is not a block memory barrier.
- Hypothesis: reducing shared-memory work/block barriers improves runtime; current kernels and new variants still require actual GPU validation/timing. Do not change softmax exponentiation policy in this experiment.
- Next: neutral values for inactive-data lanes in the merging warp, then learner implements row-sum variant. No warp code or performance evidence yet.

## Softmax host harness completed — 2026-10-23

- User delegated support work: added checked CUDA launch/allocation/transfers/free, NaN output sentinel, stable FP64 reference, and 10 cases in `softmax.cu` / `softmax_test_support.h`. Learner kernel arithmetic and barriers unchanged; learner had already removed the redundant final barrier before this edit.
- Cases cover original [37,130], short rows/inactive threads, 127/128/129 boundaries, longer rows, equal/negative/large logits, tied maxima and dominant tail values.
- Elementwise tolerance 1e-7 + 1e-5*abs(reference), row-sum tolerance 1e-5, plus finite/[0,1] checks. FP64 reference promotes the actual stored FP32 inputs before subtracting the maximum and uses sequential summation.
- `softmax_reference_test.cpp` compiled with clang++ C++17, -Wall -Wextra -Werror, and passed on the Mac. Analytic cases, shift invariance, and corrupted-output rejection tested; no CUDA kernel execution.
- Next: GPU compilation/validation batch when user is ready; measured optimization afterward. Host implementation is delegated support, not learner mastery evidence.

## Stable softmax first implementation review — 2026-10-23

- Reviewed learner's saved kernel: correct max identity and tree, barrier after copying row maximum before shared reuse, exponential sum/tree, and per-element normalized output.
- Applies to the declared contiguous finite-input contract, positive cols, one block per row and matching power-of-two N threads. Source review only; no device correctness result.
- Extra barrier after reading denominator is unnecessary because no subsequent shared write occurs. Left learner code unchanged.
- Next: host correctness harness with stable double reference and explicit floating-point error criteria, including negative/equal/large logits and tail cases. No independent unfamiliar-case transfer assessed.

## Softmax skeleton — 2026-10-23

- Learner correctly supplied max identity/combine and barrier location for safe shared-array reuse after the posed hazard.
- User requested skeleton; created `softmax.cu` with template/signature, shared array, row/thread identifiers, and TODOs. Kernel body remains learning-target work.
- First baseline uses FP32 and recomputes exponentials during output. Next: learner fills TODOs; host harness and device validation follow later.

## Stable softmax started — 2026-10-23

- Contract: contiguous FP32 [37,130] input/output, FP32 intermediates, one 128-thread block per row, finite inputs, no masking.
- Learner correctly explained overflow and cancellation under a common maximum shift before an algebraic solution was supplied. This establishes immediate conceptual reasoning, not kernel implementation mastery.
- Next: choose the maximum reduction identity and combine operation; stable softmax implementation and GPU validation remain pending.
- [Note 04](../../project_03_gpu_operator_library/notes/02-reductions/04-stable-softmax.md); [session](../sessions/2026-10-23-project-03-softmax.md).

## Confirmed matrix precision contract — 2026-10-22

- User accepted FP16 → FP16 accumulator → FP16 output and BF16 → FP32 accumulator → BF16 output as primary MMA/WGMMA variants.
- Earlier open BF16 accumulation choice is resolved. FP16 with FP32 accumulation remains a comparison; softmax statistics/reduction precision is specified separately.
- Roadmap updated; no implementation, measurements, commit, or remote synchronization.

## Accumulation preference and ISA limit — 2026-10-22

- User clarified desire for low-precision accumulation too. Native FP16 accumulator/output path is now in scope, with FP32 accumulation retained for comparison.
- NVIDIA PTX supports f16 accumulators for f16 MMA/WGMMA, but the planned bf16-input forms require f32 accumulators. Native BF16 accumulator request is unsupported here; selecting a software rounding experiment remains optional, not automatically substituted.
- This supersedes the earlier blanket FP32-accumulation plan. No advanced kernel implementation or measurements.

## MMA/WGMMA output dtype — 2026-10-22

- User requested FP16 and BF16 outputs. Planned paths: FP16 → FP32 accumulator → FP16 output; BF16 → FP32 accumulator → BF16 output.
- FP32 accumulation remains the existing plan; final output conversion occurs in the epilogue. This does not request FP16/BF16 internal accumulation. Match output precision and tolerances across comparisons.
- Documentation only; no new implementation or measurement.

## Inline PTX preference — 2026-10-22

- User chose SageAttention-style CUDA device wrappers using `asm volatile` for explicit MMA/WGMMA work. Added to canonical roadmap.
- Study operand constraints, fragment ownership, ldmatrix/copies, shared descriptors, fences/commit/wait and buffer lifetime; validate small tiles before full kernels. Volatile assembly does not itself synchronize GPU threads or wait for asynchronous MMA completion.
- SageAttention source is a wrapper-design reference, not a commitment to its quantization/accuracy policy. Current learning task remains stable softmax; no advanced code or GPU action yet.

## Performance scope revision — 2026-10-22

- User explicitly expanded the target to efficient forward/backward kernels and architecture-specific Tensor Core paths. Canonical plan: [Project 3 roadmap](../../roadmap.md#project-3-build-a-gpu-operator-library).
- Softmax: stable baseline, row-resident fusion, online/tiled variants selected by shape, backward. GEMM: naive, shared/register tiling, warp MMA, asynchronous pipelines, Hopper WGMMA; dA/dB and fused epilogue gradients. Attention: framework reference then custom tiled MMA/online softmax, WGMMA overlap, and separately optimized dQ/dK/dV.
- Proposed language workflow: interleave each operator across CUDA C++, Triton, and TileLang. CUDA owns explicit hardware understanding; Triton block programs plus Gluon when explicit WGMMA control is needed; TileLang tile layouts and pipelines. Core forward/backward coverage in all three, without every pedagogical variant duplicated.
- Performance acceptance uses fixed workload suites, precision/gradient criteria, strong named baselines, per-shape forward/backward timing and memory, and targets set before tuning. No claim of universally faster online softmax or WGMMA, and no performance guarantee.
- Current evidence unchanged: three source-reviewed CUDA kernels, no GPU compilation/execution. Next offline lesson is stable fused softmax; remote validation remains deferred until user is ready.
- Planning only; no dependency changes or advanced kernel implementation. See [session](../sessions/2026-10-22-project-03-performance-plan.md).

## Row-sum host scaffold completed — 2026-10-21

- Explicitly delegated support implementation: host code in `row_sum.cu` allocates [37,130] input and [37] output, launches one 128-thread block per row with matching template size, checks CUDA calls, compares to sequential double CPU sums, and cleans up.
- Fixture uses small integer-valued FP32 inputs; exact comparison is valid here. Distinct row totals and nonzero tail sums exercise row selection and extra per-thread loads. NaN output initialization catches unwritten rows.
- CPU-only arithmetic model, including FP32 rounding at each addition, matches all 37 sums. This does not validate CUDA compilation, indexing on device, barriers, or memory safety.
- All three initial operators have correctness harnesses; next planned milestone is deferred GPU batch validation. No remote action, timings, or mastery claim.

## Template correction delegated — 2026-10-21

- User explicitly requested the fix; assistant replaced undefined template parameter type with `int`. Kernel body unchanged.
- Next: host correctness harness, followed by the deferred GPU batch. Template syntax fix is assisted; no compilation or execution evidence.

## First row-sum implementation review — 2026-10-21

- Body correctly implements taught reduction with unconditional barriers and a single output writer, assuming actual block size equals power-of-two template N.
- Current `template <block_size N>` uses an undefined type; explained integer template parameter syntax and launch/template agreement. First version remains N=128.
- Learner implemented body after guided algorithm; no independent transfer or device correctness claim. Next: correct declaration, then host scaffold and deferred GPU batch.

## Row reduction lesson — 2026-10-21

- Correct learner responses after instruction: column assignment for 130/128, tree round yielding 16 and 20, and explanation that all threads must reach each barrier in this algorithm.
- Supplied `row_sum.cu` skeleton with fixed 128-slot shared array and TODOs. No reduction implementation supplied. Kernel takes cols; grid size supplies row count, so launch exactly one block per row.
- Next: learner writes the kernel body; host scaffolding and GPU validation follow. Immediate lesson responses do not establish independent implementation or transfer mastery.
- [Lesson note](../../project_03_gpu_operator_library/notes/02-reductions/03-row-sum-and-barriers.md).

## Bias + ReLU harness added — delegated support work

- User explicitly delegated scaffolding. Assistant added host code in `bias_relu.cu`, preserving kernel logic: [37,130] fixture, checked CUDA calls, NaN output sentinel, nested CPU reference, capped mismatch diagnostics, cleanup, and return status.
- CPU-only Python fixture check confirmed 4810 flat/reference matches, sign crossings in both directions, zero sums, and 2316 differences from the earlier incorrect operation order. Not a test of compiled GPU code.
- GPU execution remains deferred. Next learning target: row reduction and cooperative work assignment. Harness completion is delegated support, not learner mastery evidence.

## Bias + ReLU review — 2026-10-21

- Learner supplied correct element count, guard, and modulo bias selection before implementation.
- Corrected operation order and undeclared variable after targeted review. Latest source implements guarded `value = x[idx] + bias[idx % cols]`, then ReLU.
- Source-reviewed only for the initial finite FP32, contiguous, small-shape contract. No compilation, GPU run, or independent-transfer claim.
- Next: add a host correctness harness with signed inputs and distinct biases before the row-reduction lesson. GPU validation remains batched for later.

## Offline batch requested — 2026-10-21

- User explicitly deferred first GPU run to batch a few easier kernels on the Mac.
- Proposed batch: affine (source-reviewed), contiguous 2D bias + ReLU, row sum. The reduction introduces cooperation/synchronization and should remain a small first version.
- Next active task: derive output and bias indexing for FP32 X[37,130], bias[130], separate Y[37,130]. No implementation supplied yet.
- After the batch: compile and validate each kernel separately, then benchmark/profile only validated cases. Source review and CPU reasoning do not establish device correctness or performance.

## Final first-version source review — 2026-10-21

- Learner completed the reviewed fixes, including explicit standard-library header, CUDA error checks, and result-dependent exit status. No blocking source issue found for the current small FP32 test.
- N=1000, block size 256: four blocks, 24 threads skip accesses. CPU expected values are exactly representable for this case.
- **Next:** verify remote GPU/toolchain, compile, and run correctness validation. All earlier pending source-fix entries below are historical; compilation and execution remain untested.
- Changes local/uncommitted; no implementation edits by assistant during final review.

## Host harness follow-up — 2026-10-21

- Current source adds N=1000, block size 256, device allocations, input upload, kernel launch, output download, comparison, and cleanup.
- Corrected with review: declaration order, input initialization, host-array lifetime, and output upload that could conceal missing writes when prefilled with expected values.
- Comparison is valid for the current exactly representable small-integer case; no general floating-point acceptance criterion established.
- Pending: explicit exit status/success message and CUDA API/launch/execution error checks, then GPU compilation/validation. No execution evidence or new independent-transfer claim.

## First kernel source review — 2026-10-21

- Learner implemented the affine kernel body in `project_03_gpu_operator_library/kernels/cuda/affine.cu` after a supplied signature/TODO skeleton.
- Correct index, boundary guard, and output expression for the initial small-array contract. No compiler or GPU execution evidence yet; general large-index handling not covered.
- Ceiling division required correction; subsequent final-block reasoning was correct. Source implementation follows the jointly worked mapping, not an independent transfer test.
- Next active task: host launch configuration and syntax, followed by allocation and device validation. This supersedes the planning next-task entry below.
- [Session](../sessions/2026-10-21-project-03-affine-indexing.md); [lesson note](../../project_03_gpu_operator_library/notes/01-elementwise-kernels/01-thread-indexing.md).

## Status

- **Active planning**, requested 2026-10-20. Scope: [roadmap](../../roadmap.md#project-3-build-a-gpu-operator-library).
- User writes learning-target kernels; assistant teaches, reviews, and maintains authorized notes/progress.
- No kernel implementation, local execution environment verification, remote connection, or GPU measurement yet.
- Project 2 review remains paused at Q21; profiling continues within kernel experiments.

## Proposed development loop

1. **Mac:** specify shapes, dtype, layout, operation, reference, edge cases, and numerical criteria.
2. **Mac:** learner derives work assignment and indices, writes one small kernel, and checks indexing with a CPU model. CPU checks do not validate GPU races, compilation, or performance.
3. **GPU:** compile and run correctness cases, including non-divisible sizes; use relevant memory/race diagnostics before performance claims.
4. **GPU:** measure warmed-up execution with explicit synchronization and baseline boundaries; collect a small separate diagnostic profile.
5. **Mac:** inspect saved results, form a hypothesis, predict one change, implement it, and repeat.
6. **Transfer:** learner handles an unfamiliar shape or mapping change and explains correctness/performance without a complete supplied solution.

## Proposed sequence

- **First milestone:** FP32 contiguous `y = a*x + b`; CUDA C++ first, then Triton and TileLang versions of the same contract.
- **Next:** row reduction, then tiled GEMM, with practical coverage in all three languages.
- **Later roadmap coverage:** fusion, normalization/softmax, attention, representative spatial/irregular operators, one nontrivial backward, real-model integration, streams/events exercise. This is coverage planning, not a commitment to implement every operator in every language.
- First learning mechanism: mapping logical elements to parallel work and handling the final partial block.
- First independent check: derive and validate indexing for an unseen non-divisible size; no kernel solution supplied yet.

## Machine handoff proposal

- Keep source, small tests, environment specifications, notes, and compact evidence in Git. Commit/push when authorized; use the same recorded revision remotely.
- Keep large profiler reports separate; copy them back with an artifact manifest/checksums before losing access to the instance.
- Record GPU model, driver/toolkit/framework/compiler versions, command, shapes, dtype, correctness outcome, timing boundary, and code revision (plus any local diff).
- Separate CPU reference imports from GPU-only imports/builds so local reference checks can run independently.
- Check the existing GPU environment before choosing/installing versions. Use separate environments where language dependencies require them.
- Use short GPU validation loops for early kernels; batch planned experiments only once the execution path works.
- Timing and profiling remain separate runs; cross-language comparisons use equivalent contracts and boundaries.

## Current next task

- Agree the first operator contract, then ask the learner to derive work assignment and boundary handling.
- [Notes index](../../project_03_gpu_operator_library/notes/README.md).
- Latest session: [Mac/GPU workflow planning](../sessions/2026-10-20-project-03-workflow.md).
- Changes local and uncommitted; no remote synchronization.
