# Current status

## Softmax comparison checkpoint — 2026-11-08

- Saved the retained-online redundancy discussion in note 14: retention removes
  the input-pass advantage over Warp, while rescaling and repeated exponentials
  remain. A same-mapping conventional retained softmax is a proposed control,
  not an implemented or measured result.
- User authorized committing and pushing the accumulated roadmap, notes, plots
  and progress updates. Earlier dated entries retain their original sync status.
- Next: synthesize the five selected cases into the blog narrative, separating
  algorithm costs from mapping effects. GEMM resume remains preserved;
  independent understanding has not been reassessed.

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
- Active next step: explain the three PyTorch kernel families now represented, then compare algorithms and profiling evidence. GEMM indexing remains saved for later resumption. [Verification session](sessions/2026-11-02-project-03-large-softmax-results.md).

## Project 4 DSA added; later projects renumbered — 2026-10-29

- Added the approved beginner DSA curriculum in C++17 after Project 3. [Project 4](projects/04-dsa-for-ml-systems.md) is **planned / understanding not assessed**; lessons and code have not started.
- Former 4A/4B/4C → 5A/5B/5C; former 5–17 → 6–18. Runtime track is now 10–11. Older dated entries retain historical numbers; see the [migration record](sessions/2026-10-29-project-04-dsa-roadmap.md).
- Active learning stays **Project 3 GEMM**: derive grid, guard, and flat A/B/C indices for M37/K130/N65 and 16×16-thread blocks.
- Documentation only; no code execution, mastery assessment, commit, or push. Existing local work preserved.

## Naive GEMM lesson started — 2026-10-28

- User requested moving to GEMM. Active teaching target is one-output-per-thread CUDA GEMM, FP32 inputs/accumulation/output, row-major C=AB, M37/K130/N65, 16×16-thread blocks. Later agreed MMA/WGMMA precision unchanged.
- Next: learner derives grid/edge guard and flat A/B/C indices. No GEMM code supplied or execution performed. Added note 08 and GEMM section index.
- Softmax profile crossover analysis/backward/ports remain open; moving lessons does not mark them completed. Local documentation updates uncommitted.

## Mac handoff verified — 2026-10-28

- User requested pull and inspection of Project 3 results. `git pull --ff-only origin main` succeeded: 9882f73 → b664fcb. Existing separately downloaded results preserved; results are ignored by Git.
- Verified the local full-sweep package against all 227 manifest entries (186223642 bytes): zero missing/hash/size errors; all 36 Nsight reports present. This verifies this package, not every historical results folder.
- Inspected retained-variant correctness/sanitizer logs: 62 runs passed; memcheck/initcheck/synccheck report zero errors, racecheck zero hazards. Scope remains the recorded suite, not all generalized widths.
- Reviewed sweep and paired profiles: at 4096 rows, graph medians at width1024 are retained12.691µs/PyTorch9.305µs; width2048 retained25.466µs/PyTorch28.484µs. Causes remain to be analyzed; no new GPU runs.
- Next: offline comparison of the 4096×1024 and 4096×2048 instruction/mapping/traffic/stall evidence. This verification entry is local/uncommitted; no new push or rental shutdown.

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

## CUDA Graph timing note — 2026-10-27

- User requested saving the replay-timing explanation in Project 2. Added timing note21 with capture/replay distinction,500-call batch example,nine-round CUDA-event procedure, timing boundaries and per-call-p99 caveat. Linked both indexes and existing Project3 sweep harness.
- Documentation only; no new benchmark or mastery assessment. Project2 review remains paused atQ21; Project3 sweep interpretation remains active. Changes local/uncommitted.

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

## Divergence metrics revision note — 2026-10-27

- Added user-requested Project 2 note 20 on branch efficiency, active lanes, active predicated-on lanes and divergent branch-target counts, including Nsight navigation, raw metric names and interpretation pitfalls. Linked both note indexes.
- Documentation only, based on existing Project 3 profiles; no new GPU runs or independent mastery assessment. Project 2 review remains paused at Q21; active Project 3 next task unchanged. Changes local/uncommitted.

## Online softmax divergence checked — 2026-10-27

- Reused saved Nsight reports: online branch efficiency 99.01% versus warp 100%; active threads/instruction 31.38 versus 32. Divergence is present, but aggregate lane participation remains high.
- Source export attributes 8,192 divergent branch targets to two machine branches mapped to `merge_online`'s `if (b.s == 0)` guard. Not evidence of widespread data-dependent divergence in every update.
- Added source export and interpretation to `project_03_gpu_operator_library/results/2026-10-27-softmax-ncu/README.md`. No new kernel execution or code changes; local/uncommitted. Next: choose one controlled optimization, preserving correctness.

## Quick PyTorch softmax baseline — 2026-10-27

- User requested a quick framework timing before further load-stall investigation. Added `benchmarks/time_torch_softmax.py` under Project 3; eager FP32, matching inputs, 50 warmups, nine rounds × 500 calls and unchanged numerical criteria. All checks passed.
- H100/PyTorch 2.11.0+cu130: at 4096×1024, default output median 12.685 µs; preallocated output 10.175 µs. At 37×130: 5.337/4.663 µs. CUDA events include submission gaps; Python and C++ host paths differ, so this is not a pure kernel-latency ranking.
- Results: `project_03_gpu_operator_library/results/2026-10-27-pytorch-softmax/README.md`. No PyTorch kernel identity or bottleneck captured. Changes local/uncommitted.
- Next: return to controlled load-dependency investigation; framework speed alone does not explain the mechanism.

## Project 2 note follow-up — 2026-10-27

- Saved requested load-dependency stall-cycle explanation in Project 2 note 16, with an index link. Includes normalization example and warnings against interpreting it as individual-load latency or GPU idle time.
- Documentation only; no new runs. Active Project 3 profiling next step unchanged; Project 2 review remains paused at Q21. Changes local/uncommitted.

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

## Project 3 checkpoint prepared — 2026-10-26

- User requested commit/push of accumulated kernels, host validation, lesson notes, session history and expanded roadmap. Checkpoint includes affine, bias + ReLU, baseline/warp row sum, and baseline/warp/online softmax.
- CPU reference/validator and integer-fixture checks have passed as recorded below; current local documentation-link check passed (28 targets). CUDA compilation, GPU correctness/synchronization checks and timings remain pending. Source-reviewed implementations followed guided corrections.
- Suggested next offline lesson: retain softmax values in registers with the existing 128-thread mapping; not implemented. Then GPU comparisons, softmax backward and language ports before the GEMM progression. GPU session remains deferred at the user's request.
- This entry records the prepared checkpoint; successful commit/push must be verified from Git command results.

## Online softmax added to host suite — 2026-10-26

- User authorized harness extension. Softmax now schedules baseline/warp/online on the same ten cases (30 variant/case runs), with unchanged reference/tolerances, per-launch NaN reset, checked completion and per-variant diagnostics.
- Kernel arithmetic unchanged. Reviewed dispatch and ran `git diff --check`; no whitespace errors. CUDA compilation/GPU execution remain pending.
- Next: deferred GPU correctness session for row-sum and all three softmax variants before performance comparisons. Changes local/uncommitted.

## Online softmax cleanup verified — 2026-10-26

- Learner removed duplicate warp-summary loading and explicitly initializes unused final-warp lanes to the empty pair. Two required block barriers remain; no blocking source issue found in the reviewed implementation.
- Next: include online softmax in the existing common-reference host suite alongside baseline and warp variants. No CUDA compilation/execution or timings; guided implementation is source-reviewed only.

## Online softmax source review — 2026-10-26

- Learner added per-warp shared stores, a valid final shuffle loop, final pair publication and all-thread read. No remaining correctness issue found for the intended finite-input test workloads and launch contract; compilation/device validation remain pending.
- Redundant warp-0 partial loading and its extra barrier can be removed, but the retained load must explicitly assign empty pairs to padding lanes; merging with empty does not reset a nonempty state.
- Next: simplify the duplicate loading stage before adding online to the common host validation. No kernel edits during review; online is still not launched by the harness.

## Online kernel first review — 2026-10-26

- TODO 1 correctly merges singleton states over strided columns. TODO 2 shuffles both original fields before assignment with full-warp participation; lane 0 obtains the intended warp summary. Optional valid-source guard avoids self-merges in out-of-range lanes.
- Later stages were also filled but remain incorrect: no shared partial writes before reads, undefined `d` and partial-mask participation in final merge, no final pair publication/all-thread broadcast.
- Next: fix TODO 3 to publish each warp's lane-0 pair, then synchronize before reads. No code edits, compilation, or GPU execution during review.

## Online merge helper source-reviewed — 2026-10-26

- Learner corrected both exponent signs to old maximum minus merged maximum. Empty handling and nonempty pair merge now match the taught formula for valid finite-input summaries.
- Next: online kernel TODOs 1–2, private pair accumulation and first warp pair reduction. Source review only; no compilation/device validation.

## Online merge helper first review — 2026-10-26

- M1 correctly returns the other summary for an empty input, including both-empty. M2 chooses the correct maximum but reverses both rescaling exponents (new minus old instead of old minus new).
- Next: learner corrects exponent signs before implementing the kernel. Source review only; no code edits or CUDA execution.

## Online softmax skeleton added — 2026-10-25

- User delegated scaffolding. Added `OnlineState`, `merge_online` TODOs with an explicit NaN placeholder, and a six-stage `softmax_online` skeleton in `softmax.cu`.
- Next: learner implements helper empty handling/rescaling, then private state accumulation and first warp reduction. Both pair fields must be shuffled by all lanes before any guarded merge.
- Existing kernels and two-variant host validation unchanged. No CUDA compilation or execution; new skeleton is not launched.

## Online softmax recurrence covered — 2026-10-25

- Learner correctly derived rescaling for a larger maximum, unchanged-max update, parallel pair merge example, NaN from two empty maxima, and final output exp(x-m)/s. Empty-group guards and shuffle participation were explained by assistant.
- Expanded Project 3 note 06. Next: learner implements a small pair-merge helper before the online kernel. Guided conceptual understanding; no implementation or GPU validation yet.
- Documentation-only follow-up; existing GPU batch remains deferred and changes local/uncommitted.

## Online softmax lesson started — 2026-10-24

- User requested moving to online softmax. Checked original paper and existing notes/roadmap; introduced joint running maximum/shifted sum and three-to-two logical input traversals versus current kernels.
- Next: learner derives rescaling of exp(2-3)+exp(3-3) when a new value 5 raises the running maximum. Parallel pair merge and implementation follow; no online kernel created.
- Added note 06. Performance remains a hypothesis; deferred GPU validation of existing variants is still pending.

## Baseline/warp correctness harnesses ready — 2026-10-24

- User delegated host support. Row sum now tests both variants on ten widths (1/17/31/32/33/127/128/130/257/1025), 37 rows each. Softmax runs both variants on its unchanged ten cases and fixed tolerances. Each executable schedules 20 variant/case runs.
- Every variant gets a fresh NaN output sentinel, checked launch/completion/download, the same independent CPU reference, labeled diagnostics and aggregate failure status. Kernel bodies unchanged; no timing added.
- CPU softmax reference/validator tests passed; row-sum reference formula and FP32 exactness bounds verified for all 370 rows. `git diff --check` passed. No nvcc on PATH; CUDA builds and GPU execution remain untested.
- Next: deferred GPU session to compile and run both harnesses and validate synchronization/memory safety before benchmarking. Changes local/uncommitted; no remote action.

## Warp softmax source-reviewed — 2026-10-24

- Learner fixed `row_max` declaration and per-warp sum stores. Full source review found no remaining correctness issue under the declared finite-input/launch contract; both warp variants are now source-reviewed after guidance.
- Softmax uses four shared floats and five block barriers at N=128, versus baseline 128 floats and seventeen barriers. These are source counts, not measured performance.
- Next: extend the host correctness harnesses to exercise baseline and warp variants on identical inputs before the deferred GPU validation session. Hosts currently launch only baselines. No CUDA compilation/execution; changes local/uncommitted.

## Warp softmax first review — 2026-10-24

- TODOs 1–2 are correct: strided private max with negative-infinity identity and unconditional full-warp shuffle/max reduction. Learner also implemented later stages.
- Two blocking source issues remain: undeclared `row_max`, and all warp leaders store sum partials into slot 0 instead of their per-warp slots. Max-stage publication and shared-reuse barriers are correctly placed.
- Next: learner corrects declaration and sum-partial destination. Source review only; no kernel edits, CUDA compilation, or GPU execution.

## Warp softmax skeleton added — 2026-10-24

- User delegated scaffolding. Added `softmax_warp` beside the baseline with complete-warp constraints, shared partials, identifiers, and TODOs for max/sum reductions, all-thread broadcasts, safe shared reuse, and output.
- Learner implements arithmetic and synchronization. Next: TODOs 1–2, private maximum and first warp maximum reduction. Baseline host launch and numerical criteria unchanged; no CUDA compilation/device execution. Changes local/uncommitted.

## Warp row sum source-reviewed — 2026-10-24

- Learner corrected the second-stage inputs: warp 0 loads shared warp totals with zero padding before its full-warp shuffle reduction. No remaining correctness issue found under the documented launch/input contract.
- Initial shared zero stores and skeleton/TODO comments are redundant cleanup. Host still launches the baseline; no CUDA compilation, device validation, or timing.
- Next learning task: adapt this two-level reduction to softmax with all-thread result publication and safe shared-buffer reuse. GPU comparisons remain deferred; changes local/uncommitted.

## Warp row-sum follow-up review — 2026-10-24

- Learner fixed full-warp shuffle participation, lane-0-only partial stores, block synchronization, and the single output writer.
- Remaining correctness issue: warp 0 still reduces its old register values instead of loading shared warp totals with zero padding. Next: initialize the second reduction's per-lane input inside `warp == 0` before its loop. Initial shared-zero stores are unnecessary.
- Source review only; no code edits or CUDA execution. Host still launches baseline.

## Warp row-sum first review — 2026-10-24

- TODO 1 correctly accumulates strided columns. TODO 2 incorrectly guards full-mask shuffles with `lane < d`; all named lanes must participate. Next: remove that guard around the first shuffle loop and review.
- Learner also filled later stages: shared partials currently have multiple writers, final merge never loads shared partials, and every warp writes the row output. These remain pending corrections; kernel unchanged during review.
- Source inspection only; no CUDA compilation or execution. Host still launches baseline.

## Warp row-sum skeleton added — 2026-10-24

- User delegated scaffolding. Added `row_sum_warp` beside the baseline with template constraints, shared partial storage, identifiers, and staged TODOs. Reduction body remains learner work; host still launches baseline.
- Next: learner implements private accumulation and first warp reduction (TODOs 1–2). No CUDA compilation or execution; changes local/uncommitted.

## Warp-reduction synchronization reasoning — 2026-10-24

- Learner correctly selected sum/max identities, recognized all threads need the softmax maximum, and explained the read-before-overwrite race when shared storage is reused. Block-wide versus warp-only synchronization was clarified with guidance.
- Added Project 3 note 05. No kernel edits, CUDA compilation, GPU execution, or performance measurement.
- Next: learner adds a separate warp row-sum kernel alongside the shared-tree baseline, starting with private accumulation and the first warp reduction. Changes remain local/uncommitted.

## Warp-reduction optimization started — 2026-10-23

- User requested warp reductions for row sum and softmax. Keep current kernels as baselines; first develop warp-shuffle row sum, then softmax max/sum with block-wide result broadcasts.
- Plan for 128-thread blocks: private accumulation → four warp reductions → four shared partials → warp 0 merges. All 32 lanes participate in each full-mask shuffle; no-data lanes contribute identities.
- Hypothesis: less shared traffic and fewer block barriers, to test later with matched correctness/timing boundaries. No speedup measured and no warp kernel implemented yet.
- Next: learner reasons about neutral lanes in the final warp reduction before implementation. GPU execution remains deferred.

## Softmax correctness harness added — 2026-10-23

- User delegated host scaffolding/reference/test cases. Added a 10-case CUDA harness around the learner kernel, CPU-only reusable test support, and a CPU reference/validator test executable.
- Fixed acceptance: elementwise abs error <= 1e-7 + 1e-5*abs(reference); row-sum error <= 1e-5; finite outputs in [0,1]. Includes widths 1/17/127/128/129/130/257/1025/4097 and negative/equal/large/tied/peaked/mixed logits.
- Compiled and ran the CPU test with clang++ C++17 and warnings-as-errors: PASS. No CUDA compilation/device execution; no timings. Next: separate GPU correctness batch when user is ready, then measured softmax optimization. Changes local/uncommitted.

## Softmax kernel source-reviewed — 2026-10-23

- Learner completed stable FP32 softmax correctly under the declared launch/input contract: max reduction, safe shared-array reuse, shifted exponential sum, sum reduction, and normalized outputs.
- Final barrier after privately reading the denominator is redundant because the shared array is not written again; harmless, not a correctness issue. No kernel edits made during review.
- Next: host numerical-validation harness with stable FP64 reference, explicit tolerances, and adversarial inputs. No compilation/GPU run; local changes remain uncommitted.

## Softmax skeleton ready — 2026-10-23

- Learner correctly selected negative infinity/max combine and placed the barrier after privately reading the shared maximum, before shared-array reuse.
- Created `project_03_gpu_operator_library/kernels/cuda/softmax.cu` with signature/shared storage/TODOs only, at user request. Next: learner implements the stable FP32 baseline, recomputing exponentials for output.
- No softmax implementation, compilation, or GPU execution yet. Notes updated; changes local/uncommitted.

## Stable softmax lesson — 2026-10-23

- Started FP32 [37,130] row-wise softmax, one 128-thread block per row, finite inputs/no mask. Learner correctly identified exponential overflow and explained shift invariance through cancellation of the common factor.
- Added note 04. Next: derive the maximum reduction's initial value and tree combine operation, then implement the stable kernel. No softmax code or GPU execution yet; batch validation remains deferred.

## MMA/WGMMA precision agreed — 2026-10-22

- User confirmed primary variants: FP16 inputs/accumulation/output; BF16 inputs with FP32 accumulation and BF16 output. Canonical roadmap updated; this resolves the earlier BF16 accumulation question.
- Applies to matrix accumulation, not an implicit change to softmax/reduction precision. No code or GPU run; next lesson remains stable softmax.

## Low-precision accumulation request — 2026-10-22

- User requested FP16/BF16 accumulation as well as outputs. Added native FP16 accumulation to the MMA/WGMMA plan, superseding the FP32-only accumulation preference below.
- Checked NVIDIA PTX instruction forms: FP16 accumulation supported with FP16 multiplicands; BF16 warp MMA and Hopper WGMMA require FP32 accumulators. Native all-BF16 requested path cannot be supplied on these instructions; no silent emulation/substitution planned.
- Roadmap records limitation and optional separately labeled rounding experiments. Documentation only; no GPU or code changes.

## MMA/WGMMA output precision — 2026-10-22

- User requested FP16/BF16 outputs. Initial variants now explicitly pair FP16 inputs/output or BF16 inputs/output, retaining FP32 accumulation and converting at the output epilogue.
- Roadmap updated; output dtype must match benchmark baselines and numerical criteria. Planning only; no kernel changes or GPU run. Stable softmax remains the next lesson.

## Inline PTX direction — 2026-10-22

- User requested SageAttention-style `asm volatile` wrappers for the CUDA GEMM/MMA work. Roadmap now explicitly includes learner-written device wrappers for existing MMA/WGMMA and supporting instructions.
- Plan: validate a small instruction-level tile and fragment layout before building tiled GEMM/attention; preserve forward/backward correctness and generated-code inspection requirements. No new kernel implementation or GPU execution.

## Project 3 performance scope revised — 2026-10-22

- User requested efficient forward/backward operators, online/optimized softmax, naive → shared → MMA → WGMMA GEMM, and direct MMA → WGMMA custom attention.
- Updated Project 3 in `roadmap.md`; proposed interleaved CUDA C++, Triton (Gluon for explicit Hopper control), and TileLang implementations with common contracts and measured performance targets.
- Hopper/H100 is the planned WGMMA target, not a newly verified environment. No new GPU activity; current three kernels remain source-reviewed only.
- Next offline learning task: stable fused softmax and its numerical contract; existing-kernel GPU batch remains deferred at user request. Advanced optimization requires subsequent GPU validation/measurement. Changes local/uncommitted.

## Row-sum host scaffold added — 2026-10-21

- User delegated host scaffolding; assistant added a [37,130] fixture, checked allocations/transfers/launch/completion, NaN output sentinel, sequential CPU reference, comparison, cleanup, and exit status. Learner reduction logic unchanged.
- CPU-only FP32 arithmetic model matches all 37 reference sums; every row total differs, and omitting columns 128/129 fails every row. No CUDA compilation or execution.
- Affine, bias + ReLU, and row sum now have first correctness harnesses. Next planned milestone is the separate GPU compilation/validation batch when the user is ready. No remote session started; changes local/uncommitted.

## Row-sum template fixed — 2026-10-21

- On explicit request, assistant changed `template <block_size N>` to `template <int N>` in `row_sum.cu`; reduction logic unchanged.
- Next: add host correctness scaffolding. Intended launch remains `row_sum<128>` with 128 threads/block; no compilation or GPU execution yet. Changes local/uncommitted.

## Row-sum first implementation reviewed — 2026-10-21

- Learner implemented private accumulation, shared staging, halving reduction, unconditional barriers, and thread-0 output correctly under a matching power-of-two block-size contract.
- Learner introduced a template with invalid/undefined parameter type `block_size`; correction needed to `template <int N>`. Actual threads/block must equal N; first intended instantiation remains 128.
- Next: learner fixes template declaration; then review and add host scaffolding. Static review only; no compilation/GPU run.

## Row reduction lesson — 2026-10-21

- Learner correctly assigned 130 columns across 128 threads, completed the 8-slot tree example, and explained why a barrier inside the participating-thread condition is invalid for this algorithm.
- Added `row_sum.cu` signature/shared-array/TODO skeleton; reduction implementation remains learner work. First launch contract: 37 blocks, 128 threads each, one row per block.
- Next: learner implements private accumulation, shared-memory reduction with unconditional block barriers, and thread-0 output. GPU batch remains deferred; no compile/run or independent implementation evidence yet.

## Bias + ReLU harness delegated — 2026-10-21

- User requested scaffolding; assistant added the host correctness harness around the learner's kernel: signed test data, allocation/transfers, output NaN sentinel, checked launch/execution, CPU reference, comparison, cleanup, and exit status.
- CPU fixture arithmetic checks passed for 4810 elements, both sign-crossing directions, and zero sums; incorrect ReLU-before-bias order differs at 2316 elements. This did not compile or execute CUDA.
- Next: row-sum reduction concepts and work assignment. GPU compilation/validation of both kernels remains deferred to the planned batch session; local uncommitted changes only.

## Project 3 bias + ReLU source reviewed — 2026-10-21

- Learner correctly derived flat bounds and modulo bias indexing. Initial kernel applied bias after ReLU and used an undeclared variable; corrected after explanation.
- Saved `bias_relu.cu` now computes input plus column bias before ReLU, with guarded input/output accesses. Static review only; no compilation or execution.
- Next: learner adds a host correctness harness for [37,130], with varied signed inputs and distinct column biases. GPU validation remains deferred to the requested batch session.

## Project 3 offline kernel batch — 2026-10-21

- User deferred GPU execution and requested writing the next small kernels first, followed by a separate GPU launch session.
- Proposed bounded batch: existing affine, then 2D bias + ReLU, then a row-sum reduction. Continue TUTOR mode; learner implements kernels. Every kernel remains unvalidated until GPU compilation/testing.
- Next: define contiguous [37,130] FP32 bias + ReLU and have learner derive flattened indexing and bias-column selection. Reduction cooperation/synchronization will be taught before implementation.

## Project 3 affine source review complete — 2026-10-21

- Final saved source includes required headers, guarded indexing, initialized input, checked CUDA allocations/transfers/frees, launch and execution checks, output comparison, cleanup, and explicit pass/fail exit status.
- No blocking source-level issue found for the current N=1000 FP32 case. Review only: no compilation, GPU execution, sanitizer check, or timing performed.
- Next: verify the GPU environment, compile, and run the first correctness test. Source and documentation remain local and uncommitted; no remote synchronization.

## Project 3 host harness review — 2026-10-21

- Learner added allocation, transfers, launch, and comparison. Through review, corrected declaration order, input initialization, premature host-array deletion, and unnecessary output upload.
- Current case: N=1000, 256 threads/block, four blocks; comparison precedes host cleanup. Still needs explicit success/failure exit status and CUDA API/launch/execution error checks.
- Next: add result status and error checks before first GPU validation. Static review only; no compile or run. Source/notes remain local and uncommitted.

## Project 3 first kernel reviewed — 2026-10-21

- Learner filled in `project_03_gpu_operator_library/kernels/cuda/affine.cu`: correct 1D index, `i < N` guard, and affine output. Source review only; no compilation or execution.
- Index/guard supplied correctly; block-count expression needed correction from ceiling after integer division. Last-block calculation then answered correctly: 104 valid threads, 24 skip accesses.
- Next: introduce host launch syntax and have learner derive grid size for the launcher. Allocation, GPU validation, numerical criteria, timing, and profiling remain pending. Changes local, uncommitted.

## Active focus: Project 3 — 2026-10-20

- User requested moving to kernel writing, integrating profiling into each operator. Project 2 review remains paused at Q21; its numerical-acceptance and independent-transfer work remains open.
- [Project 3 plan](projects/03-gpu-operator-library.md): Mac for design/reference checks/source review; Linux NVIDIA GPU for compilation, device validation, timing, and profiling. Detailed sequence is proposed, not yet exercised.
- Next: define the first FP32 contiguous elementwise operator contract and have the learner derive its work assignment and boundary handling before implementation.
- Documentation only; no kernel implemented, environment installed, GPU contacted, commit, or push. Earlier runtime-track edits remain local.

## Runtime track planned — 2026-10-20

- User approved a connected inference-runtime track across Projects 9–10 in `roadmap.md`: executor, execution backends, request scheduler, decoder/KV runtime, and vLLM source trace/comparison.
- First version: one GPU, one small model at a time, in-process requests. TensorRT-LLM comparison remains an extension. Planning only; no implementation, experiment, or mastery claim.
- Plan record: [runtime track](projects/10-11-inference-runtime.md). Project 2 review remains paused after Q20; next learning task is unanswered Q21.
- These roadmap/progress edits are local and uncommitted; no push in this update.

## Notes organized — 2026-10-20

- Moved 19 Project 2 notes into five topic folders: timing/setup, timeline profiling, GPU hardware/scheduling, performance comparisons, and review. Root/section indexes updated; lesson content/numbers preserved.
- Updated links and active repository references; checked 91 local note-link targets. Ignored bundled study snapshots retain their independent paths. Added future folder/index/link-maintenance rules to AGENTS.md and learning instructions.
- Learning review remains paused after Q20; resume Q21 from `project_02_performance_laboratory/notes/05-review/19-profiling-review-questions-and-answers.md`. Changes local, uncommitted; no push or new experiment.

## PAUSED — profiling review saved 2026-10-20

- User requested stopping and saving. **20/24 questions reviewed; Q21 asked but unanswered. Resume at Q21**, the off/on coverage comparison; do not restart the quiz.
- Separate revision file: `project_02_performance_laboratory/notes/05-review/19-profiling-review-questions-and-answers.md` includes Q1–Q20 corrected answers, follow-ups, topic links, and exact Q21 prompt. Notes index updated.
- Tracker reconciled through Q20. Assisted topics include Q15 latency/dependent loads, Q19 bank conflicts/broadcast, and Q20 service versus total latency; saved answers do not establish mastery. Q22–Q24 remain unasked; throttle classification still untested.
- No new GPU runs. Changes remain local and uncommitted; no push or remote synchronization performed.

## Service-latency example — 2026-10-20

- Added requested 7-versus-8-cycle worked example to note 17; general service-work/latency caveat already existed. Q20 throughput arithmetic correct; total-latency interpretation assisted.
- Next: answer pending Q21 off/on GPU coverage comparison. No new measurements or remote sync.

## Dependent-load explanation — 2026-10-20

- Q15 introduced a prerequisite not previously taught explicitly: load-result-dependent addresses. Explained embedding lookup versus independent loads and how few requests in flight can coexist with low bandwidth and long waits. Added requested example to note 16.
- Learner reports understanding; dependent-load transfer remains unassessed. Next: one changed Q15 example before Q16. Documentation local; no new experiment.

## Q15 clarification — 2026-10-20

- Q15 diagnosis required guidance: low DRAM bandwidth use does not imply little memory latency. Learner suggested memory/warp statistics; per-tier throughput inspection explained. Follow-up on dependent loads remains pending.
- Added requested resident-state/admitted-to-SM explanation and changing eligibility to note 16, plus L1TEX operation meaning and shared-memory classification caveat. No new measurement.

## Instruction-fetch and constant stalls — 2026-10-20

- Before Q15, explained No Instructions versus IMC Miss. Existing notes covered IMC reissue and briefly instruction-fetch waiting; added explicit comparison to note 16 and cross-link from note 15.
- Q15 remains unanswered: low DRAM throughput plus Long Scoreboard diagnosis. No new profiling or mastery claim.

## Stall-state review — 2026-10-20

- Review tracker updated through Q14 using recorded answers; Q10/Q11 corrections and assisted explanations preserved. Q14 five state matches all correct; throttle distinction remains untested.
- Not Selected already appeared briefly in note 16; added dedicated explanation and Details/Source/Raw lookup instructions with sampling and missing-data caveats. Next Q15 low-DRAM/Long-Scoreboard diagnosis. No new profiling or live UI inspection.

## Review round 2 complete — 2026-10-19

- 8/24 reviewed. Q7 matching correct, maximum aggregation refined. Q8 active/elapsed percentages correct; different clock domains explained with toy frequencies, then learner correctly selected DRAM elapsed cycles.
- Added requested clock-domain example and matching-denominator guidance to note 17. Next: Q9 resource-limited theoretical residency. No GPU measurement or synchronization.

## Review pipeline clarification — 2026-10-19

- 6/24 reviewed: Q5 normalization correct; Q6 maximum aggregation correct with interpretation refinements. Q7 pending after explaining memory-instruction pipelines.
- Added requested plain-language Mem Pipes Busy definition and scheduler/pipeline distinction to note 17. Next: match Q7 instruction, internal service, and transfer limits to the three memory summaries. Documentation only; no new GPU run.

## Review round 1 complete — 2026-10-19

- 4/24 reviewed: Q2 cache denominators and Q4 counter lookup answered independently with notes allowed; Q1 reuse and Q3 measurement selection assisted. Q3 follow-up correctly identified shorter time; concurrent loads remain a possible cause, not established evidence.
- Q4 correctly named integer thread instructions and DRAM read bytes; clarified total DRAM traffic also includes writes. Missing counter correctly identified as potentially uncollected.
- Next: Q5 normalization using events, elapsed cycles, and peak events/cycle. No new experiment or remote sync.

## Instruction-counter notes — 2026-10-19

- Q3 partly supported: learner correctly distinguished total instructions from floating-point work; instruction category/opcode lookup explained after proposing throughput metrics. Added integer counter, thread/warp units, and address-purpose distinction to note 18.
- Next: pending Q3 follow-up, unchanged bytes with higher bytes/second implies what timing change? No new GPU measurements; documentation local.

## Cache review — 2026-10-19

- Review: 2/24 questions covered. Q1 assisted: cache reuse and requests versus hits explained; measurement selection to revisit in Q4. Q2 answered correctly without further hints: 400 L2 requests, 300 hits, 100 sectors from DRAM.
- Next: Q3, distinguish total instructions, DRAM bytes, and useful FLOPs. Include short-note links in feedback. No GPU runs or remote synchronization.

## Comprehensive review — 2026-10-18

- User approved the 24-question plan. [Tracker](projects/02-profiling-review.md): 1/24 attempted; Q1 partly supported, changed-example follow-up pending. One question at a time, notes allowed, feedback every answer and progress recap every four.
- Follow-ups address named gaps; no new GPU runs or mastery claim. Learner identified variable copy size; cache reuse and traffic-versus-issued-count distinction explained. Next: Q1 changed-example check. Include short-note references in future feedback.


## Nsight metric lookup — 2026-10-18

- Mapped issue activity, total instructions, opcode/category breakdown, and memory traffic to Details/Raw locations. Rechecked on035 saved counters; added reference table to note 18. No new GPU run or live UI inspection.
- Open-note review: learner explained more concurrency/prefetching and register-residency tradeoff; refined activity-versus-runtime interpretation. Next apply the lookup to compare useful progress with added instruction/traffic work.

## Dependency-stall revision — 2026-10-17

- Added requested note 16 clarification: dependencies do not always cause stalls; independent work can allow a result to arrive before use. Distinguished execution latency, producer category, and scheduler idleness. Learner reports understanding; transfer not yet assessed.
- Next: answer the pending hypothetical 3%-DRAM / Long-Scoreboard / no-eligible-warps question. No new GPU run or remote synchronization.

## Open-note review — 2026-10-16

- Further scoreboard clarification: separated operation execution, dependency bookkeeping, and profiler stall measurement. Explained physical L1TEX/shared grouping versus scoreboard classification, MIO meaning, and producer-dependent waits despite register destinations. No new mastery claim; low-DRAM exercise still pending.

- Further scoreboard clarification requested: separated readiness tracking from stall measurement; explained MIO, shared SRAM versus operation paths, and register destinations versus producer dependencies. Updated note 16. Low-DRAM exercise remains pending.

- Explained scoreboard dependency tracking with a global-load/accumulator consumer example. Long/Short classify operation paths, not a measured-cycle threshold; Long Scoreboard does not establish a DRAM miss or saturation. Added note 16. Pending low-DRAM/Long-Scoreboard exercise remains unanswered.

- Learner correctly calculated eligibility averages and issue counts after clarification of eligible-warp cycles versus issued instructions; final case = one average eligible warp, 0.5 instructions/cycle. Units corrected. Added requested formulas and three worked cases to note 16. Next pending question: low DRAM utilization with Long Scoreboard/no-eligible cycles, distinguishing latency from bandwidth.

- On request, completed note 16 with the eligibility checklist, selection distinction, and shared-tile barrier example; added an index link. Pending exercise remains equal average eligibility with different timing distributions.

- Learner correctly calculated 32 active/100 idle SMs, one resident warp/scheduler, 6.25% active-cycle occupancy, and 1.21% elapsed resident-warp capacity use; definition/percent notation refined with assistance. Correctly distinguished residency from readiness.
- Paused eligible-average exercise to explain instruction fetch, barriers, pipeline/queue availability, control flow, and sleep as eligibility blockers. Added Q&A to note 16. Next resume the 100-cycle eligibility-distribution exercise; answers not yet supplied. No new GPU experiment.

## Variant instruction check — 2026-10-16

- Added requested instruction-counter lookup to note 18 and notes index: Details → Instruction Statistics, Raw metric names, warp-instruction units, and saved CSV/log verification method. Next: final learner synthesis and independent transfer; numerical acceptance remains unresolved.

- On explicit request, expanded note 18 with percentage changes and concise explanations of instruction counts versus DRAM bytes and useful arithmetic work. Accumulation increase remains unverified.

- Learner predicted increased memory/accumulation instructions. Re-read on035/on031 counters: total instructions 18,097,152→21,179,392; shared loads 902,656→1,198,080; LDGSTS 378,880→606,208. DRAM bytes slightly decrease. Memory-instruction increase confirmed; accumulation count unverified (no direct FFMA count in inspected exports).
- Updated note 18; no GPU run. Next distinguish more instruction work from useful convolution work and avoid claiming increased accumulation without opcode evidence.

Last updated: 2026-10-15 (memory notes consolidated; no new GPU experiment).

## Off/on synthesis — 2026-10-15

- Began requested 512-block transfer comparison: re-read on031/032 and on035/036/037 raw CSVs. 512-block kernel uses 64 threads/block, 92 registers/thread, ~12.04% achieved occupancy, ~1.95 active/~1.20 eligible warps per scheduler, ~75% active-cycle issue. Repeat profile durations overlap at 29.920 µs. Added comparison to note 18; learner interpretation of utilization versus runtime pending.

- Supplied requested full explanation: smaller tiles/more blocks improve potential GPU coverage despite similar achieved occupancy and lower active-cycle issue rate. Added note 18 with separate unprofiled 2.45×/59.18% result, identity variability, and unresolved numerical acceptance. Assisted explanation; next transfer to saved alternate 512-block on capture.

- Added explicitly requested memory-metrics Q&A to note 17 and linked it from the index; existing formulas/examples retained. Off/on coverage explanation remains the pending learner task.

- Began paired comparison using expanded off030/on035 only. Re-read raw CSVs: 32/128 blocks, both 128 threads/block; duration 73.984/30.144 µs; achieved occupancy both about 6.25%; active-cycle issue 72.40/65.11%; elapsed SM throughput 17.06/57.58%.
- Next learner explains how wider device coverage can coexist with similar occupancy and lower issue rate during active cycles. Profiler timing is diagnostic; numerical tolerance failure remains unresolved. No GPU run.

## Notes consolidation — 2026-10-15

- User requested all covered material in short notes. Consolidated note 17 with a quick revision summary, cache-hit denominators/reuse, SRAM management, hardware names, metric formulas, wavefront examples, and bounded memory/scheduler conclusions; linked note 16 and updated the index.
- Learner correctly rejected sustained DRAM saturation, but reasoning was refined with assistance; alternative dependency/latency explanation was supplied. No independent transfer or project completion claimed.
- Next: combine the saved off/on kernel coverage and throughput findings to explain the performance difference, preserving measurement and numerical-acceptance limitations. Documentation remains local; no new GPU run.

## Memory metrics explanation — 2026-10-15

- Worked scalar shared-load example added: `s[lane]` → one wavefront; `s[2*lane]` → two due to bank conflicts. One instruction/warp remains intact; service steps are not full load latency. No GPU run or independent transfer assessment.

- Follow-up: service throughput versus completion latency, stage-specific wavefronts, and clock-specific peak rates. Saved H100 normalization implies 1,280 aggregate DRAM bytes/cycle; no B200 measurements. Next clarify this model before the pending diagnosis.

- User requested first-principles measurement of Mem Busy, Max Bandwidth, and Mem Pipes Busy. Rechecked raw on035 summary/component matches (41.909052%, 30.280275%, 17.281627%); explained distinct instruction, service, and transfer capacities plus max-of-normalized-constituents aggregation.
- Notes updated; no new GPU run or independent conclusion. Next: learner gives the bounded memory-bandwidth diagnosis. See [session](sessions/2026-10-15-project-02-memory-metrics.md).

## Notes update — 2026-10-14

- Condensed L2/LTS, TEX/L1TEX, and H100 cache-sharing notes into a quick-reference table and weight-reuse example; linked from the notes index.
- Next: learner gives a bounded memory-bandwidth conclusion from the existing on035 measurements. No new assessment or GPU run. See [session](sessions/2026-10-14-project-02-cache-notes.md).

## Current lesson — 2026-10-09

- Hardware-name clarification: LTC/LTS and TEX/L1TEX; verified saved device attributes (132 SMs, 50 MiB L2). Explained device-wide L2 sharing versus physical partitions. Memory conclusion remains pending; no new GPU run.

- Proceeding to Mem Busy 41.91%, Max Bandwidth 30.28%, Mem Pipes Busy 17.28% from the supplied screenshot; checked NVIDIA definitions. Next learner gives a bounded memory-bandwidth conclusion using these summaries and 2.28% DRAM read utilization.

- Memory follow-up: explained sector hit-rate denominators and moved to shared SRAM management. Distinguished per-block shared allocation from per-SM carveout; used illustrative dense-layer tiles and convolution reuse. No kernel change or new measurement. Next return to memory-analysis interpretation after clarifying the allocation/cache distinction.

- Moved to memory analysis after learner selected Wait for investigation (reason refined with assistance). Verified on035 DRAM reads: 2.299392 MB, 76.280255 GB/s, 2.279788% of peak sustained elapsed; duration 30.144 µs. Explained byte counting and elapsed-cycle normalization in note 17. Next distinguish cache hit-rate denominators. No new GPU run or independent mastery claim.

- Clarified operand collection as reading stored input values for execution. NVIDIA's public Dispatch Stall definition does not specify registers only; register-bank contention is a documented example. Understanding remains under discussion; no new experiment.

- User accepted bounded scope and requested the next step. Current exercise: interpret on035 all/Not-Issued sample counts (Wait 382/376, Dispatch Stall 223/5). Values rechecked; learner interpretation pending before moving to memory behavior.
- UI follow-up: learner could not locate counters and requested Wait/Dispatch refresher. Clarified default chart versus CSV-derived PC-sampling counts; provided expanded-report Raw filters and Profile > Metric Details navigation. Interpretation still pending.
- Clarified "base" as informal all-samples shorthand and `_not_issued` as scheduler-wide no-issue filtering. Exact internal Dispatch Stall conflict remains unknown; no specific queue/port cause inferred from the label.
- Concrete dispatch follow-up: found NVIDIA engineer explanations of register-file bandwidth and bank contention delaying operand collection. Added example distinguishing a value not produced yet from a produced value whose register access is delayed. Mechanism documented elsewhere; H100 attribution remains unknown.
- Learner questioned why register/shared-memory bank conflicts have different labels. Explained physical cause versus observed instruction stage: register operand collection before dispatch versus an already-dispatched shared load delaying its consumer (Short Scoreboard). Understanding still under discussion; defer further counter interpretation until distinction is clear.

- Scope review requested before continuing: current convolution study has five proposed closing steps (scheduler/stalls, memory, combined kernel comparison, bounded conclusion/numerical limitation, fresh report transfer). Full Project 2 still includes broader harness/timing coverage, memory accounting, controlled bottlenecks, precision/quality experiments, encoder/decoder profiling, and unfamiliar-workload exit. This organizes existing scope; no roadmap change or completion percentage claimed.

- Learner previously answered yes to another warp filling a dependency wait, then requested more detail. Explained overlapping independent warp instructions with a four-cycle toy dependency and a pipelined execution unit.
- Added note 16; no kernel/source change or GPU run. Understanding remains assisted; two-warp timeline transfer pending.
- Next: learner predicts the two-warp toy schedule, then compare all versus Not-Issued stall samples. See [session](sessions/2026-10-09-project-02-latency-hiding.md).

## Current clarification — 2026-10-04

- Learner asked why warp cycles per issued/executed instruction both show 1.53. Verified on035 raw ratios **1.531242 / 1.532910** and issued/executed counts **18,116,865 / 18,097,152**; display equality is rounding.
- Explained warp-cycle accumulation and different denominators, separate from individual instruction completion latency. Possible reissue mechanism documented, exact cause of observed difference unknown.
- Next: return to warp waiting versus scheduler idleness. See [session](sessions/2026-10-04-project-02-issued-executed.md).

## Current task — 2026-10-03

- Clarified active versus elapsed cycles: resident-but-waiting warps can contribute active cycles. No independent normalization transfer checked.
- Proceed to expanded on035 **Warp State Statistics**. Learner identifies the two largest non-Selected reasons, preserving values and units, before causal interpretation.
- Saved CSV contains both normalized state ratios and PC-sampling counts; keep those measurements distinct. See [session](sessions/2026-10-03-project-02-warp-state-reading.md).
- Learner correctly read **Stall Wait** and **Stall Dispatch Stall**, but could not find units. Verified normalized values 0.179721/0.101297; explaining fixed-latency dependency versus dispatch blockage. Next interpret a dependent-accumulator example, then compare no-issue-cycle evidence. Report-reading success is distinct from independent causal diagnosis.

## Resume — 2026-10-02

- User resumed **Project 2 compute throughput** in TUTOR mode on the Mac.
- Local results verifier passed **229 files**, confirming manifest-listed payload sizes/hashes. Saved reports are available for offline study.
- Resume with the original off007/on008 occupancy/throughput comparison; ask the learner's current interpretation before proceeding to resident/eligible/issued warps in expanded off030/on035.
- No new assessment or performance result. See [local resumption](sessions/2026-10-02-project-02-throughput-resume.md).
- Follow-up: learner requested a direct throughput explanation. Rechecked original CSV pipeline values and NVIDIA definitions; added counters versus controllers and count/rate/activity toy example to note 13. Independent interpretation remains unassessed.
- Further questions: explained possible reasons for unused instruction-issue capacity and why Nsight summarizes normalized constituents with a maximum. No measured attribution of the missing capacity; next examine scheduler readiness.
- Requested in-depth DL example: note 14 follows a dense layer through scalar FMA dependencies, a hypothetical six-cycle issue schedule, grid coverage, and maximum aggregation. Connected to the recorded convolution shapes. No GPU experiment or independent assessment.
- Latest: learner reports understanding throughput normalization (self-report, no transfer test). Read saved expanded Scheduler Statistics; resume with on035 active/eligible/issued values about 1.00/0.65/0.65 per scheduler and ask what to investigate. These use active-cycle normalization, distinct from elapsed-cycle SM throughput.
- Learner selected readiness as the issue and refined wording to "not ready on time"; supported immediate interpretation after explanation. Requested graph calculations before proceeding. Note 13 now derives hardware maximum 16 and resource ceiling 3, and explains measured active/eligible/issue averages with a toy counter table. Next: learner applies averaging to another short sequence, then inspect stall causes.

## Roadmap revision — 2026-09-30

- User accepted the coverage review with practical, comparable first-pass depth
  in **CUDA C++, Triton, and TileLang**; defer extreme specialization until useful.
- `roadmap.md` now includes **Project 4C: Quantization Laboratory**, with a broad
  algorithm survey, bounded core experiments, optional methods, and deferred
  KV/fine-tuning applications. Existing project IDs are preserved.
- Added explicit numerical accuracy, training-memory, GPU concurrency, attention,
  topology, and serving-cost milestones; marked advanced branches optional.
- Project 2 remains active. **Next learning task:** explain accumulation error
  and distinguish numerical tolerances from application-level acceptance in the
  recorded cuDNN/float64 comparison. No result or mastery status changed.
- See the [revision session](sessions/2026-09-30-roadmap-revision.md).

## Earlier review — 2026-09-29

Roadmap review (2026-09-29): reviewed planned coverage and existing progress at
the user's request. Recommended a dedicated quantization sequence, earlier
numerical-accuracy work, explicit training-memory and GPU-concurrency exercises,
and optional specialization branches to reduce mandatory scope. These are
recommendations, not an adopted roadmap revision. See the
[review session](sessions/2026-09-29-roadmap-review.md). Project 2 remains active;
its latest numerical-acceptance question remains open. No experiments rerun.

The experiment history below was last updated on 2026-09-27; later follow-ups
within this file supersede its earlier resume instructions.

## Active focus

**Project 2: ML Performance Laboratory.** The user explicitly switched to it after
the scalar-autograd/MLP work. The learner's CNN baseline now has 10 warmup steps
and five timed blocks of 50 CUDA training steps. The corrected repeated-block
benchmark completed. Learner correctly identified that observed block variation
can resemble a 1% improvement; causal interpretation was clarified with assistance.
Scope explicitly includes whole-model and individual GPU-kernel profiling, with
conceptual learning and experiments at both levels.

Support setup: local Git and durable progress tracking for multiple machines.
The earlier session deferred GitHub setup. A GitHub `origin` is now configured;
remote synchronization has not been checked in this session.

## Resume next

**Latest verified baseline (2026-09-27):** Learner corrected the final timing
boundary; conv_dgrad.py ran successfully in kvforge on H100. Five unprofiled
100-call blocks: 147.50, 144.53, 144.49, 144.49, 144.49 us/completed helper call
(rounded printed values). Correctness passes, max error 7.6293945e-06. Separate
capture reports dgrad mean 142.068 us and scale helper mean 1.354 us, three each.
Learner initially suggested low CPU overhead/caching; after overlap explanation,
correctly predicted per-call synchronization increases time by preventing next-call
submission during previous GPU work. Assisted understanding; comparison experiment
explicitly skipped at learner request. Next: estimate whole-step benefit from a
hypothetical 2x improvement to a kernel group (Amdahl's law); no optimization yet.
Timing implementation was assisted; earlier pending entries below are history.

**Current task: unprofiled isolated-call baseline.** Corrected conv_dgrad.py runs
and exports three labels/six kernels: one scale helper + one dgrad per call.
Exact dgrad name and grid/block/register/shared-memory configuration match model.
Profiled dgrad durations 143.711/142.880/142.176 us; no speedup claim. Correctness
passes; cuDNN benchmark False, deterministic False, allow_tf32 True.
Next learner adds five perf_counter blocks of 100 helper calls after warmup/checks
and before profiler, sync at block boundaries, report us/completed call. Distinguish
helper wall time from dgrad-only device execution. Implementation pending.
First timing review: learner currently synchronizes/times each of 100 individual
calls using time.time, then averages. Explained this is serialized-call latency,
not requested block measurement. Next move both syncs/timestamps outside inner
loop, use perf_counter, store one duration/block and print after all five blocks.
No execution of this draft or new baseline measurements.
Second timing review: timestamps now surround block, but completion sync still
inside inner loop; printed block time lacks division by 100; printing loop binds
local name time, which shadows imported module throughout main and would cause
UnboundLocalError at first perf_counter call. Next fix all three. Not executed.
Third timing review: variable naming and normalization fixed, sync moved outside
inner loop but placed after end timestamp. Next swap sync and end timestamp so
remaining GPU work is included. No baseline run; timing boundary still assisted.

**Active learner task: profile isolated dgrad.** Reviewed and ran learner's
conv_dgrad.py: exit 0, float32 CUDA dX [32,512,2,2], finite, assert_close passes,
max absolute error 7.6293945e-06. Autograd reference shares backend; correctness
check is assisted API consistency, not independent backend proof. Next learner
adds 10 warmup calls and a 3-call CPU+CUDA capture of only the input-gradient
helper, each labeled dgrad_only; boundary synchronization, export
conv_dgrad.trace.json. Record cuDNN flags; then verify kernel name/grid/block
against model capture. No timing baseline or kernel match yet.
First profiling review: warmup/reference separation and cuDNN flag printing are
correct. Missing pathlib.Path import; capture contains one call instead of three;
final synchronization is outside profiler. Next learner fixes these, then run
and verify kernel selection. This draft reviewed only, not executed.
Second review: Path import and three-call loop fixed. Synchronization now sits
inside the loop (one wait per call), not after it. Next dedent synchronize to
the for-loop's indentation, keeping it within profiler; then run capture.

**Latest: candidate shapes recovered.** Learner ran shape capture; verified three
steps / 975 kernels, original captures preserved. For candidate 1208/7174:
grad_output and X [32,512,2,2], W [512,512,3,3], float32 contiguous;
stride/padding/dilation [1,1], groups 1, transposed False. Input ordering decoded
against installed operator schema. New kernel duration 142.302 us is diagnostic.
Next learner compares X/W element counts to revisit batch-size reasoning, then
implements an isolated input-gradient workload using these shapes/settings.
No isolated code or GPU experiment created yet; prior workload-ranking task done.
Count check follow-up: learner said X is larger; corrected with explicit arithmetic
X=65,536 and W=2,359,296 (36x X). Independent tensor-count reasoning not yet
demonstrated; kernel cost cannot be inferred from gradient output count alone.
Follow-up asks why dgrad is slower. Explained per-output accumulation and
implementation efficiency; shape trace pair is 142.302 versus 37.248 us. Exact
cause remains open pending isolated reproduction and kernel-level measurements.

**Current task: prepare one kernel workload for isolation.** Aggregated saved
step-2 backward work: 184 kernels / 1598.380 us. Largest exact-name group is
dgrad_engine: 4 calls / 449.821 us / 28.14% of backward kernel duration; largest
in steps 1 and 3 too. Selected previously inspected 142.015 us invocation as a
study candidate. Same name also has a 24.864 us launch with different grid;
tensor shapes/conv parameters are not recorded yet. Next recover those before
building an isolated reproducer. Learner predicts dgrad due to X's batch dimension;
explained that shared-weight gradients also accumulate across batch/spatial data.
Ranking is observed, causal explanation and isolated optimization remain open.

Immediate next action: learner could not find shapes; confirmed they are absent
from the existing trace. Learner adds record_shapes=True to active profiler and
exports resnet18_shapes.trace.json. Review before running; inspect CPU operator
input dimensions, then verify model-layer convolution settings. New capture/IDs
pending; no assistant source changes or GPU run for this instruction.
First review: record_shapes=True is correctly set, but export still targets
resnet18_phases.trace.json. Learner must change it to resnet18_shapes.trace.json
before running to preserve the timing/aggregation evidence. Not executed.

**Active task: interpret verified phase annotations.** Learner corrected the outer
scope and phase names. Bounded script run passed; resnet18_phases.trace.json has
three CPU train_step scopes with five ordered sibling phases each, GPU kernels,
and no device synchronization inside steps. Original trace checksum unchanged.
New capture includes gpu_user_annotation ranges; summary forward_pass Self CUDA
14.660 ms / 212.55% is not a valid fraction of kernel execution. Next open the
second CPU train_step scope and report forward_pass/loss_backward CPU durations;
then separately examine GPU attribution. Follow-up: learner read first-step
forward/backward durations and hypothesized activity-buffer overhead explains
smaller later steps. Verified buffer events overlap first forward/backward only
(1.763/1.748 ms), but do not fully account for the differences. Explained workload
versus profiler warmup. Next correlate one second-step backward kernel with its
CPU launch/operator, accounting for autograd worker threads. Implementation and
causal interpretation assisted; phase prediction remains unassessed.

Latest learner evidence: correctly selected second-step dgrad_engine kernel,
142.015 us, launching CPU thread 55549. Verified pt_autograd_0 worker, launch
correlation 7174, convolution_backward External id 1208; main CPU scope is on
55492. Learner then correctly identified wgrad_alg1_engine, 37.247 us, correlation
7194, also External id 1208. Verified three kernels for this operation (including
1.344 us scale kernel), totaling 180.606 us of kernel duration. Trace-linking task
completed with correct learner observations; explanations assisted. Next aggregate
backward kernel costs to choose a candidate for isolated study, without equating
kernel sums to elapsed phase time.

**Latest exercise:** learner identified two cudaFuncGetAttributes calls on the
launching thread between 13.333352 and 13.345284 ms. Verified durations 5.097 and
2.374 us (7.471 us total); about 4.46 us outside these calls remains unaccounted.
Explained attribute queries and bounded local interpretation. Learner proposed
one query per following kernel; this remains unverified because query records
omit the target function. Explained queries can be grouped before launches. Learner
supplied timings for the previously discussed kernel pair and correctly answered
that launch had not finished when the prior GPU kernel ended. Assisted refinement:
launch had not started; gap 20.256 us = 11.932 before launch + 6.894 inside launch
+ 1.430 after launch. Independent transfer to a different pair remains unchecked.
No new capture or code change requested; complete gap cause remains unresolved.
A different-pair transfer check remains pending; now moving to phase attribution.

Background from the preceding discussion:

1. Learner has opened Perfetto and observed gaps. Clarify CPU thread IDs versus
   CUDA stream IDs, nested CPU operation scopes, and highlighting versus actual
   device events. First conv2d CPU scope is 2127.788 us; its four associated GPU
   kernels sum to 31.839 us. These are different measurements.
2. Learner selected the second-repetition 11.808 us kernel and its 1.430 us launch
   flow. Matched correlation 5036; pasted args were from correlation 132 of the
   same kernel. Explained device 0 versus stream 7, 20.256 us preceding stream
   gap versus launch-end delay, and metadata/resource estimates. Follow-up explains
   operation IDs versus CUDA launch correlations, context resources, and queued
   timestamps. Further explained context implementation through memory mappings,
   work-queue ownership, and execution state; stream hierarchy means shared
   resource ownership. Learner requested simpler prerequisites: execution state
   and virtual addresses, illustrated with two independent training processes
   sharing a GPU. Next match a different kernel to its CPU launch and enclosing
   operation using both IDs; gap causality remains unresolved.
   Trace is local and Git-ignored; see the latest
   [timeline session](sessions/2026-09-27-project-02-timeline-reading.md).

Timing discussion started: learner recognizes the need to wait for GPU completion
with assistance. Clarifying that a CPU wall-clock timer includes waiting and that
earlier work can contaminate timing only to the extent it remains when timing starts.
Learner now reports understanding; independent application has not been checked.
Learner placed synchronization correctly around the measured block; one preliminary
run reported 4.475789 ms/step and 7149.577 images/s. Latest `perf_counter` run
completed five blocks: median 4.467561 ms/step, range 4.458838–4.512978 ms/step.
This is within-execution block variation, not established across-session stability.
Learner correctly identified training as included
and model setup/warmup as excluded; host/loop overhead distinction was explained.
Revision notes are in
each started project's `notes/` folder, with continued maintenance authorized.
Revision-note style preference: short bullets, clear headings, bold key terms,
numbered procedures, and small tables/examples. Existing notes reformatted;
`AGENTS.md` records this preference for future sessions.

Before moving to another machine, verify remote synchronization when authorized.

Initial benchmark acceptance criteria: reproducible configuration and seed, stated
device/software/dtype/batch/input shape, warmup, completed-work timing, repeated
measurements, step latency, throughput, and correctness checks. No optimization yet.
Synthetic pre-created inputs exclude real dataset loading and preprocessing; define
explicitly whether device transfer is inside or outside the timing boundary.

## Project state

| Project | Implementation | Learning evidence |
| --- | --- | --- |
| [1: Mini framework](projects/01-mini-framework.md) | Paused after scalar MLP; full project incomplete | Assisted implementation and passing checks; independent mastery not established |
| [2: Performance laboratory](projects/02-performance-laboratory.md) | Corrected five-block benchmark runs | Learner recognized variation can mimic a small gain; causal distinction clarified |

## Latest verified local environment

2026-09-25 current workspace: Linux 6.11.0-1016-nvidia x86_64; NVIDIA H100 80GB
HBM3, 81559 MiB reported memory, driver 580.173.02. Default `python3` is 3.14.7;
`importlib.util.find_spec("torch")` returned no module. Other environments,
CUDA execution, and profiler permissions have not been verified.

Follow-up: existing Conda environment `kvforge` has Python 3.12.14 and PyTorch
2.11.0+cu130 (built CUDA 13.0). `torch.cuda.is_available()` returned True;
torchvision imports and ResNet-18 executes. Two steps passed on `cuda:0` in training
mode, with finite scalar loss, gradients, and parameters. The subsequent bounded
timing run completed; see the latest review session for results and limitations.

Profiler setup check: torchvision 0.26.0+cu130, cuDNN version code 91900; supported
activities include CPU and CUDA; first capture returned GPU events. Fresh-process flags:
matmul precision `highest`, matmul TF32 False, cuDNN TF32 True, cuDNN benchmark
False, cuDNN deterministic False, deterministic algorithms False. These were read
without modification, not captured inside earlier timing runs. See the
[profiler setup session](sessions/2026-09-25-project-02-profiler-setup.md).

Project 1 checks previously passed on a Mac, not this Linux runtime. See the
[setup session](sessions/2026-09-25-git-setup.md) and the current
[Project 2 session](sessions/2026-09-25-project-02-scope.md).


## Follow-up — cuDNN search correctness failure (2026-09-27)

- Learner enabled `cudnn.benchmark=True` before warmup; user-reported run
  failed `assert_close` before timing: 55,004/65,536 elements mismatched,
  max absolute difference 0.006866455078125, max relative difference
  0.5672147274017334. Assistant reviewed source, did not reproduce the run.
- TF32 was allowed in the established environment. Algorithm-dependent
  precision/accumulation is a hypothesis, not a confirmed explanation.
- Next learner diagnostic: benchmark=True with cuDNN TF32 disabled before
  warmup in a fresh process; preserve tolerances and label new artifacts with
  precision. Any performance comparison requires equal precision in both modes.
- No learning-code edits or GPU runs by assistant. Consulted official PyTorch
  2.11 CUDA semantics documentation. Changes remain local; no commit/push.


## Follow-up — TF32-disabled diagnostic passes (2026-09-27)

- Reviewed learner edits: benchmark=True, allow_tf32=False before warmup;
  assertion unchanged, export named conv_dgrad_benchmark_on_no_tf32.trace.json.
- Ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_dgrad.py`; exit 0.
- Correctness passes, max absolute difference 0 in this run. Flags:
  benchmark=True, deterministic=False, allow_tf32=False.
- Five printed block averages: 120.88, 117.20, 117.21, 117.20, 117.23 us/call.
  Separate profiler reports three sm80_xmma_dgrad_implicit_gemm kernels,
  mean 115.882 us. Trace exported locally; raw artifacts not synchronized.
- Disabling TF32 supports precision/algorithm selection as a contributor to
  the reported mismatch; does not prove exact cause or independent correctness.
  Earlier ~144.5 us baseline allowed TF32, so this is not a controlled search-only
  speedup. One execution, no repeatability claim.
- Next learner interprets result, then measures benchmark=False with TF32 still
  disabled and a distinct trace filename. No assistant learning-code edits,
  commit, or push. Learning conclusion not yet supplied by learner.


## Follow-up — Matched-precision baseline capture inspected (2026-09-27)

- Learner correctly stated that TF32 must also be disabled for benchmark=False;
  reported running that configuration. Source verifies both flags False and
  distinct off/on precision-labeled trace filenames.
- Inspected saved traces, no new GPU run: off has three scale helper events
  (mean 1.365333 us) and three dgrad_engine events (mean 142.036 us).
  On has three sm80_xmma_dgrad_implicit_gemm events (mean 115.882333 us).
- Evidence: project_02_performance_laboratory/conv_dgrad_benchmark_off_no_tf32.trace.json
  and conv_dgrad_benchmark_on_no_tf32.trace.json in the same directory.
- App terminal unavailable; user's off-mode block timings and printed correctness
  error not read. Trace export is consistent with execution reaching the end,
  but exact stdout is unavailable. Next obtain those outputs, then ask learner
  to interpret matched-precision comparison before repeated-run confirmation.
- No speedup established from unprofiled off-mode timing yet. No source edits,
  commit, push, or remote synchronization check.


## Follow-up — Matched-precision timing output received (2026-09-27)

- User supplied benchmark=False, allow_tf32=False stdout: correctness passes,
  max absolute error 7.6293945e-06; blocks 147.48,144.46,144.41,144.46,144.44
  us/call. Median of printed values 144.46 us/call.
- Previously assistant-run benchmark=True, allow_tf32=False blocks:
  120.88,117.20,117.21,117.20,117.23 us/call; median 117.21 us/call.
- Both use unchanged per-run autograd consistency assertions. Direct cross-mode
  output comparison and repeated fresh-process confirmation remain pending.
- First matched comparison only; no new execution this follow-up. Ask learner
  for percent time reduction and their interpretation before supplying analysis.
- No learning-code edits, commit, push, or remote sync. Results remain local.


## Follow-up — Six fresh-process comparisons completed (2026-09-27)

- Reviewed learner CLI: explicit off/on choices map to booleans; integer run IDs
  retained. All trace names now end in .trace.json. No competing GPU processes
  were listed in the preceding preflight check.
- Executed six sequential fresh processes in order off/on/on/off/off/on, run IDs
  1–6, explicitly passing --cudnn-allow-tf32 off. Each used 10 warmups, five
  100-call blocks, and three separately profiled calls. All exited 0 and passed
  existing warmup-output versus autograd assertions. No learning-code edits.
- Median us/call by run: 144.83,117.26,117.20,145.47,144.45,117.28.
  Median of process medians: off 144.83, on 117.26; reduction
  19.04%, speedup 1.235x. Derived from rounded printed values.
- Off-mode reference max errors 7.6293945e-06; on-mode errors 0 in these runs.
  These checks are API consistency, not independent backend correctness.
- Evidence: progress/sessions/2026-09-27-project-02-cudnn-repeat-runs/ contains
  full stdout/stderr logs plus summary.json with exact commands, per-block
  timings, trace paths/hashes, kernel names/durations/grid/block.
- All off captures contain three scale helpers + three dgrad_engine kernels;
  all on captures contain three sm80_xmma_dgrad_implicit_gemm kernels.
- Remaining planned checks: final timed-output and direct cross-mode numerical
  comparison, input checksums, unrounded timing serialization. Current script
  does not implement those; no claims that they passed. Raw traces remain local.
- Next ask learner whether variation plausibly explains the observed difference,
  then complete direct cross-mode correctness validation before model transfer.
  Learner independently calculated 18.86% for the first pair; kernel explanation
  was supplied by assistant. No commit/push or remote sync check.


## Follow-up — Direct cross-mode correctness fails (2026-09-27)

- Reviewed learner save/check additions: final timed dX checked and CPU tensor
  copies saved after all timing, before profiler. Placement correct.
- Executed fresh off run 007 and on run 008, both with TF32 disabled. Existing
  warmup and final timed-output comparisons to each run's autograd reference pass;
  max errors off 7.6293945e-06, on 0. Logs saved in cudnn-repeat-runs directory.
- Loaded local saved tensors with weights_only=True on CPU. W and dY exactly
  equal; both dX finite. Direct cross-mode assert_close FAILS with unchanged
  float32 defaults: 12473/65536 mismatches, max abs 0.0003509521484375,
  max relative 0.009784771129488945 (atol 1e-5, rtol 1.3e-6).
- Exact metrics and tensor paths saved in cross-mode-correctness.json under
  progress/sessions/2026-09-27-project-02-cudnn-repeat-runs/.
- Per-mode backend-sharing references do not establish cross-mode agreement.
  Performance observations remain measured, but optimization has not passed
  planned numerical acceptance. No tolerance changes or confirmed numerical cause.
- Next learner builds higher-precision reference from saved identical float32
  inputs promoted to float64, outside timing; compare both modes against it.
  No assistant learning-code edits, commit, push, or remote sync.


## Follow-up — CPU float64 reference implemented and run (2026-09-27)

- User explicitly delegated accuracy-script implementation. Existing
  check_dgrad_accuracy.py was empty; assistant filled it. This is worked-through
  support code, not evidence of independent learner implementation/mastery.
- Ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/check_dgrad_accuracy.py`; exit 0 (diagnostic report, not acceptance).
- Script loads local runs 007/008 with weights_only=True, verifies exact W/dY
  equality, float32 finite saved tensors and dX shape, promotes original inputs
  to CPU float64, computes conv2d_input, and reports both modes without early
  exit on numerical mismatch. Explicit atol=1e-5, rtol=1.3e-6 preserved.
- Against float64 reference: off max abs 0.00011968580707844012, mean abs
  1.423485832375312e-05, 4821/65536 mismatches; on max abs
  0.00029977440388506693, mean abs 2.575085995746367e-05,
  9384/65536 mismatches. Both fail original tolerance; off is closer by max
  and mean error. Reference is higher precision, not exact arithmetic.
- Evidence saved as float64-reference.json in existing cudnn-repeat-runs
  session directory. This undermines treating baseline/backend-sharing
  autograd agreement as ground truth. No tolerances changed; measured timing
  advantage does not establish an accepted equivalent optimization.
- Next: explain accumulation error and distinguish chosen tolerance from an
  application accuracy requirement before choosing further validation.
  Changes local, no commit/push or remote sync check.


## Next task — Error scale (2026-09-29)

- Reviewed float32 addition order and tolerance defaults. Learner correctly
  predicted 1/0 for the supplied rounding example; near-zero tolerance reasoning
  correct, decimal conversion of 1e-5 corrected with assistance.
- Clarified that default assert_close failure is not proof of unacceptable
  training accuracy; no tolerance changed or new numerical experiment run.
- Next: compare each saved dX to the float64 reference using relative L2 error
  (norm of error / norm of reference), and inspect reference magnitudes of failed
  elements. These diagnostics contextualize error; they do not replace the
  existing acceptance check or establish downstream training quality.


## Follow-up — Relative L2 error verified (2026-09-29)

- Reviewed learner addition: torch.linalg.norm with no ord/dim computes the
  intended overall norm here. Executed `conda run --no-capture-output -n kvforge
  python project_02_performance_laboratory/check_dgrad_accuracy.py`; exit 0.
- Relative L2 errors: off 4.0907820293055765e-07; on
  8.088131227128621e-07. Existing max/mean errors and mismatch counts unchanged;
  both still fail the preserved elementwise criterion. No new GPU runs.
- Learner implemented supplied formula; prediction/explanation not provided.
  Small aggregate error does not imply all elements pass or establish training
  quality. Near-zero concentration remains unmeasured.
- Next: learner interpret aggregate versus elementwise error, then inspect failed
  elements by reference magnitude. No source edits by assistant or tolerance changes.


## Follow-up — Editable magnitude buckets implemented (2026-09-29)

- User explicitly delegated fix while retaining bucket list. Preserved latest
  learner edits; implemented internal boundaries [0.1,1,10] with automatic
  [0,first) and [last,infinity) tails, strict finite positive increasing boundary
  validation, reference-based masks, per-bin counts/rates, and empty-bin None.
- Added coverage assertions: bin totals equal tensor size and bin failure counts
  equal existing mismatch total. Tolerances unchanged; no learning mastery claim.
- Ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/check_dgrad_accuracy.py`; exit 0; coverage assertions passed.
- Bin totals: 120,1086,10193,54137. Off failures: 57,530,2886,1348;
  on failures: 63,601,3645,5075. Percent rates off:
  47.50,48.802946593,28.313548514,2.489979127;
  on: 52.50,55.340699816,35.759835181,9.374365037.
- Existing relative L2/mismatch results unchanged. No new GPU workload, tolerance
  revision, commit, push, or downstream accuracy claim.
- Next learner interpret per-bin rates versus raw failure counts: particularly
  search-on >=10 has most failures despite its lower failure rate.


## Follow-up — Move to exploratory whole-model comparison (2026-09-29)

- Learner independently explained why the 10+ bin can have more failures:
  it contains far more elements. Clarified rate versus count and influence of
  the tolerance formula. Added concise recap to note 11.
- Inspected benchmark.py: seeded ResNet18, batch32/images64, SGD, 10 warmup
  training steps, 5x50 timed steps, separate 3-step phase capture. Warmup mutates
  weights and BatchNorm buffers; current trace export would overwrite the
  historical resnet18_shapes.trace.json. No edits/runs this follow-up.
- Next milestone: exploratory model-level search off/on comparison, TF32 off
  in both, fresh processes with identical seeded initialization/input generation,
  equal warmup/update counts and unique trace names. Search can affect multiple
  convolutions; isolated speedup cannot predict model speedup directly.
- First teaching task: reason about model state if modes run sequentially on
  the same model before adding arguments. Existing numerical acceptance remains
  unresolved; model timing/sanity checks will not establish training quality.
- No assistant learning-code edits, tests, commit, push, or remote sync check.


## Follow-up — Model benchmark main wrapper (2026-09-29)

- User explicitly requested wrapping benchmark.py like the isolated scripts.
  Moved setup, warmup, timing, reporting and profiling into main with a main guard.
  Kept train_step/train_loop/profiled_train_step as module helpers; benchmark
  now receives model/optimizer/loss/input/targets/steps explicitly instead of
  relying on module globals. Preserved initialization order and workload.
- Syntax parsed and module imported successfully in kvforge; helper/main callables
  present, no model constructed at import. No full GPU benchmark or trace export.
- CLI/cuDNN configuration still pending for learner; no unrelated experiment
  changes, commit or push. Next learner adds mode/run-ID arguments, fixed TF32
  off and distinct trace filenames before whole-model comparison.


## Follow-up — First whole-model cuDNN comparison (2026-09-29)

- Verified learner trace filename includes both mode and TF32 argument values;
  parser/settings remain before initialization and warmup. GPU preflight listed
  no competing compute processes. No assistant learning-code edits.
- Ran two fresh kvforge processes, benchmark.py modes off/on, TF32 explicitly
  off, IDs 1/2; both exit 0. Seed42, ResNet18, SGD lr .01, synthetic batch32
  images64; 10 warmups then 5x50 timed steps plus separate 3-step capture.
- Off block ms: 3.79108804,3.99189432,3.78297808,3.78589958,3.78354948.
  On: 3.56896328,3.43436880,3.45913894,3.43669854,3.43218738.
  Median off 3.785899580 vs on 3.436698540 ms; 9.22% lower time in first pair.
- Final reported loss off 0.006522412411868572 vs on 0.006526823155581951;
  both finite, not proof of equivalent training or gradient/parameter accuracy.
- Each trace has 3 CPU train_step scopes; kernel counts off690/on693 across
  three steps. Attribution analysis not yet performed. No claim that the isolated
  dgrad improvement accounts for the full model difference.
- Full logs, exact commands, block timings, trace paths/hashes saved under
  progress/sessions/2026-09-29-project-02-model-cudnn-comparison/.
- Next learner interpret model versus isolated improvement; then repeat with
  varied process order before causal phase/kernel analysis. One process per mode,
  no across-run stability established. Earlier numerical acceptance unresolved.
- Notes local; no commit, push or remote sync check.


## Follow-up — Revision-note maintenance corrected (2026-09-29)

- User flagged missing revision updates. Numerical lessons were in note 11,
  but recent model-state, Amdahl and whole-model interpretation lessons were
  missing from quick notes despite progress-log updates.
- Added note 12 (from kernel to model performance), linked it in notes/README,
  and verified local evidence links in notes 11/12. Includes dX versus dW,
  Amdahl example, fresh initialization, repeatability and accuracy limitations.
- Learner correctly explained dilution by other training operations; clarified
  isolated workload covers dX only and model-wide search can affect many kernels.
- Next remains repeated whole-model comparison with varied process order.
  No new code, benchmark, commit or push in this documentation follow-up.


## Follow-up — Repeated whole-model comparison (2026-09-29)

- Reviewed current benchmark and checked nvidia-smi: no competing compute
  processes listed before runs. Ran six sequential fresh processes, order
  off/on/on/off/off/on, IDs3–8, TF32 explicitly off throughout; all exit 0.
- Five-block process medians in order (ms): 3.791747500,3.560385120,
  3.529186760,3.838448600,3.840634080,3.523138960.
- Median of process medians: off 3.838448600 ms, on 3.529186760 ms;
  8.06% time reduction, 1.088x speedup in this set. All on process
  medians below all off medians. Original pair ~9.22% remains historical;
  no cherry-picking or pooling to hide between-run variation.
- All final losses finite (see summary); no gradient/parameter checks or training
  equivalence established. Kernels per 3-step capture: off690 each;
  on699,699,690, versus original on693. Counts alone do not identify cause.
- Full stdout/stderr, exact commands, block results, trace paths/hashes and
  kernel counts saved in progress/sessions/2026-09-29-project-02-model-cudnn-repeat-runs/.
- Added quick-note interpretation: about 8% observed in repeats versus about9%
  first pair, consistent direction but variable magnitude. Next inspect paired
  median-representative traces (off006/on005), attribute actual kernel work to
  phases through CPU launch correlations, then investigate changed kernels.
- No learning-code edits, commit, push or remote sync check; numeric acceptance
  from isolated experiments remains unresolved. Learner interpretation pending.


## Handoff — Model phase attribution pending

- User authorized committing and pushing the current work. Remote fetch
  succeeded; final push outcome will be verified after commit.
- Current task is learner-led Perfetto comparison: second train_step in
  resnet18_benchmark_off_off_006.trace.json and
  resnet18_benchmark_on_off_005.trace.json. Record forward/backward CPU durations
  and identify a GPU kernel/launch correlation for each phase in each mode.
- Assistant inspected trace structure only; no phase aggregation completed.
  User explicitly requested performing the profiling exercise themselves.
- Later compare both absolute and relative attributed GPU-time reductions;
  distinguish those from wall-clock improvement. No numerical acceptance change.
- Code and compact logs/summaries accompany this handoff. Full traces and saved
  tensors are ignored local artifacts; pushing Git does not transfer them.
- Validation already performed: CPU float64 diagnostic with bin coverage checks,
  import-safe main wrapper, two initial and six repeated model runs. No new GPU
  runs for this commit; whitespace-only cleanup and diff check at handoff.


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

## Follow-up — Nsight introduction (2026-10-01)

- Starting commit71ec783. Next milestone planned: controlled isolated forward convolution and bounded Nsight Compute comparison.
- Read-only checks found Nsight Compute2025.3.1 and driver RmProfilingAdminOnly0; actual hardware-counter access untested.
- User requested prerequisite explanation. Introduced Systems versus Compute and hardware counters; revision note13 added.
- No new experiment or learning-target code. Understanding not yet assessed.
- Next: identify which tool fits GPU gaps versus behavior inside a kernel, then implement forward operation and correctness checks.
- Details: progress/sessions/2026-10-01-project-02-nsight-introduction.md. This follow-up remains local.


## Follow-up — First forward-script review

- Learner independently chose Systems for CPU activity during GPU gaps and Compute for memory traffic inside a kernel.
- Reviewed new conv_fprop.py: correct tensor shapes, float32 CUDA construction, seed42, convolution parameters, CLI backend settings and main guard; output shape assertion present.
- Output dtype, device and finiteness checks remain to be added. Default random tensors do not require gradients.
- Static review only; no GPU execution or numerical-reference validation yet. Learning-target code left unchanged.
- Next: learner adds the three missing output checks, then runs searchoff/TF32off before adding the float64 reference.

### Forward checks verified

- Learner added dtype, device and finiteness assertions. Assistant ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_fprop.py --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 1`; exit0, all four assertions passed.
- This validates output properties, not numerical accuracy. Next learner computes CPU float64 reference from the existing float32 input/weight and reports maximum absolute error against promoted GPU output. No timing/profiling performed.


### Float64 reference review — corrections pending

- Learner correctly reused existing input/weight values, converted them to CPU float64 and used identical convolution arguments.
- Reference was then cast back to GPU float32 before comparison; requested correction is to retain CPU float64 reference and promote GPU output to CPU float64.
- Reduction currently computes mean absolute error, not requested maximum absolute error. Also identified unnecessary `from kvforge import reference` import.
- Static review only; revised code not executed. Next: learner fixes comparison precision, reduction and import, then rerun searchoff/TF32off.


### Corrected forward reference executed

- Learner retained CPU float64 reference, promoted GPU output to CPU float64, changed reduction to maximum and removed the unused kvforge import after review.
- Ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_fprop.py --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 1`; exit0; printed maximum absolute error0.0001594768301060867. Output property checks passed.
- Comparison implementation corrected with assistance; no accuracy pass criterion evaluated yet. Printed MAD label remains ambiguous.
- Next: learner counts elementwise failures using explicit atol1e-5/rtol1.3e-6 on CPU float64 comparison; do not infer acceptance from maximum error alone or adopt float64 default tolerances.


### Delegated forward-error diagnostics completed

- User explicitly delegated repetitive diagnostic implementation. Assistant renamed maximum-error output, added explicit atol1e-5/rtol1.3e-6 elementwise failure count/rate and PASS/FAIL reporting, and checks finiteness before comparison.
- Executed fresh processes: `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_fprop.py --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 1` and the same command with benchmarkon/run-id2.
- Both exited0 and reported maximum absolute error0.0001594768301, 4736/524288 failures (0.90%), criterionFAIL. Diagnostic exit0 is not numerical acceptance. Original tolerances retained.
- Equal summary metrics do not establish equal output tensors or identical selected kernels. No kernel capture, timing or independent learner-mastery claim.
- Next: add warmup and block timing in separate diagnostic/timing paths, then verify actual selected kernels with a separate trace before Nsight Compute collection. Relative L2 diagnostic from milestone plan remains outstanding.
- Changes remain local; no commit or push.


### Delegated timing review and fixes

- User authorized fixes. Original synchronization boundaries and per-call conversion were correct. Assistant retained warmup/timed outputs, moved existing validation to final timed output, buffered times until all blocks complete, added median, explicitly set deterministicFalse, removed unused imports and corrected backend comments.
- Workload unchanged; 10 warmup calls, 5 blocks x100 calls, perf_counter and device synchronization at block boundaries. CPU float64 reference and diagnostics excluded from timing. No profiler active.
- Ran `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_fprop.py --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 3`: exit0; blocks67.50,67.53,67.54,67.56,67.57us; median67.54us.
- Ran same command with benchmarkon/run-id4 in a fresh process: exit0; blocks27.70,27.71,27.68,27.66,27.67us; median27.68us.
- Both final timed outputs reported maxabs0.0001594768301 and4736/524288 tolerance failures (0.90%). Failure is reported, not suppressed; exit0 denotes completed diagnostic.
- Initial pair only, about59.0% lower median call time. No competing-process preflight or six-process repetition campaign performed here; no selected-kernel identity established. Matching error summaries do not prove identical outputs/kernels.
- Next: separate benchmark/trace execution paths and capture warmed-up forward calls to verify kernel selection before Nsight Compute. Planned repeat campaign and relativeL2 remain outstanding.
- Implementation delegated, not independent mastery evidence. Changes local; no commit/push.


### Isolated forward kernel captures verified

- User requested captures. Assistant added --mode benchmark|trace (defaultbenchmark), separate three-call CPU/CUDA profiler with fprop_only labels, warmup outside capture, final synchronization, unique trace filenames and overwrite protection. Timed benchmark loop is skipped in trace mode.
- GPU preflight `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` returned no listed compute processes.
- Executed `conda run --no-capture-output -n kvforge python project_02_performance_laboratory/conv_fprop.py --mode trace --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 5` and same command with benchmarkon/run-id6. Both exit0; numerical summaries unchanged (4736/524288 failures). Profiler emitted cycle-clearing warning; all three expected CPU ranges and kernels verified in each exported trace.
- Searchoff kernel tile256x64x8: grid[1,32,1], block[128,1,1], registers/thread255, shared memory33280bytes; durations67.904,66.464,66.400us.
- Searchon kernel tile128x32x8: grid[2,64,1], block[128,1,1], registers/thread166, shared memory16896bytes; durations27.360,26.688,26.719us.
- Exact kernel names match representative ResNet second-forward call2. Each kernel maps via correlation to a unique launch inside a CPU fprop_only range. Trace occupancy fields remain estimates, not hardware-counter measurements.
- Machine-readable evidence and trace hashes: progress/sessions/2026-10-01-project-02-isolated-forward-captures.json. Raw trace files are ignored local artifacts, named conv_fprop_benchmark_off_tf32_off_005.trace.json and conv_fprop_benchmark_on_tf32_off_006.trace.json in project folder.
- Next: introduce NVTX marking and perform bounded Nsight Compute counter-access smoke test for this verified pair. No Nsight Compute capture yet; repeated-process timing and relativeL2 remain outstanding. Code/analysis delegated; no new independent mastery claim. No commit/push.


### Nsight Compute invocation explained

- User asked how to collect hardware counters. Explained NVTX as a CPU-side named range selecting kernels launched inside it; supplied ncu-mode branch and bounded first-capture command for learner implementation.
- First capture planned: searchoff/TF32off, run7, one matching fprop kernel, LaunchStats/Occupancy/SpeedOfLight, kernel replay, cache-controlall, clock-controlnone, timeout300s. Output project_02_performance_laboratory/conv_fprop_ncu_off_007.ncu-rep.
- Checked current script and installed help; verified trailing-slash push/pop syntax against NVIDIA CLI docs. No code changes or Nsight Compute execution this follow-up.
- Next: learner adds ncu choice/branch and runs first command; review report or exact error. Counter permissions remain untested.


### First Nsight Compute counter capture succeeded

- User explicitly requested assistant execution. Learner added ncu branch; assistant removed quit() because it bypassed post-capture validation. No other learning-target changes this follow-up.
- Preflight listed no GPU compute processes and report target did not exist.
- Executed `conda run --no-capture-output -n kvforge timeout 300s ncu --nvtx --nvtx-include 'conv_fprop_target/' --kernel-name 'regex:sm80_xmma_fprop_implicit_gemm' --launch-count 1 --section LaunchStats --section Occupancy --section SpeedOfLight --replay-mode kernel --cache-control all --clock-control none --export project_02_performance_laboratory/conv_fprop_ncu_off_007 --page details python project_02_performance_laboratory/conv_fprop.py --mode ncu --cudnn-benchmark off --cudnn-allow-tf32 off --run-id 7`.
- Exit0, one matching tile256x64x8 kernel, 10 replay passes. Hardware-counter collection now confirmed accessible. Raw .ncu-rep exists as ignored local artifact.
- Launch32 blocks x128threads on132SMs,255regs/thread,33280B dynamic shared memory. Register block limit2, shared-memory block limit3; theoretical8activewarps/SM and12.5%occupancy; achieved4activewarps/SM and6.25%occupancy.
- Profile duration73.98us; computeSMthroughput17.05%, DRAMthroughput0.94%. Nsight flags undersized grid; rule estimates are not measured speedups. Occupancy is not whole-GPU utilization. Unfixed-clock warning expected from clock-controlnone; no clock settings changed.
- Final output checks executed; numerical criterion still fails4736/524288, maxabs0.0001594768301. Completed capture does not resolve numerical acceptance.
- Exported raw counter CSV via `ncu --import ... --page raw --csv`, without GPU rerun; small JSON includes report SHA256 and key metrics: progress/sessions/2026-10-01-project-02-ncu-off-007.{csv,json}.
- Next: explain register-limited theoretical occupancy versus achieved occupancy and grid coverage, then collect matched searchon report. Repeated counter/timing runs still outstanding. No independent learner interpretation yet, no commit/push.


### Occupancy recap and learner check

- Learner opened report on Mac and read register block limit2, shared-memory block limit3, theoretical active warps8.
- After requesting a refresher, learner correctly computed4/64 =6.25% occupancy. This demonstrates the immediate arithmetic after explanation, not independent transfer or full report interpretation.
- Added requested short occupancy recap to note13: resident versus executing, resource ceiling, achieved residency, H100 example and distinction from grid coverage/utilization.
- No code edits or GPU runs. Next: connect achieved4warps/SM to the small grid, then obtain matched searchon counter report. Changes remain local.


### Matched search-on Nsight Compute capture

- User authorized execution; script checked, no source changes required. GPU preflight listed no compute processes and new report path was unused.
- Ran `conda run --no-capture-output -n kvforge timeout 300s ncu --nvtx --nvtx-include 'conv_fprop_target/' --kernel-name 'regex:sm80_xmma_fprop_implicit_gemm' --launch-count 1 --section LaunchStats --section Occupancy --section SpeedOfLight --replay-mode kernel --cache-control all --clock-control none --export project_02_performance_laboratory/conv_fprop_ncu_on_008 --page details python project_02_performance_laboratory/conv_fprop.py --mode ncu --cudnn-benchmark on --cudnn-allow-tf32 off --run-id 8`.
- Exit0; expected tile128x32x8, grid[2,64,1], block[128,1,1],166regs/thread;10 replay passes. Final numerical criterion unchanged:4736/524288failures,maxabs0.0001594768301.
- Searchon: register block limit3, shared-memory block limit7; theoretical12warps/SM,18.75%occupancy; achieved4warps/SM,6.25%occupancy. Profile duration30.27us, computeSMthroughput57.77%, DRAMthroughput2.27%.
- Searchoff comparison:32vs128blocks,12.5%vs18.75%theoretical occupancy, achieved6.25%in both, computeSMthroughput17.05%vs57.77%, profile duration73.98vs30.27us. Same achieved occupancy coexists with different throughput and duration; broad scheduling/cause attribution not established from occupancy alone.
- Both used kernel replay/cache-controlall/clock-controlnone. Unfixed-clock warning; measuredSMfrequency1.99GHz off vs1.96GHz on. Single capture per mode; do not equate instrumented timings with final benchmark estimate or rule estimates with measured gains.
- Saved raw CSV export and report hash/metric summary at progress/sessions/2026-10-01-project-02-ncu-on-008.{csv,json}; binary report ignored/local. No commit/push.
- Learner prediction not supplied before capture; next ask learner to interpret unchanged achieved occupancy despite larger grid and higher throughput. Further repetitions/compute-memory sections and relativeL2 remain outstanding.


### Throughput and exact resource accounting explained

- Learner proposed wider parallelism from128blocks but called hardware SMs blocks. Corrected terminology and qualified exact scheduling as unproven from grid count alone.
- User asked compute throughput definition/calculation and registers/shared-memory accounting. Inspected saved raw CSVs and installed SpeedOfLight.section; consulted NVIDIA profiling and Hopper guides. No new GPU runs.
- Verified SM throughput metric uses avg.pct_of_peak_sustained_elapsed; its value matches sm__issue_active in both reports (17.045637%/57.768060%). Explained maximum normalized constituent metric versus convolution FLOP/s and occupancy.
- Verified allocated register slots/thread256/168 versus reported255/166;128threads gives32768/21504slots per block, accounting for register limits2/3 with65536slots per SM.
- Verified configuredSMsharedmemory135168B, static0B, dynamic33280/16896B, driver1024B; totals34304/17920B give sharedmemory block limits3/7. Explained compiler/kernel choices determine requirements; exact cuDNN buffers not inferred.
- Extracted values/units saved in progress/sessions/2026-10-01-project-02-resource-accounting.json; concise revision section added to note13. New concepts explained, independent transfer untested.
- Next: learner interprets occupancy versus throughput using this comparison before expanding compute/memory counter collection. Changes local; no commit/push.


### Driver reservation and throughput foundations

- User requested driver reservation meaning and a first-principles throughput explanation; no independent understanding assumed from previous explanation.
- Checked NVIDIA Hopper guide: CUDA reserves1KiBsharedmemory/block; exact internal contents not specified. Explained using existing33280+1024B accounting.
- Explained machine instructions, ready warp scheduling, execution pipelines, clock cycles, activity/time, peak normalization and Nsight maximum-of-constituents aggregation. Toy arithmetic and four-SM examples separate occupancy from execution activity and coverage.
- Notes13 updated; no source changes or GPU experiment. Next learner explains how occupancy can stay fixed while compute throughput rises before collecting deeper sections/repetitions. Local changes only.


### Hardware counters and pipeline breakdown

- User asked whether the report identifies hardware units, what counters are, whether memory loading counts and why throughput is below100%.
- Read existing CSVs (no GPU rerun): FMAactive%12.856127/35.187090, ALUactive%5.289765/31.182361, LSUinstruction%3.532645/17.338046, tensor-MMA-specificactivity0/0; instructionissue17.045637/57.768060. Units/pipeline definitions checked against NVIDIA guide.
- Explained event/cycle counters versus derived metrics, normalization and overlap (not summable instruction shares), memory issue versus completed transfers, and potential limits without diagnosing unmeasured stalls.
- Note13 updated. Next inspect the report's throughput breakdown with learner, then expand memory/scheduler collection if needed; no new code, captures or mastery claim. Local only.


## Handoff — Ready for offline report study (2026-10-01)

- User requested wrapping GPU runs and preparing a local learning handoff. Completed6fresh benchmarks and8expanded Nsight captures:3off originalkernel,3on originalkernel,2on alternatekernel. All reports imported/identity-checked and all12requested sections present;36replaypasses each.
- Unprofiled medianofmedians67.76275us off vs27.65771us on,59.1845%reduction for this synthetic forward operation. Capture identities do not establish identities in separate unprofiled runs.
- On031/032 selected alternate tile32x32x8; on035/036/037 reproduced tile128x32x8. Preserve variants; do not average their counters as one kernel. Cause of selection variability unproven.
- Added optional resultsJSON/tensor exports and relativeL2 diagnostic. Saved off020/on021 input/weight/output exactly equal; relativeL2~4.1820152e-7, existing criterion fails4736/524288. Numerical acceptance unresolved.
- Latest session: progress/sessions/2026-10-01-project-02-offline-handoff.md. Exactcommands/logs/counters: progress/sessions/2026-10-01-project-02-offline-batch/. Start locally with OFFLINE_PROJECT02.md.
- Package includes current source/notes/progress and ignored binary reports, timelines and tensor snapshots. No commit/push or cloudshutdown performed. Transfer archive to Mac and verify before stopping rental; remote copy not verified here.
- Next task requires no GPU: inspect Scheduler Statistics in expanded off030/on035 and teach resident/eligible/issued warps, then request learner interpretation. Full project/independent exit incomplete.


## Follow-up — Organized folder handoff replaces archive workflow

- User requested ordinary folders/subfolders and updated Python output/input paths. Moved38original artifacts with SHA256 unchanged:10Nsightreports,24timelines,4tensors.
- New canonical data root: project_02_performance_laboratory/results/. Workload folders resnet18,conv_dgrad,conv_fprop; traces/tensors/benchmarks separated. Nsight basic versus expanded reports separated, with expanded searchmode/tile/run subfolders.
- benchmark.py and conv_dgrad.py write under their workload result folders; check_dgrad_accuracy.py reads relocated dgrad tensors. conv_fprop.py writes traces under results/conv_fprop/traces, defaults machine-readable benchmark/diagnostic JSON to results/conv_fprop, and creates parents for explicit result/tensor destinations. Nsight binary output is still controlled by ncu --export, not the Python script.
- Historical executed commands/evidence records preserved; progress/sessions/2026-10-01-project-02-result-layout.json resolves original artifact paths. Root OFFLINE_PROJECT02.md updated to current report locations and recursive folder download.
- results/ includes README, copied notes/source/session/instruction context, original binary reports, text logs and CSV exports. Whole projectfolder is recommended for executable scripts; resultsfolder alone is sufficient for report study.
- Verified syntax/import safety of all4Pythonfiles and ran check_dgrad_accuracy on CPU: relocated snapshots load and reproduce4821/9384mismatches. No GPU experiments rerun.
- Results folder remains Gitignored; it must be copied separately. Prior tar retained as an earlier backup, not current handoff. No commit/push, Mac transfer or rental shutdown performed.
- Next session: open results/README.md, then original expanded off030/on035 Scheduler Statistics. Full project/accuracy acceptance remain incomplete.


### Folder verification completed

- Results folder contains228payload files (~71.9MiB), including10binary Nsight reports,24timeline traces and4tensor snapshots. Standard-library `verify_download.py` passed all228SHA256/size checks.
- All38original artifacts retain their pre-move hashes. Root offline guide and resultsREADME local links resolve. Current source and session snapshots included for local continuation.
- Download project_02_performance_laboratory recursively; read results/README.md. Only results/ is also self-contained for report study. No tar extraction is needed.


### Local commit requested

- User authorized committing all completed code, evidence, notes and handoff changes; results/ remains ignored and will be downloaded separately.
- Only the results folder is needed for offline study: README, reports, copied source/instructions/notes/session context, and the standard-library verifier are bundled there.
- Validation already passed: four Python syntax/import checks, CPU dgrad reference using relocated tensors, original artifact hash preservation, folder manifest verification and whitespace check. No additional GPU runs needed.
- No push or shutdown requested in this action. The results snapshot will record the completed commit identifier after commit succeeds.
