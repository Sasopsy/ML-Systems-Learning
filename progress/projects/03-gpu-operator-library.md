# Project 3: GPU Operator Library

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
