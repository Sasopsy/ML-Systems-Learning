# 2026-10-21 — Affine kernel indexing

## Context

- Project/milestone: Project 3, first CUDA C++ elementwise kernel.
- Mode: TUTOR / REVIEW; user writes the kernel body.
- Learning target: thread-to-element mapping and partial-block handling.
- Baseline: last verified commit `f6d6002`; new kernel untracked at review.
- Environment: macOS; no compiler/framework version inspected or GPU contacted.

## Work and evidence

- Assistant supplied a function signature and TODO skeleton in `project_03_gpu_operator_library/kernels/cuda/affine.cu`.
- Learner supplied correct index and guard in conversation; ceiling-after-integer-division block count corrected with explanation.
- Learner correctly identified final-block threads 0–103 as valid and 24 as skipping accesses for N=1000, B=128.
- Read saved source with `cat`; it contains the correct index, bounds check, and affine expression. No learning-target edits during review.
- `git status --short` showed the kernel directory untracked before this documentation update.
- Added concise lesson notes and progress updates.

## Experiment

- No compilation, device execution, numerical check, benchmark, or profiling performed.
- FP32 separate contiguous arrays, scalar a/b, positive N; numerical acceptance and launch/test harness still pending.

## Learning and handoff

- Learner conclusion: final-block count supplied correctly; reports implementation done.
- Evidence: body implemented by learner after jointly worked mapping; independent unfamiliar-case transfer not tested.
- Next: teach host launch syntax and have learner derive launch configuration.
- Source and new notes remain local/uncommitted; no push or remote synchronization.

## Follow-up — host harness source reviews

- Learner expanded the source to a standalone host harness. Repeated `cat` reads verified each saved revision; assistant did not edit implementation.
- Corrected with hints: using N before declaration, void main, uninitialized input, use after host deletion, preloading expected outputs, and cleanup bypass on mismatch.
- Latest source removes output upload and runs comparison before host deletion. N=1000 and 256 threads/block exercise a partial final block.
- Remaining: falling off main still returns success on mismatches; success message absent; all CUDA calls unchecked. Next explain explicit result status and CUDA failure handling.
- No compilation, GPU execution, or validation result. Error-handling guidance checked against NVIDIA asynchronous-execution documentation.

## Final source review

- Re-read the complete saved file: explicit headers, correct launch/indexing/guard, initialized input, no unnecessary output upload, checked CUDA calls, launch/error synchronization, comparison before host cleanup, explicit PASS and mismatch exit status.
- No blocking source-level finding for N=1000, FP32, a=2, b=1. Unused TODO wording is cosmetic. No learning-target source edits made by assistant.
- `git diff --check` passed for tracked changes; new source is still untracked. This is not compilation or GPU validation.
- Next: inspect GPU environment, compile, and run the first correctness test. No remote connection, commit, or push performed.

## Follow-up — GPU session deferred by user

- User requested batching several small kernel implementations locally before a separate GPU session.
- Proposed next operators: 2D bias + ReLU (flattened indexing and broadcast), then row sum (cooperation and reduction).
- Current task becomes deriving indexing for contiguous FP32 X[37,130] plus bias[130]. No new kernel code or tests yet; affine remains source-reviewed only.

## Bias + ReLU implementation review

- Learner correctly supplied 37*130 elements, flat bound, and modulo bias index. Assistant supplied a signature skeleton in chat.
- Initial saved implementation used undeclared `value` and added bias after ReLU. Explained ordering with x=-3, bias=2: required output 0 versus 2 from the incorrect order.
- Re-read corrected `project_03_gpu_operator_library/kernels/cuda/bias_relu.cu`: guarded input+bias followed by ReLU is correct for the specified initial contract.
- No implementation edits by assistant, compilation, or GPU execution. Next: learner writes a host correctness harness with sign/broadcast edge cases; later GPU batch unchanged.

## Delegated bias + ReLU scaffolding

- User explicitly requested scaffolding. Assistant added host allocations, deterministic signed input/bias initialization, checked transfers, output NaN sentinel, launch and execution checks, nested CPU reference, bounded mismatch logging, cleanup, and status return in `bias_relu.cu`.
- Learner kernel logic preserved; removed one trailing whitespace character. No affine implementation changes.
- Executed CPU-only Python arithmetic checks of the chosen fixture: 4810 flat/nested matches, both sign-crossing directions and zero sums present; incorrect ReLU-before-bias order disagrees at 2316 elements. These checks do not execute the CUDA file.
- Source trailing-whitespace check and tracked `git diff --check` passed. No compilation, GPU session, benchmark, commit, or push.
- Next: teach row-reduction work assignment. GPU validation remains explicitly deferred by user; harness work is delegated, not independent learner evidence.

## Row-sum lesson and skeleton

- Taught one block per row, thread-strided private sums, shared staging, and halving tree reduction with barriers outside the participating-thread branch.
- Learner correctly answered column assignment, intermediate sums 16/20, and barrier participation reasoning after instruction.
- Added signature/shared-array/TODO skeleton `row_sum.cu`; learner implements reduction body next. Exactly rows blocks and 128 threads/block required; no host harness yet.
- Added revision note 03 and reduction section index. No compilation, execution, remote session, commit, or push.

## First row-sum source review

- Read learner implementation: thread-strided private sum, shared store/barrier, halving conditional additions with unconditional barriers, and thread-0 output are correct under the intended 128-thread launch.
- Learner generalized shared size/stride/reduction bound to template N, but declared `template <block_size N>` with no such type. Provided `template <int N>` syntax and explained matching actual launch size plus power-of-two requirement.
- No learner code changed by assistant. Next: learner fixes template declaration; then add host scaffold. No compilation or GPU execution.

## Delegated template correction

- User requested correction; assistant changed only `template <block_size N>` to `template <int N>` in kernel source.
- Reduction body unchanged. Next: host harness. No compilation, GPU execution, commit, or push.

## Delegated row-sum host scaffolding

- User requested host support analogous to bias + ReLU. Added stdio/stdlib headers, checked allocation/transfers/free, NaN output sentinel, matching template/launch constant 128, sequential CPU double reference, capped mismatch logging, cleanup, and result exit status.
- Learner reduction body unchanged. Fixture: rows=37, cols=130, x[row,col]=row+(col%17)-8. Reference row totals are 130*row-33; bounded integer partial sums permit exact FP32 comparison.
- Ran CPU-only Python fixture/model verification with struct-based FP32 rounding at every addition: all 37 modeled tree sums match sequential reference, row totals are distinct, and dropping columns 128/129 changes every row. Not execution of the CUDA source or validation of its barriers.
- Source whitespace check and tracked `git diff --check` passed. No GPU connection, compilation, profiling, commit, or push.
- Next planned milestone: separate GPU correctness batch for affine, bias + ReLU, and row sum when user is ready. Scaffolding is assistant-built support, not independent learner implementation evidence.
