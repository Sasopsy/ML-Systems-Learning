# 2026-10-23 — Stable softmax foundations

## Context

- Project/milestone: Project 3, stable fused softmax baseline before online/tuned variants.
- Mode: TUTOR.
- Learning target: numerical stability and adapting row reduction to maximum then exponential sum.
- Starting commit: f6d6002; prior kernels/planning/notes already uncommitted.
- Environment: local macOS; no GPU or compiler environment used.

## Work and evidence

- Read learning instructions, status, Project 3 progress, latest planning session, and row-sum source.
- Established first contract: [37,130], contiguous FP32 input/output/intermediates, separate output, one block per row with 128 threads, finite values and no mask.
- Learner correctly identified overflow/infinity/NaN for direct exponentials of [1000,1001,1002] and explained common exp(-max) cancellation in numerator/denominator.
- Added concise stable-softmax note and progress updates. No kernel code written or changed.

## Experiment

- None. No compilation, numerical test execution, profiling, or remote action.

## Learning and handoff

- Learner conclusion: subtracting a shared maximum preserves the mathematical probabilities through cancellation.
- Correct immediate conceptual answer; implementation, synchronization, precision acceptance, and independent transfer remain unassessed.
- Next task: derive suitable per-thread maximum initialization and the shared-tree combine operation.
- Changes local and uncommitted; no commit/push. Earlier GPU batch remains deferred by user.

## Maximum reduction, shared reuse, and skeleton

- Learner correctly chose negative infinity, maximum combine, and a barrier after reading the shared row maximum before overwriting shared storage. CUDA fmaxf syntax supplied by assistant.
- Explained read-before-overwrite versus write-before-read barriers. Agreed teaching baseline recomputes exponentials for output; retaining values is future optimization.
- User requested skeleton; assistant created `kernels/cuda/softmax.cu` under Project 3 with template/signature, shared array, row/thread identifiers, and six TODO stages. No reduction/normalization implementation supplied.
- Updated note 04 and progress. Next: learner implements kernel body. No compile/run, commit, push, or remote activity.

## First softmax source review

- Read full saved `softmax.cu` with `cat`. Learner implemented all six stages correctly for the documented contract, including the shared-memory read-before-overwrite barrier.
- Final barrier after copying denominator is redundant but safe; explained because no later shared-memory write exists. No kernel edits performed.
- This is guided implementation plus static review, not compilation, execution, or independent-transfer evidence.
- Next: host validation with stable FP64 reference, specified tolerances, invariants, and varied logits/row widths. GPU session remains deferred. No commit/push.

## Delegated softmax host scaffolding and CPU tests

- User requested host scaffolding, reference, and cases. Added a 10-case host harness to `softmax.cu`; kernel body unchanged. Learner had removed the redundant final barrier before this read.
- Added `softmax_test_support.h` with CPU-only fixtures/reference/validator and `softmax_reference_test.cpp`. Reference uses double arithmetic on promoted FP32 values, maximum subtraction, and sequential summation.
- Fixed tolerances before GPU testing: atol=1e-7, rtol=1e-5, row-sum atol=1e-5. Explicitly reject nonfinite and out-of-range outputs, report max errors and counts, return failure if any case fails. CUDA API/execution errors terminate with a nonzero status.
- Cases: mixed [37,130], singleton columns, equal short and 128-wide rows, negative width127, large positive width129, large negative width130, tied maxima width257, dominant tail width1025, mixed width4097. Output initialized with NaN sentinel to detect missing writes.
- Executed: `clang++ -std=c++17 -Wall -Wextra -Werror project_03_gpu_operator_library/kernels/cuda/softmax_reference_test.cpp -o /tmp/ml_systems_softmax_reference_test` then the temporary executable. Exit0, CPU reference/validator checks passed.
- CPU checks: known uniform/singleton/two-logit results, shift invariance, rounded-reference acceptance for all cases, rejection of zeros/invalid probabilities/NaN/infinity and permuted normalized outputs. This does not compile or execute CUDA or validate GPU synchronization.
- No GPU contact, benchmark, commit, or push. Next: deferred GPU correctness batch when user is ready. Scaffolding is delegated implementation.

## Warp reductions requested

- User requested another optimization level for row sum and softmax. Read both current kernels and checked NVIDIA shuffle documentation for mask/participation and memory-ordering rules.
- Plan: retain baselines, introduce warp-private reduction followed by shared exchange of four warp totals and warp-0 merging, then adapt to softmax's all-thread result use.
- Explain full-mask calls by all 32 lanes, identity values for lanes without data, and that shuffle does not replace block barriers for cross-warp shared-memory communication.
- Hypothesis is reduced shared traffic/barrier cost; no benchmark or speedup claim. Next learner checks neutral values for final-warp unused lanes; implementation pending.
