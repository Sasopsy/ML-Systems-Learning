# 2026-10-24 — Warp-reduction synchronization

## Context

- Project 3, TUTOR; local macOS. Learning target: warp reductions and cross-warp result lifetime.
- Existing code and progress changes were already uncommitted; GPU batch remains deferred.

## Work and evidence

- Read repository learning instructions, current progress, prior softmax session, and row-sum source. No kernel edits.
- Learner correctly supplied 0 / negative infinity identities and all-thread use of the softmax maximum.
- Learner requested thread synchronization; assistant clarified `__syncthreads()` and shuffle scope.
- Learner correctly explained that another thread could overwrite the shared maximum before all readers consume it. These are guided conceptual checks, not independent implementation evidence.
- Added topic note 05 and updated indexes/progress. No compilation, CUDA execution, timing, or remote action.

## Learning and handoff

- Next: learner adds a separate warp row-sum kernel alongside the baseline, starting with private accumulation and the first warp shuffle reduction.
- Preserve workload and softmax exponentiation policy for later matched comparison. Performance benefit remains a hypothesis.
- Changes local/uncommitted; no commit or push.

## Delegated skeleton follow-up

- User requested scaffolding. Added `row_sum_warp` to `project_03_gpu_operator_library/kernels/cuda/row_sum.cu`: template checks, shared warp partials, row/thread/lane/warp identifiers, and TODOs for accumulation, warp reduction, shared exchange, final merge, and output.
- Baseline kernel and host launch unchanged. Incomplete variant is explicitly labeled and is not launched.
- Next: learner implements TODOs 1–2 for review. No CUDA compilation, execution, or performance claim. Ran `git diff --check`; no whitespace errors reported.

## First learner implementation review

- Inspected saved `row_sum_warp` after user requested review of TODOs 1–2. TODO 1 correct; TODO 2 places full-mask shuffle inside `lane < d`, excluding required participant/source lanes.
- Later TODOs were also implemented: all lanes write the same per-warp shared slot; second reduction repeats private sums without reading shared partials; each warp's lane 0 writes the same output. Flagged these as pending issues, with first shuffle participation as the immediate fix.
- No code edits, compilation, or device execution. Host still launches the baseline. Guided implementation not yet correct.

## Follow-up source review

- Re-read saved kernel: learner fixed first shuffle participation, lane-0 shared writes, warp-0-only final reduction and output ownership; block barrier remains correctly outside conditions.
- Remaining correctness issue: second reduction never loads `warp_partials`; it reuses old register sums from warp 0. Next task is loading one partial per lane for lanes below `num_warps`, zero otherwise, before that loop.
- Added initial zero stores to shared partials are redundant because each slot is overwritten by its owner before any read. No code changes or CUDA execution during review; baseline remains the host launch target.

## Completed row-sum source review

- Re-read kernel and searched launch sites. Learner correctly replaces warp 0's register values with shared partials for lanes below `num_warps`, zero otherwise. Full-warp shuffle and single output writer follow correctly.
- No remaining source correctness issue found for the documented launch/input contract. Redundant initial shared zero stores and stale exercise comments remain optional cleanup.
- Host still launches `row_sum`, not `row_sum_warp`. No compiler/device tests or performance measurement. No code edits during review; implementation follows guided corrections.
- Next learning task: warp softmax with block-wide result publication and safe shared reuse. Baseline/variant device correctness and matched timing remain deferred.

## Delegated warp-softmax skeleton

- User requested scaffolding. Added `softmax_warp` beside the baseline with full-warp template constraints, shared storage, identifiers, identity initializers and eight staged TODOs. No reduction/broadcast/normalization body supplied.
- Comments guide block-wide publication of the maximum and denominator, protection of maximum reads before shared reuse, identity padding, and keeping exponential recomputation unchanged.
- Host still launches baseline; numerical reference/cases/tolerances unchanged. Next: learner implements private maximum and first warp max reduction (TODOs 1–2).
- Ran `git diff --check`; no whitespace errors. No CUDA compilation or execution, commit, push, or remote activity.

## Warp softmax first implementation review

- Read the entire saved warp kernel when user requested TODOs 1–2 review. Both are correct: strided maximum accumulation with -INFINITY and unconditional shuffle/max reduction. Learner had also filled the remaining stages.
- Found two blocking source issues: `row_max` is assigned without a declaration; each warp leader stores its sum to `warp_partials[0]`, racing and leaving old maxima in other slots used by the final sum reduction.
- Maximum merge/publication and block barriers around shared-result lifetime are correctly structured; final denominator publication precedes normalization with an unconditional block barrier.
- Next: learner fixes local declaration and per-warp sum store. No implementation edits, CUDA compilation, or GPU execution during review.

## Completed warp-softmax source review

- Inspected full saved kernel and launch site. Both fixes are present: local `float row_max`, per-warp sum slots. No remaining source correctness issue found under the finite FP32 input and matching complete-warp launch contract.
- Source counts for N=128: warp softmax uses four shared floats/five block barriers; baseline uses 128 floats/seventeen barriers. Added these comparisons to note 05; they do not establish speedup.
- No kernel edits, compiler/device tests, timing, or remote activity. Host still launches baseline only; stale skeleton/TODO comments are cosmetic.
- Next: extend both host harnesses to validate baseline and warp variants against the same references, followed by deferred GPU compilation/correctness. Implementation follows guided review; independent transfer is still unassessed.

## Delegated baseline/warp host validation

- User requested host scaffolding for both variants of both operators. Updated host code in `row_sum.cu` and `softmax.cu`; kernel bodies unchanged. Corrected stale baseline-only launch comments.
- Row sum: 37 rows, widths 1/17/31/32/33/127/128/130/257/1025; preserved original integer input formula and exact acceptance. Builds sequential double reference once per shape. Softmax retains all ten existing fixtures and fixed validator tolerances.
- Each variant independently resets device output to NaN, launches with 128 threads/block, checks launch/completion, downloads, validates against the shared reference, and reports labeled results. Numerical failures accumulate across runs; CUDA errors terminate. Allocations freed after both variants. Each executable schedules 20 variant/case runs; no timing introduced.
- Executed `clang++ -std=c++17 -Wall -Wextra -Werror project_03_gpu_operator_library/kernels/cuda/softmax_reference_test.cpp -o /tmp/ml_systems_softmax_reference_test` and the executable: PASS. This tests CPU support only.
- Executed Python fixture audit: parsed row-sum widths from source, checked each reference against `cols*row + r*(r-1)/2 - 8*r` for `r=cols%17`, and bounded sums of absolute inputs below 2^24. PASS for 370 rows, supporting exact FP32 comparison regardless of reduction order for these fixtures.
- Inspected final host sources; `git diff --check` passed. `command -v nvcc` returned no path. No CUDA compilation, GPU execution, benchmark, remote action, commit, or push.
- Next: deferred GPU validation of both executables, including memory/synchronization diagnostics before timing. Changes remain local/uncommitted.

## Online softmax lesson start

- User requested online softmax next. Read stable-softmax notes and roadmap; opened original paper at https://arxiv.org/html/1805.02867v2 and verified online normalizer recurrence and parallel pair formulation.
- Initial teaching: current kernels have three logical input traversals; online normalizer combines maximum/sum traversal, retaining an output traversal. Fewer logical reads do not directly imply fewer DRAM reads or speedup.
- Introduced running state (maximum, sum shifted by that maximum) with [2,3] then 5; next learner check asks for rescaling old contributions under the new maximum. Recurrence/code not yet supplied; parallel mapping remains next.
- Added note 06/index/progress. No kernel changes, numerical experiment, GPU activity, commit, or push.
