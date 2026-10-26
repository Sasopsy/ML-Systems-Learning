# 2026-10-26 — Online merge helper review

## Context

- Project 3, TUTOR/REVIEW, local macOS. User requested review of helper M1–M2.

## Work and evidence

- Inspected saved `merge_online` in `project_03_gpu_operator_library/kernels/cuda/softmax.cu` and latest progress/session context.
- M1 correctly returns a when b is empty, b when a is empty; both-empty remains empty.
- M2 sets max correctly but uses exp(new-old) for both groups, incorrectly enlarging contributions. Explained old-new rescaling with maxima 3 and 5.
- No source edits, compilation, GPU execution, or performance measurements. Placeholder comment is stale, but not a correctness blocker.

## Learning and handoff

- Next: learner corrects both exponent signs, then review helper before kernel implementation.
- Prior conceptual derivation was correct; translating it into code required this targeted correction. Independent implementation not established.
- Changes local/uncommitted; no remote activity, commit, or push.

## Corrected helper follow-up

- Re-read saved helper: both exponent differences now use old minus merged maximum. Empty handling, maximum and rescaled sum match the intended merge for valid finite-input summaries.
- No kernel edits, compiler/device tests, or new numerical results. Stale placeholder comment remains cosmetic.
- Next: learner fills online TODOs 1–2 (private state accumulation and warp pair reduction).

## Online kernel first source review

- User requested TODOs 1–2 review; inspected saved full online kernel, including later stages learner also filled.
- TODO 1 correctly accumulates singleton summaries. TODO 2 uses unconditional two-field shuffles and assigns only after both fields are fetched. Unguarded out-of-range lanes self-merge, but the lane-0 reduction tree remains correct; suggested valid-partner merge guard after both shuffles to match scaffold.
- Later stages have blocking issues: shared partials read before any stores, final merge references out-of-scope `d` and excludes named shuffle lanes, final pair is never published and broadcast to all threads. Output expression itself has the expected form but uses wrong/incomplete state.
- Next targeted task: TODO 3 stores each warp's lane-0 state into its own shared slot, followed by unconditional block synchronization before any shared reads.
- No kernel edits, compilation, numerical execution, or GPU activity. Source remains incomplete.

## Shared exchange and final merge follow-up

- Re-read saved online kernel. Per-warp lane-0 stores precede unconditional barrier; warp 0 loads summaries/empty padding, performs a full-warp offset loop and publishes final pair; unconditional barrier precedes all-thread read/output. Previous blocking issues corrected.
- Current duplicate load/padding stage and middle block barrier are redundant, not incorrect. Later `merge_online(state, empty)` leaves state unchanged and works only because earlier padding set it empty. Explained that cleanup should use direct empty assignment in retained loading block.
- Unguarded out-of-range shuffle self-merges remain harmless for the consumed lane-0 result in this tree. No remaining blocking source issue found for intended finite-input workloads/launch contract.
- Next: simplify redundant loading stage before online host validation. No code edits, compiler tests, GPU execution, or speedup claims.

## Cleanup verified

- Inspected saved online kernel: redundant loading stage/barrier removed; padding explicitly assigns empty pairs. Two required publication barriers remain, before shared summary reads and before final row-pair reads.
- No blocking source issue found. Guided implementation is source-reviewed only; no source edits, compilation or GPU execution during this check.
- Next: include online in the existing common-reference softmax host suite, keeping cases and tolerances fixed, before deferred device validation.

## Delegated online host integration

- User requested online harness integration. Added online label and launch branch, derived variant count from label array, and corrected stale skeleton/launch-status comments. Kernel arithmetic untouched.
- Ten existing cases now schedule 30 variant/case runs, each using unchanged CPU reference/criteria and its own NaN reset, checked launch/completion/download, labeled diagnostics and aggregate failure status.
- Inspected final dispatch and ran `git diff --check`: no whitespace errors. CPU reference code was unchanged and its previous tests were not rerun. No CUDA compilation or GPU execution.
- Next: deferred GPU correctness session before matched performance measurement. Changes local/uncommitted; no remote activity, commit or push.

## Checkpoint preparation and next direction

- User requested commit/push of all accumulated changes. Inspected Git status, branch/remotes, roadmap diff, new-file inventory and whitespace. Branch main, starting commit f6d6002; changes cover Project 3 kernels, host support, organized notes, progress and agreed roadmap expansion.
- Local Markdown-link audit passed: 28 targets across 17 relevant files. CPU reference tests were already run as recorded in prior sessions; no repeated CPU tests or CUDA execution for this Git-only checkpoint.
- Refreshed notes index and current progress. Suggested next offline lesson is register-retained softmax while preserving 128-thread mapping. GPU correctness/comparisons, backward and language ports remain future work before advancing through GEMM/attention.
- Commit/push requested and prepared; outcome is recorded by subsequent Git commands rather than assumed here. No completion/mastery or GPU-performance claim.
