# 2026-09-29 — Numerical tolerances

## Context

- Project: 2, isolated dgrad accuracy investigation; mode TUTOR.
- Learning target: accumulation order and absolute/relative tolerance meaning.
- Starting commit: b2700b3d79df51b390bd5021caf158fc611e018e.
- Existing workspace; hardware/framework not rechecked this lesson.

## Work and evidence

- Read learning instructions, latest progress, note 11, and session template;
  ran `git rev-parse HEAD`. Consulted official PyTorch 2.11 testing documentation.
- Added short tolerance recap to project_02_performance_laboratory/notes/11-isolating-convolution-input-gradients.md.
- No learning-code edits, new numerical tests, or GPU runs. Earlier float64
  comparison remains in progress/sessions/2026-09-27-project-02-cudnn-repeat-runs/float64-reference.json.

## Learning and handoff

- Learner correctly answered 1/0 for two addition groupings after rounding
  behavior was provided. Correctly reasoned the near-zero limit of tolerance;
  decimal conversion corrected: 1e-5 is 0.00001, not 0.00005.
- Explained rtol=1.3e-6 is a PyTorch default, not a workload-derived guarantee.
  Default criterion failure is not proof of unusable training gradients.
- Next: measure relative L2 error and examine failed-element reference magnitudes
  using saved outputs/float64 reference, without changing acceptance tolerances.
- Notes/progress changes local; no commit, push, or remote sync verification.


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
