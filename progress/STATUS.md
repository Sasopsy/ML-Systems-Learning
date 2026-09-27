# Current status

Last updated: 2026-09-27 (UTC; first visual trace interpretation).

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
