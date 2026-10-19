# 2026-10-09 — Latency hiding in a dependent arithmetic chain

## Context

- **Project/milestone:** Project 2, offline scheduler and warp-state interpretation.
- **Mode:** TUTOR; fuller conceptual explanation requested.
- **Learning target:** distinguish a warp's dependency latency from aggregate instruction throughput and scheduler idleness.
- **Starting commit:** `e6def5a40598e69dd53b93b667f751337d34ea45`.
- **Environment:** continuing Mac workspace; versions/GPU availability not re-probed.

## Work and evidence

- Read learning instructions, current status/project sections and prior issued/executed session; ran `git status --short` and `git rev-parse HEAD`. Existing local changes preserved.
- Consulted NVIDIA CUDA execution-model and best-practices documentation for resident warp state, independent work and latency hiding.
- Added note 16/index entry and updated progress. No learning-target code, GPU run, or new benchmark.
- Example is invented: single issue/cycle, four-cycle arithmetic result latency, one versus four independent accumulator chains, pipelined acceptance of one instruction/cycle. Steady-state issue capacity illustration, not a kernel speedup measurement.

## Learning and handoff

- Learner previously answered yes to another warp filling a wait; conditions were supplied by assistant. More detail now requested; no independent transfer established.
- Explain that original latency persists while independent work fills otherwise unused cycles; occupancy alone does not ensure readiness.
- **Next task:** learner writes cycles 0–3 for two independent warps under the toy assumptions, then compare all and Not-Issued stall samples in on035.
- Notes/progress remain local; no commit, push, or remote sync verification.

## Follow-up — Remaining-scope review

- User asked how much and what remains before proceeding. Read Project 2 roadmap (including exit criterion), project history, whole-model repeat measurements, and phase/kernel attribution records.
- Distinguished completed evidence collection from interpretation and independent transfer. Existing repeated CNN comparison/isolated workloads/counter captures remain valid evidence; accuracy acceptance remains unresolved.
- Proposed five closing chunks for current convolution case: scheduler/stalls; memory behavior; combined compute/coverage/off-on comparison; supported conclusion including numerical limitations; fresh alternate-report interpretation.
- Full Project 2 outstanding breadth: reusable harness and additional timing/tool coverage, memory accounting, deliberately bad workloads, precision/quality experiments, transformer encoder and decoder/LLM metrics, and unfamiliar-workload diagnosis. SSM/diffusion/world-model workloads are later additions in the roadmap.
- No numerical completion percentage, fixed duration, new required scope, or roadmap edit. Current saved-report interpretation can proceed offline; broader experiments need suitable hardware.
- **Next task:** resume latency-hiding transfer and bounded scheduler/stall comparison after answering the scope question.

## Follow-up — No-issue-cycle sample comparison

- User accepted the bounded scope recommendation and asked to proceed.
- Re-read on035 raw CSV using Python: Wait all samples 382, Not-Issued 376; Dispatch Stall all 223, Not-Issued 5. Checked NVIDIA's documentation for the no-issue-cycle filter.
- Presented counts separately from previously discussed normalized ratios. No inference of exact lost cycles or removable runtime.
- Added comparison to note 16. No GPU run or source change.
- **Next task:** learner explains which of these two states to investigate first and what Dispatch Stall's count change suggests; then move to memory behavior. Interpretation remains pending.

## Follow-up — Locate counters and recap labels

- User could not find warp-state counters and requested Wait/Dispatch explanations again. No interpretation assumed.
- Checked NVIDIA Nsight Compute UI manual: Raw page lists collected metrics and supports filtering; Profile > Metric Details provides descriptions/raw values and prefix-based search.
- Provided exact on035 PC-sampling metric prefixes and distinguished default Warp State Statistics graph from CSV-derived sample comparison. Earlier screenshot showed a collapsed Warp State Statistics section; no current live UI inspected.
- Recalled Wait as fixed-latency dependency and Dispatch Stall as ready instruction held back at dispatch. No new experiment or claim of specific dispatch cause.
- Next: locate the all/Not-Issued pair, then interpret the existing counts. Notes/progress remain local.

## Follow-up — Meaning of dispatch and filtering

- User asked why dispatch can hold a ready instruction and what base/Not-Issued mean.
- Rechecked NVIDIA definitions. Public category identifies dispatch held back by conflicts/events, without identifying this kernel's exact internal conflict. No unsupported claim about a specific busy pipeline/queue.
- Clarified base as informal unfiltered metric name, and Not-Issued as no instruction issued by the scheduler on the sampled cycle. Used a toy waiting-warp/other-warp example to distinguish both counts.
- No new counter capture or independent learner assessment. Next remains interpretation of sample association with scheduler idleness.

## Follow-up — Concrete dispatch mechanism found

- Learner described Wait as instruction dependencies and asked for concrete dispatch causes. Clarified fixed-latency dependency scope.
- Found NVIDIA engineer confirmation of register-file read/write bandwidth limiting IMAD/IADD3 dispatch on RTX 3080, and separate engineer explanation of register-bank contention during operand collection.
- Explained produced-value availability versus physical register access, using illustrative `R0 = R1 * R2 + R3`; distinguished register allocation capacity from access bandwidth.
- Updated note 16 with links. This improves the earlier vague dispatch explanation; exact H100 cause remains unproven. No GPU experiment or independent transfer assessment.

## Follow-up — Bank-conflict classification boundary

- Learner remained confused by register-bank contention versus shared-memory bank conflicts receiving different labels.
- Checked NVIDIA operand-collection explanation and Short Scoreboard/shared-memory bank-conflict definitions.
- Traced two illustrative paths: register operand reads before dispatch of arithmetic versus shared-bank conflict while servicing an already-dispatched load, then its consumer waiting on a scoreboard dependency.
- Clarified physical root cause is distinct from observed warp state; labels are not a one-to-one classification of physical mechanisms. No causal claim for the saved kernel.
- Added distinction to note 16. Learner interpretation pending; next confirm this distinction before proceeding to memory-counter study.

## Follow-up — Operand collection and definition scope

- Learner requested a plain explanation of operand collection and whether Dispatch Stall is exclusively register-related. Explained reading stored input numbers for delivery to arithmetic hardware.
- Rechecked NVIDIA Profiling Guide and linked engineer explanation: broad public definition does not specify registers only; register contention is a documented example. Exact H100 cause remains unknown.
- Updated note 16. No GPU experiment, code change, or independent mastery evidence. Next resolve the concept before returning to stall-count interpretation.

## Follow-up — DRAM throughput measurement

- Learner selected Wait due to many stalls; refined interpretation using no-issue samples, so this remains assisted reasoning.
- Screenshot showed Memory Throughput 76.28 GB/s, not an explicitly labeled DRAM percentage. Verified raw on035 counters locally with Python CSV parsing: 2.299392 MB read, zero recorded writes, 30.144 µs, 76.280255 GB/s, 2.279788% of peak sustained elapsed.
- Explained absolute byte rate versus capacity over elapsed memory-unit cycles; checked NVIDIA normalization explanation. Added note 17 and index link.
- No new GPU execution or remote synchronization. Next explain L1/L2 cache-hit denominators after throughput normalization is clear.

## Follow-up — Shared memory and L1 coexistence

- Learner recognized that low DRAM throughput need not be bad, then requested cache and SRAM clarification. Explained sector-based L1/L2 hit rates with different denominators.
- Checked NVIDIA Hopper guide: combined 256 KiB L1/texture/shared resource and supported per-SM carveouts. Distinguished kernel per-block allocations from per-SM capacity reservation.
- Added illustrative 32×32 FP32 dense-layer tiles (8 KiB user shared allocation/block) and direct-versus-tiled convolution examples to note 17. Examples are teaching material, not measured cuDNN implementation claims.
- No GPU runs or independent transfer check. Next resolve shared allocation versus automatic caching, then resume memory-analysis conclusion.

## Follow-up — Memory utilization summaries

- Learner requested moving on after L1/shared-memory explanation. Introduced screenshot values Mem Busy 41.91%, Max Bandwidth 30.28%, Mem Pipes Busy 17.28%, using official section definitions.
- Updated note 17; next learner assesses whether DRAM-bandwidth-bound is supported. No new GPU experiment or independent conclusion recorded.

## Follow-up — L2 and TEX hardware names

- Checked official Nsight hardware names and Hopper architecture documentation; clarified LTC versus LTS and TEX versus Tensor Cores.
- Read saved on035 device attributes with Python CSV parsing: H100 80GB HBM3, 132 SMs, 52,428,800-byte L2. Explained aggregate GPU-wide sharing with physically partitioned cache hardware, without inventing a fixed SM-to-slice ratio.
- Updated note 17. Next return to pending bounded memory-bandwidth conclusion; no remote action or new experiment.
