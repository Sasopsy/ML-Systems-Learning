# 2026-10-15 — Memory utilization measurement

## Context

- Project 2, offline Nsight analysis; TUTOR.
- Target: distinguish instruction throughput, internal memory service, and transfer bandwidth.
- Environment: local macOS; no GPU execution. Starting commit not rechecked this turn.

## Work and evidence

- Read status and searched for local section files; none found in repository.
- Python CSV reads verified three summary values and matching component metrics in `progress/sessions/2026-10-01-project-02-offline-batch/ncu_on_035-raw.csv`.
- Consulted NVIDIA Profiling Guide for throughput aggregation, LSU, and L1TEX wavefront definitions. Full architecture-specific constituent membership not queried.
- Extended `project_02_performance_laboratory/notes/03-gpu-hardware-and-scheduling/17-dram-throughput.md` with formulas, counter mapping, limitations, and a hypothetical example.

## Experiment

- No new experiment. Saved counters are observations; matching values do not establish a complete constituent list. Toy rates are not measured H100 capacities.

## Learning and handoff

- Learner requested explanation; no new independent assessment.
- Next: learner explains whether existing data supports DRAM bandwidth saturation and what remains possible.
- Documentation changes remain local; no commit, push, or remote synchronization.

## Follow-up — Rate versus latency and wavefront size

- User asked whether handling means completion, and requested H100/B200 wavefront sizes and per-unit bytes/cycle.
- Checked official wavefront and shared-bank definitions; distinguished stage service rate from full load latency and avoided a universal wavefront byte size. No B200-specific peak measurement available.
- Re-read saved raw metrics and calculated an implied H100 aggregate DRAM peak of 1,280 bytes per DRAM-unit cycle. L1TEX and DRAM elapsed clocks differ; 100 cycles was only illustrative.
- Note 17 extended. Next resolve these measurement distinctions before requesting the pending memory diagnosis.

## Follow-up — Concrete shared-load wavefronts

- User requested a worked example. Derived bank mapping for a 32-lane scalar FP32 shared load: stride 1 needs one wavefront, stride 2 needs two due to two distinct words per even-numbered bank.
- Recorded the example in note 17, including illustrative arbitration order, unchanged 128 requested bytes, broadcast exception, and distinction from end-to-end latency. Checked CUDA shared-memory documentation; no GPU execution.
- Understanding level: explained/worked-through; independent transfer still pending. Next return to memory diagnosis once wavefront splitting is clear.

## Follow-up — Learner diagnosis and consolidated notes

- Learner answered no to DRAM saturation, citing elapsed-duration measurement and possible on-chip transfers. Refined the reason: 2.28% average DRAM utilization is evidence against sustained saturation, but does not establish where most time went.
- Learner requested the alternative cause; supplied dependency-latency/limited-independent-work hypothesis using saved Wait/no-issue counts and roughly one active warp/scheduler. This is assisted explanation, not independent diagnosis.
- On explicit request, consolidated covered concepts in note 17 with concise bullets, tables, formulas, examples, and a supported-conclusion/hypothesis distinction. Added note 16 cross-reference and updated README.
- Checks: reviewed note sections and verified local Markdown links. No code changes, new GPU experiments, commits, pushes, or remote synchronization.
- Next task: combine off/on kernel coverage and throughput evidence to explain the performance difference, without mixing captures/benchmarks or dropping the numerical-acceptance limitation.

## Follow-up — Start off/on synthesis

- User requested connecting completed memory lesson to off/on performance. Python CSV reads rechecked expanded off030/on035 grid/block, duration, occupancy, active warps/scheduler, issue-active percentage, and elapsed SM throughput.
- Presented a matched profiler comparison; kept it distinct from non-profiled timing runs. No new causal experiment, independent synthesis, or numerical acceptance claimed.
- Next: learner reconciles similar active-cycle occupancy with different launch coverage and elapsed utilization.

## Follow-up — Requested Q&A revision section

- Confirmed existing definitions, formulas, and examples, then added a concise question/answer section to note 17 matching the learner's questions about Mem Busy, Mem Pipes Busy, Max Bandwidth, measurement windows, throughput versus completion, wavefront sizes, and per-unit peak rates.
- Linked the section from the notes index. Preserved uncertainty about a complete H100/B200 capacity table.
- Documentation only; no new measurements or mastery claims. Next task remains the pending off/on GPU-coverage explanation.

## Follow-up — Off/on performance explanation

- User requested the speed-difference explanation after active-cycle denominator clarification. Read offline handoff and project benchmark records; no new GPU execution.
- Connected tile/grid changes to improved potential GPU coverage; similar achieved occupancy and lower active-cycle issue rate prevent claiming improved per-active-scheduler efficiency.
- Added note 18 and index link; distinguished profiler durations from six-process unprofiled 67.76275/27.65771 µs result. Retained variant identity and numerical-acceptance limitations.
- Understanding remains assisted; next task is a fresh transfer using saved 512-block variant. Local documentation only, no commit/push.

## Follow-up — 512-block variant comparison

- User requested comparison with 128-block kernel. Parsed all five relevant saved raw CSVs (on031/032, on035/036/037) using Python stdlib, checking identities, launch resources, occupancy, scheduler counters, elapsed throughput, and durations.
- Added measured comparison to note 18. Highlighted smaller 64-thread blocks and overlapping repeat durations, preserving variant identities.
- No new execution or isolated-variant benchmark. Next learner interprets why improved utilization does not guarantee proportional speedup; causal explanation of equal runtimes remains pending.
