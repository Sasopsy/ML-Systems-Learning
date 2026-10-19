# 2026-10-02 — Project 2 compute-throughput resumption

## Context

- **Project/milestone:** Project 2, offline interpretation of isolated forward-convolution counters.
- **Mode:** TUTOR. User requested continuation from compute throughput.
- **Learning target:** distinguish measured compute activity from warp residency, then connect to scheduler readiness.
- **Starting commit:** `e6def5a40598e69dd53b93b667f751337d34ea45`.
- **Environment:** local Mac, Darwin arm64, Python 3.9.6. Framework/GPU availability not probed.

## Work and evidence

- Read learning instructions, status, project record, latest GPU handoff, offline guides and Nsight revision note.
- Ran `python3 project_02_performance_laboratory/results/verify_download.py`: **PASS: 229 files verified.** This checks manifest-listed payload sizes and SHA256 hashes.
- Ran `git status --short` before changes: clean. Ran `git rev-parse HEAD`, `uname -sm`, and `python3 --version`.
- Saved evidence available locally under `project_02_performance_laboratory/results/`; original throughput values also recorded in `progress/sessions/2026-10-01-project-02-resource-accounting.json`.
- Updated progress records only. No workload source changes or GPU experiment.

## Learning and handoff

- **Learner conclusion:** not yet provided in this session.
- Previous explanations are recorded; no new independent understanding established.
- **Next task:** ask learner to interpret identical achieved occupancy with different compute throughput in off007/on008; use response to resume foundations before expanded off030/on035 Scheduler Statistics.
- Progress changes remain local and uncommitted. No remote synchronization or GPU-instance shutdown checked.

## Follow-up — Throughput foundations

- Learner requested explanation before interpretation: throughput, measurements of hardware units, and hardware controllers.
- Parsed original off007/on008 CSVs with Python standard-library `csv`; verified SM aggregate/issue activity 17.045637/57.768060%, FMA activity 12.856127/35.187090%, ALU activity 5.289765/31.182361%, LSU instruction rate 3.532645/17.338046%, tensor-MMA-specific activity 0/0, DRAM activity 0.940845/2.270934%.
- Consulted NVIDIA Nsight Compute Profiling Guide metric/pipeline definitions and Hopper architecture documentation. No GPU run or new capture.
- Added concise counter/controller definitions and toy count/rate/activity calculation to note 13. Toy values are illustrative, not measurements.
- **Understanding:** explanation requested; no learner conclusion or independent transfer yet.
- **Next task:** learner distinguishes operation throughput from active-cycle percentage in a fresh toy example, then connects instruction issue to scheduler readiness.

## Follow-up — Unused capacity and maximum aggregation

- Learner asked why throughput is about 57.5% rather than 100%, and why compute throughput is an aggregate.
- Used previously verified 57.768060% report value and NVIDIA definitions; no new report analysis or GPU run.
- Explained issue opportunities, ready warps, dependencies, resource availability, grid coverage and startup/tail effects as possible causes. No exact contribution assigned to any cause.
- Distinguished maximum across normalized constituent metrics from hardware-instance averaging; used hypothetical saturated arithmetic/unused tensor pipelines to explain why averaging can conceal a limiting resource.
- Updated note 13. Examples are illustrative; no independent mastery evidence.
- **Next task:** connect resident, eligible and issued warps to actual Scheduler Statistics in off030/on035.

## Follow-up — Deep-learning worked example

- User requested an in-depth DL example of unused capacity and aggregate throughput.
- Prepared a deliberately simple dense layer: X[32,128], W[128,64], Y[32,64], one scalar-FMA thread per output. Not a reconstruction of a library kernel.
- Toy two-warp/single-scheduler window: loads at cycles 0/1 become ready at 4/5; no issue at 2/3; FMAs issue at 4/5. Four of six issue opportunities used; two arithmetic instructions. Timing is invented and explicitly labeled.
- Connected dependency/readiness reasoning to coverage and maximum aggregation. Read conv_fprop.py to verify actual convolution input/weight/output shapes and 576 products per interior output.
- Consulted official NVIDIA profiling documentation; visualization skill used for a static dependency diagram in the explanation. Added note 14 and index entry; no learning-target source edits or GPU execution.
- **Understanding:** worked-through explanation only; no learner conclusion or transfer response yet.
- **Next task:** inspect resident/eligible/issued warps in expanded off030/on035 and request interpretation.

## Follow-up — First scheduler reading

- Learner reports understanding the throughput-normalization formula and requests next steps; self-report is not independent transfer evidence.
- Read saved expanded raw CSVs/logs with Python standard library, no GPU run. Evidence: `progress/sessions/2026-10-01-project-02-offline-batch/ncu_{off_030,on_035}{-raw.csv,.log}`.
- Off030: active warps/scheduler 1.007840, eligible 0.724008, displayed issued 0.72; active-cycle issue percentage 72.404340%, elapsed-cycle SM throughput 17.060174%.
- On035: active warps/scheduler 0.997023, eligible 0.651120, displayed issued 0.65; active-cycle issue percentage 65.112002%, elapsed-cycle SM throughput 57.580204%.
- Introductory task uses on035 only; avoid silently mixing these expanded reports with basic off007/on008. Explain resident/eligible/issued and fractional averages before learner interpretation.
- **Next task:** learner judges whether on035 first suggests investigating lack of ready warps or ready warps not being issued, with evidence and without claiming a specific stall cause.

## Follow-up — Scheduler graph calculations

- Learner selected lack of readiness and refined the description to not becoming ready in time for the issue opportunity. Correct immediate prompted interpretation after explanation; independent transfer remains unchecked.
- User attached a graph with maximum/theoretical/active/eligible/issued bars consistent with 16/3/1/0.65/0.65, requesting how each is calculated. Screenshot does not expose report identity.
- Re-read on035 raw CSV: maximum 16 warps/scheduler, 4 schedulers/SM, 64 warps/SM; theoretical 3 warps/scheduler; register/shared-memory/warp/block/barrier residency limits 3/7/16/32/32 blocks. Active 0.997023, eligible 0.651120; issue-active per-cycle displayed 0.65, issue-active percentage 65.112002.
- Consulted NVIDIA profiling and Hopper guides. Local search did not find SchedulerStats.section; no claim of inspecting installed section source.
- Added resource-limit derivation and toy cycle-count averaging example to note 13; clarified residency versus issue rate, active-cycle normalization, and unknown exact block placement.
- **Next task:** learner computes active/eligible/issued averages in a fresh short example, then inspect Warp State Statistics. No GPU run; progress/notes remain local.
