# 2026-10-01 — Project 2 offline handoff

## Context

- Project2; TUTOR with explicitly delegated capture/support work. Starting commit71ec783; branchmain.
- User requested completion of the planned runs, reports and session records before stopping the rented GPU and studying offline.
- H10080GBHBM3,132SMs, driver580.173.02; environment and Nsight version rechecked in the batch evidence directory.
- No inference of independent mastery from assistant-run experiments.

## Work and evidence

- Added optional `--results-json` and `--save-tensors` outputs to conv_fprop.py; refuse existing destinations. Added relativeL2 against existing CPUfloat64reference; no tolerance changes. Diagnostics and serialization stay outside measured/captured regions.
- Six fresh benchmark processes, run20off/21on/22on/23off/24off/25on;10warmup,5blocks x100calls; synchronized block walltime.
- Off process medians68.170,67.728,67.763us; on27.658,27.624,27.853us (rounded here; JSON preserves full precision).
- Medianofmedians67.7627500044764/27.657710015773773us;59.184492934618646%reduction. Synthetic operation-call result, not training-step timing.
- Expanded capture sections: LaunchStats,Occupancy,SpeedOfLight,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Chart,MemoryWorkloadAnalysis_Tables,SchedulerStats,WarpStateStats,InstructionStats,SourceCounters,WorkloadDistribution.
- Eight fresh expanded captures, each one selected kernel,36replaypasses, kernelreplay/cache-controlall/clock-controlnone, bounded by300stimeout. All executed commands, logs and return codes saved in batch summary.
- Off030/033/034: tile256x64x8,32blocks,255regs/thread, durations73.984/73.824/74.176us.
- On035/036/037: tile128x32x8,128blocks,166regs/thread, durations30.144/29.920/29.920us.
- On031/032: alternate tile32x32x8,512blocks,92regs/thread, duration29.920us each. First alternate caused identity guard to stop the batch; resumed while preserving all outputs and explicitly labeling variants. Cause of search variability not established.
- One binary report per capture imported successfully and its CSV contained exactly one kernel result. Each log lists all12sections. Commands/source did not force a particular cuDNN algorithm.
- All final numerical diagnostic results are recorded individually. Saved benchmark off020/on021 tensors are bitwise equal for input,weight,output. Snapshot equality check used CPU only. Existing criterion fails4736/524288; maxabs0.0001594768301; relativeL2approximately4.1820152e-7 for this pair. Completion/exit0 is not accuracy acceptance.
- GPU preflight and postflight listed no competing/remaining compute processes; no driver or clock configuration changes.

## Evidence locations

- `progress/sessions/2026-10-01-project-02-offline-batch/summary.json`: exact commands, full benchmark results, kernel identity and report hashes.
- Same directory: environment, text profiler logs, all rawCSVcounterexports, per-run accuracyJSON, snapshot-equalityJSON.
- `artifacts/project02-offline/`: eight expanded binary reports and two benchmark tensor snapshots (ignored by Git).
- Earlier report pair and24timeline traces plus prior dgrad snapshots remain in project02folder; included in offline package.
- `OFFLINE_PROJECT02.md`: report index, reading order, limitations and next-session prompt.
- Offline tar archive includes repository working files and ignored Project2 artifacts; no Git metadata. It preserves currently uncommitted code/notes as a portable snapshot.

## Learning and handoff

- Learner has read launch resources and independently answered the immediate occupancy arithmetic check after explanation. Asked for foundational throughput/counter clarification; explanations and pipeline readings recorded in note13.
- Scheduler eligibility, detailed memory interpretation and independent performance diagnosis remain to be taught/tested. Full Project2 and independent exit test remain incomplete.
- **Next concrete task:** locally open expanded off030/on035 and explain resident versus eligible versus issued warps using Scheduler Statistics; ask learner interpretation before giving a diagnosis.
- No further GPU execution is required to read this captured milestone. A new hypothesis might require a future experiment; do not claim all future Project2 GPU work is finished.
- No commit, push, cloud shutdown, or Mac transfer performed. Verify archive transfer/extraction before stopping rental. Archive creation/integrity verification recorded below when completed.


## Archive validation

- Built `artifacts/project02-offline-2026-10-01.tar.gz` from current repository working files and ignored Project2 artifacts, with an embedded per-file SHA256 manifest.
- Validated every archive payload file against its manifest; snapshot includes 10 Nsight reports, 24 timeline traces and 4 tensor snapshots. Rebuilt/revalidated after adding this handoff entry.
- External SHA256 sidecar accompanies archive. Transfer to Mac and cloud shutdown remain unperformed/unverified. All source/progress changes remain uncommitted, but are preserved in this snapshot.


## Follow-up — Organized folder handoff replaces archive workflow

- User requested ordinary folders/subfolders and updated Python output/input paths. Moved38original artifacts with SHA256 unchanged:10Nsightreports,24timelines,4tensors.
- New canonical data root: project_02_performance_laboratory/results/. Workload folders resnet18,conv_dgrad,conv_fprop; traces/tensors/benchmarks separated. Nsight basic versus expanded reports separated, with expanded searchmode/tile/run subfolders.
- benchmark.py and conv_dgrad.py write under their workload result folders; check_dgrad_accuracy.py reads relocated dgrad tensors. conv_fprop.py writes traces under results/conv_fprop/traces, defaults machine-readable benchmark/diagnostic JSON to results/conv_fprop, and creates parents for explicit result/tensor destinations. Nsight binary output is still controlled by ncu --export, not the Python script.
- Historical executed commands/evidence records preserved; progress/sessions/2026-10-01-project-02-result-layout.json resolves original artifact paths. Root OFFLINE_PROJECT02.md updated to current report locations and recursive folder download.
- results/ includes README, copied notes/source/session/instruction context, original binary reports, text logs and CSV exports. Whole projectfolder is recommended for executable scripts; resultsfolder alone is sufficient for report study.
- Verified syntax/import safety of all4Pythonfiles and ran check_dgrad_accuracy on CPU: relocated snapshots load and reproduce4821/9384mismatches. No GPU experiments rerun.
- Results folder remains Gitignored; it must be copied separately. Prior tar retained as an earlier backup, not current handoff. No commit/push, Mac transfer or rental shutdown performed.
- Next session: open results/README.md, then original expanded off030/on035 Scheduler Statistics. Full project/accuracy acceptance remain incomplete.


### Folder verification completed

- Results folder contains228payload files (~71.9MiB), including10binary Nsight reports,24timeline traces and4tensor snapshots. Standard-library `verify_download.py` passed all228SHA256/size checks.
- All38original artifacts retain their pre-move hashes. Root offline guide and resultsREADME local links resolve. Current source and session snapshots included for local continuation.
- Download project_02_performance_laboratory recursively; read results/README.md. Only results/ is also self-contained for report study. No tar extraction is needed.


### Local commit requested

- User authorized committing all completed code, evidence, notes and handoff changes; results/ remains ignored and will be downloaded separately.
- Only the results folder is needed for offline study: README, reports, copied source/instructions/notes/session context, and the standard-library verifier are bundled there.
- Validation already passed: four Python syntax/import checks, CPU dgrad reference using relocated tensors, original artifact hash preservation, folder manifest verification and whitespace check. No additional GPU runs needed.
- No push or shutdown requested in this action. The results snapshot will record the completed commit identifier after commit succeeds.
