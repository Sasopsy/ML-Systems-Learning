from pathlib import Path
import csv,json,hashlib,subprocess
p=Path('project_03_gpu_operator_library/results/2026-10-27-offline-profiles')
cases=json.loads((p/'selection.json').read_text())
metrics=[('Kernel','Kernel Name'),('Profiled duration','gpu__time_duration.sum'),('Executed warp instructions','smsp__inst_executed.sum'),('Global-load warp instructions','smsp__sass_inst_executed_op_global_ld.sum'),('Shared-load warp instructions','smsp__sass_inst_executed_op_shared_ld.sum'),('Shared-store warp instructions','smsp__sass_inst_executed_op_shared_st.sum'),('Registers/thread','launch__registers_per_thread'),('Achieved occupancy','sm__warps_active.avg.pct_of_peak_sustained_active'),('Branch efficiency','smsp__sass_average_branch_targets_threads_uniform.pct'),('Active threads/warp instruction','smsp__thread_inst_executed_per_inst_executed.ratio'),('Long scoreboard cycles/issued instruction','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),('Short scoreboard cycles/issued instruction','smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),('Barrier cycles/issued instruction','smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'),('Math-pipe throttle cycles/issued instruction','smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'),('Not-selected cycles/issued instruction','smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio'),('DRAM read bytes (scaled unit below)','dram__bytes_read.sum'),('DRAM write bytes (scaled unit below)','dram__bytes_write.sum'),('Local spilling requests','derived__local_spilling_requests')]
index='| Shape | Win category | Pair comparison |\n|---|---|---|\n'
flat=[]
for case in cases:
 shape=f"{case['rows']}x{case['cols']}";folder=p/shape
 data={};units={}
 for v in ['online_registers','pytorch']:
  capture=(folder/(v+'_capture.txt')).read_text()
  assert 'PASS '+v in capture and capture.rstrip().endswith('EXIT: 0'),(shape,v)
  assert (folder/(v+'.ncu-rep')).stat().st_size>0
  records=list(csv.DictReader((folder/(v+'_raw.csv')).open()))
  assert len(records)==2,(shape,v,len(records))
  units[v],data[v]=records
  # Export SASS so basic source inspection also works without the GUI.
  with (folder/(v+'_sass.txt')).open('w') as out:
   subprocess.run(['ncu','--import',str(folder/(v+'.ncu-rep')),'--page','source','--print-source','sass','--metrics','smsp__branch_targets_threads_divergent'],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=60)
 wins={mode:values['ours_faster'] for mode,values in case['timings'].items()}
 category=('Ours wins both modes' if all(wins.values()) else
           'PyTorch wins both modes' if not any(wins.values()) else
           'Ours wins direct; PyTorch wins graph' if wins['direct'] else
           'PyTorch wins direct; ours wins graph')
 index+=f'| {shape} | {category} | [Open comparison]({shape}/README.md) |\n'
 text=f'# {shape}: online registers versus PyTorch\n\n- Selection: **{category}**. This is based on prior unprofiled sweep medians, not these replay durations.\n\n| Timing mode | Online registers µs | PyTorch µs |\n|---|---:|---:|\n'
 for mode,v in case['timings'].items():text+=f"| {mode} | {v['online_registers']:.3f} | {v['pytorch']:.3f} |\n"
 text+='\n## Profile metrics\n\n| Metric | Export unit | Online registers | PyTorch |\n|---|---|---:|---:|\n'
 for label,key in metrics:
  a,b=data['online_registers'].get(key,'not collected'),data['pytorch'].get(key,'not collected')
  ua,ub=units['online_registers'].get(key,''),units['pytorch'].get(key,'')
  unit=ua if ua==ub else f'ours:{ua};torch:{ub}'
  text+=f'| {label} | {unit} | {a} | {b} |\n'
  flat.append([shape,label,key,ua,a,ub,b])
 text+='''
## Offline files

- [Our report](online_registers.ncu-rep), [PyTorch report](pytorch.ncu-rep).
- [Our readable details](online_registers_details.txt), [PyTorch readable details](pytorch_details.txt).
- [Our raw CSV](online_registers_raw.csv), [PyTorch raw CSV](pytorch_raw.csv).
- [Our SASS](online_registers_sass.txt), [PyTorch SASS](pytorch_sass.txt).
- `*_capture.txt` records commands, profiler warnings and independent reference-validation results.
- Profiler durations are diagnostic. Stall ratios are normalized per issued instruction, not wall-time percentages. Not-selected means another ready warp issued.
- Raw exports include units; byte/time values may be scaled. Do not assume all numeric columns use base units.
- PyTorch source-line mapping is unavailable in this binary. See separately bundled [pinned sources](../source/pytorch/); our source is embedded and also [bundled](../source/softmax.cu).
- [Bundle index and limitations](../README.md).
'''
 (folder/'README.md').write_text(text)
with (p/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['shape','label','metric','ours_unit','ours_value','pytorch_unit','pytorch_value']);w.writerows(flat)
(p/'README.md').write_text('''# Project 3 offline profiles: full sweep — 2026-10-27

**Download this whole folder. No GPU execution is required to inspect it.**

## Start here on the Mac

1. Start with [4096×1024](4096x1024/README.md), where PyTorch wins, then [4096×2048](4096x2048/README.md), the crossover, then [4096×4096](4096x4096/README.md) and [4096×8192](4096x8192/README.md).
2. Open the two `.ncu-rep` files in Nsight Compute. Reports were generated by **Nsight Compute2025.3.1.0**; use a compatible viewer. A newer viewer may be needed if the installed one rejects the report.
3. Select PyTorch's result and use **Add Baseline**, then select ours. Inspect **Instruction Statistics**, **Warp State Statistics**, **Memory Workload Analysis**, **Scheduler Statistics**, and **Occupancy**. The original report remains unchanged by comparison.
4. Inspect **Source → SASS** for generated instructions. PyTorch's installed binary lacks CUDA line information; pinned source files are supplied for separate reading. Our custom source was imported; source snapshots are included as a fallback.
5. If no compatible GUI is available, the pair READMEs, full details, raw CSV and SASS text exports contain the analysis data. This does not require CUDA on the Mac.

## Captured cases

- **36 full reports: two kernels ×18 shapes.** All sweep shapes are covered, including wins and losses; these observed medians are not statistical proof of universal superiority.
- Our retained variant won nine shapes in both modes; PyTorch won seven shapes in both modes. Our37×128 and256×128 won only with direct launches. Their graph timings favor PyTorch, so host-submission differences may explain the direct-only wins.

'''+index+'''
## Reproducibility and validation

- H10080GBHBM3, driver580.173.02, CUDA13.0.88, PyTorch2.11.0+cu130; source revision70d99e998b4955e0049d13a98d77ae1b14db1f45. Compiler flags and resources copied in `sweep_evidence/build.txt`.
- Same FP32 contiguous fixture as sweep, same row/width, output preallocated. One selected kernel after50 warmups; independent NumPyFP64 reference checked before/after. Fixed atol1e-7,rtol1e-5,row-sum tolerance1e-5, finite/range checks. **All36 captures exited0 and passed validation.**
- Full captures use kernel replay, `--profile-from-start off --launch-count 1 --set full --cache-control none --clock-control none --import-source yes`. Each command is preserved in its log; per-kernel subprocess timeout240s. Source snapshots and capture/sweep scripts included.
- **Uncontrolled caches/clocks:** chosen to retain warm-cache intent; replay may change cache conditions. Profiles are separate from timing evidence. Do not substitute replay duration for the nine-round sweep medians.
- Six CTC transfer metrics could not be collected, as recorded in logs; unavailable metrics are not zero. PyTorch CUDA source cannot be retroactively mapped without its compiled line information; SASS/metrics remain available.
- [Selection and timing medians](selection.json), [paired selected metrics](comparison.csv), [full sweep summaries](sweep_evidence/summary.csv), [raw sweep batches](sweep_evidence/rounds.csv).

## Tests and scope

| Evidence | Result | Limit |
|---|---|---|
| Existing expanded softmax suite |62 variant/case runs passed |New retained variant exercised at width1024 on eight fixtures |
| Existing memcheck/racecheck/synccheck/initcheck |All four passed |Expanded suite at that checkpoint; not a sanitizer sweep of every new width |
| Width sweep |54 initial comparisons and every post-batch check passed;972 timing samples |Synthetic fixture,18 shapes; two timing modes |
| Combined capture batches |36 independent pre/post reference checks passed |All18 sweep shapes, diagnostic profiler replay |
| Compiler reports |Zero spills across retained width specializations |Static compiler evidence, not proof for arbitrary widths/builds |

- `validation_evidence/` contains original correctness/sanitizer logs, labeled with their narrower scope. `sweep_evidence/` carries the independent-reference checks and timing metadata.
- Source snapshots are assisted implementations. Passed tests and profiles do not establish independent learner mastery. Project3 is not finished.
- Next offline task: compare4096×1024 (PyTorch wins) and4096×2048 (ours wins) instruction work, branch efficiency, load traffic and stall ratios; propose one controlled next experiment. No further GPU run is required to study these reports.
- Files are local/uncommitted/unpushed. No rental shutdown performed. Verify the download before shutting down the machine.
- `manifest.json` records file sizes and SHA256 checksums for transfer verification; it excludes itself.
''')
manifest=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='manifest.json':manifest.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Validated reports:',len(cases)*2,'files:',len(manifest),'bytes:',sum(x['bytes'] for x in manifest),flush=True)
