from pathlib import Path
import argparse,csv,json,hashlib,subprocess
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('project_03_gpu_operator_library/results/2026-10-27-offline-profiles'))
p=parser.parse_args().output
cases=json.loads((p/'selection.json').read_text())
for case in cases:
 case['dims']=case.get('dims',case.get('cols'))
metadata=json.loads((p/'sweep_evidence/metadata.json').read_text())
refreshed_date=metadata.get('refreshed_date', '2026-10-27')
ours_both=sum(all(v['ours_faster'] for v in c['timings'].values()) for c in cases)
torch_both=sum(not any(v['ours_faster'] for v in c['timings'].values()) for c in cases)
mixed=len(cases)-ours_both-torch_both
metrics=[('Kernel','Kernel Name'),('Profiled duration','gpu__time_duration.sum'),('Executed warp instructions','smsp__inst_executed.sum'),('Global-load warp instructions','smsp__sass_inst_executed_op_global_ld.sum'),('Shared-load warp instructions','smsp__sass_inst_executed_op_shared_ld.sum'),('Shared-store warp instructions','smsp__sass_inst_executed_op_shared_st.sum'),('Registers/thread','launch__registers_per_thread'),('Achieved occupancy','sm__warps_active.avg.pct_of_peak_sustained_active'),('Branch efficiency','smsp__sass_average_branch_targets_threads_uniform.pct'),('Active threads/warp instruction','smsp__thread_inst_executed_per_inst_executed.ratio'),('Long scoreboard cycles/issued instruction','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),('Short scoreboard cycles/issued instruction','smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),('Barrier cycles/issued instruction','smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'),('Math-pipe throttle cycles/issued instruction','smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'),('Not-selected cycles/issued instruction','smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio'),('DRAM read bytes (scaled unit below)','dram__bytes_read.sum'),('DRAM write bytes (scaled unit below)','dram__bytes_write.sum'),('Local spilling requests','derived__local_spilling_requests')]
index='| Shape (rows × dims) | Win category | Pair comparison |\n|---|---|---|\n'
flat=[]
for case in cases:
 shape=f"{case['rows']}x{case['dims']}";folder=p/shape
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
report_count=len(cases)*2
rounds=list(csv.DictReader((p/'sweep_evidence/rounds.csv').open()))
(p/'README.md').write_text(f'''# Project 3 offline softmax profiles — {refreshed_date}

**Download this whole folder. No GPU execution is required to inspect it.**

## Start on the Mac

1. Pick a shape from the table below. Dimensions are **rows × dims**; `N = 128` threads per block in our measured kernels.
2. Open the paired `.ncu-rep` files in a compatible Nsight Compute viewer. The capture version is recorded in `ncu_version.txt`.
3. Add PyTorch as a baseline and compare Instruction Statistics, Warp State Statistics, Memory Workload Analysis, Scheduler Statistics and Occupancy.
4. Source → SASS shows machine instructions. PyTorch's installed binary lacks CUDA line mapping; pinned sources are bundled separately. Our source is embedded and snapshotted.
5. Text/CSV/SASS exports and per-case tables provide a fallback without the GUI.

## Captured cases

- **{report_count} full reports: two kernels × {len(cases)} shapes.** All shapes in this sweep are covered, including losses.
- Median classifications: ours wins both modes for **{ours_both}** shapes; PyTorch wins both for **{torch_both}**; **{mixed}** change winner across direct/graph modes. These are observations, not statistical proof of universal superiority.
- Actual dims in this package: {', '.join(str(d) for d in sorted({int(c['dims']) for c in cases}))}.

'''+index+f'''
## Measurement and validation

- FP32 contiguous inputs, preallocated outputs, current `softmax_online_registers<128, dims>` and PyTorch softmax. The sweep additionally includes the current conditional online baseline.
- Sweep: {metadata['warmup']} warmups, {metadata['rounds']} rounds × {metadata['repeats']} calls, rotated order; **{len(metadata['checks'])} initial checks and all post-batch checks passed; {len(rounds)} samples**. Direct host launch paths differ; graph batches remove per-operation Python submission.
- Profile: one selected call after 50 warmups; independent NumPy FP64 pre/post reference and finite/range/row-sum checks. **All {report_count} captures passed.** Fixed atol 1e-7, rtol 1e-5, row-sum tolerance 1e-5.
- Full kernel replay, cache-control none and clock-control none. Caches/clocks are uncontrolled; profiler durations are diagnostic and must not replace unprofiled timing medians. Logs retain exact commands, per-capture timeout settings and unavailable-counter warnings. Missing metrics are not zeros.
- **Register retention is not guaranteed by the variant name.** Inspect `sweep_evidence/build.txt`, local-memory spilling requests and memory traffic for each dims value. Larger private arrays can spill.
- Device: {metadata['device']}; PyTorch {metadata['torch']}, revision {metadata['revision']}. Compiler flags/resources are in `sweep_evidence/build.txt`; source snapshots are in `source/`.
- [Selection/timings](selection.json), [selected metrics](comparison.csv), [sweep summaries](sweep_evidence/summary.csv), [raw batches](sweep_evidence/rounds.csv), [metadata/checks](sweep_evidence/metadata.json).
- Any `validation_evidence/` logs describe their own fixture scope. Reference checks in this sweep do not by themselves establish sanitizer coverage for every shape.
- Source and measurements are assisted work, not independent mastery. Results are ignored by Git and require separate download. No shutdown performed.
- `manifest.json` records every other file's size and SHA256 for transfer verification.
''')
manifest=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='manifest.json':manifest.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Validated reports:',len(cases)*2,'files:',len(manifest),'bytes:',sum(x['bytes'] for x in manifest),flush=True)
