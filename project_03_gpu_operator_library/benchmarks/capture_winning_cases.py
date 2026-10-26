"""Save paired offline NCU reports for every observed sweep win in either mode."""
import csv
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'project_03_gpu_operator_library'
OUT=PROJECT/'results/2026-10-27-offline-profiles'

def run(cmd,path,timeout):
    with path.open('w') as f:
        f.write('COMMAND: '+' '.join(map(str,cmd))+'\n');f.flush()
        result=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        f.write(f'\nEXIT: {result.returncode}\n')
    if result.returncode:raise RuntimeError(f'Failed: {path}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--all-cases",action="store_true",help="Include all sweep shapes, not only observed wins")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((PROJECT/'results/2026-10-27-softmax-sweep/summary.csv').open()))
    cases=[]
    for shape in sorted({(int(r['rows']),int(r['cols'])) for r in rows}):
        modes={}
        for mode in ['direct','graph']:
            values={v:float(next(r['median_us'] for r in rows if r['mode']==mode and (int(r['rows']),int(r['cols']))==shape and r['variant']==v)) for v in ['pytorch','online_registers']}
            modes[mode]=dict(**values,ours_faster=values['online_registers']<values['pytorch'])
        if args.all_cases or any(v['ours_faster'] for v in modes.values()):cases.append(dict(rows=shape[0],cols=shape[1],timings=modes))
    (OUT/'selection.json').write_text(json.dumps(cases,indent=2)+'\n')
    source=OUT/'source';source.mkdir(exist_ok=True)
    for name,original in [('softmax.cu',PROJECT/'kernels/cuda/softmax.cu'),('softmax_sweep_bridge.cu',PROJECT/'benchmarks/softmax_sweep_bridge.cu')]:
        if (source/name).exists() and (source/name).read_bytes()!=original.read_bytes():
            raise RuntimeError('Source changed; use a new capture folder rather than mix revisions')
    shutil.copyfile(PROJECT/'kernels/cuda/softmax.cu',source/'softmax.cu')
    shutil.copyfile(PROJECT/'kernels/cuda/softmax_test_support.h',source/'softmax_test_support.h')
    shutil.copyfile(PROJECT/'benchmarks/softmax_sweep_bridge.cu',source/'softmax_sweep_bridge.cu')
    run(['ncu','--version'],OUT/'ncu_version.txt',30)
    run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],OUT/'gpu.txt',30)
    run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],OUT/'compute_processes_before.txt',30)
    for i,case in enumerate(cases):
        shape=f"{case['rows']}x{case['cols']}";folder=OUT/shape;folder.mkdir(exist_ok=True)
        for variant in ['online_registers','pytorch']:
            report=folder/(variant+'.ncu-rep')
            if report.exists():
                log=folder/(variant+'_capture.txt')
                if not log.exists() or 'PASS '+variant not in log.read_text() or not log.read_text().rstrip().endswith('EXIT: 0'):
                    raise RuntimeError(f'Incomplete existing capture: {report}')
                if not all((folder/(variant+suffix)).exists() for suffix in ['_details.txt','_raw.csv']):
                    raise RuntimeError(f'Missing existing exports: {report}')
                print(f'{i+1}/{len(cases)} {shape} {variant}: retained verified existing report',flush=True)
                continue
            cmd=['ncu','--set','full','--profile-from-start','off','--launch-count','1','--cache-control','none','--clock-control','none','--import-source','yes','--source-folders',str(source)+','+str(source/'pytorch'),'-o',folder/variant,sys.executable,PROJECT/'benchmarks/profile_sweep_case.py','--rows',case['rows'],'--cols',case['cols'],'--variant',variant]
            run(cmd,folder/(variant+'_capture.txt'),240)
            for page in ['details','raw']:
                cmd=['ncu','--import',report,'--page',page]+(['--csv'] if page=='raw' else [])
                with (folder/(variant+'_'+page+('.csv' if page=='raw' else '.txt'))).open('w') as f:
                    subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
            print(f'{i+1}/{len(cases)} {shape} {variant}: captured, validated, exported',flush=True)
    print('PASS: all paired reports complete',flush=True)

if __name__=='__main__':main()
