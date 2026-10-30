"""FP32 matched-workload sweep: eager/direct and CUDA Graph batches, separately."""
import argparse
from datetime import datetime, timezone
import ctypes
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'project_03_gpu_operator_library/results/2026-10-27-softmax-sweep'
DIMS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
ROWS = [37, 256, 4096]
REPEATS, ROUNDS, WARMUP = 500, 9, 50


def validate(y, reference):
    actual = y.cpu().numpy().astype(np.float64)
    error = np.abs(actual-reference)
    assert np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
    assert (error <= 1e-7 + 1e-5*np.abs(reference)).all(), float(error.max())
    assert (np.abs(actual.sum(axis=1)-1) <= 1e-5).all()
    return float(error.max())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dims',type=int,nargs='+',choices=DIMS,default=DIMS)
    parser.add_argument('--output',type=Path,default=DEST)
    parser.add_argument('--library',type=Path,default=ROOT/'artifacts/project03-2026-10-26/bin/softmax_sweep.so')
    args=parser.parse_args()
    selected_dims=args.dims
    dest=args.output
    dest.mkdir(parents=True, exist_ok=True)
    lib = ctypes.CDLL(str(args.library.resolve()))
    for name in ['launch_online','launch_registers']:
        fn = getattr(lib,name)
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        fn.restype=ctypes.c_int
    records=[]; checks=[]
    with (dest/'rounds.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['mode','variant','rows','dims','round','repeats','us_per_call'])
        for rows in ROWS:
            for dims in selected_dims:
                host=((np.arange(rows*dims,dtype=np.int64)%41-20)*0.25).astype(np.float32).reshape(rows,dims)
                reference=host.astype(np.float64)
                reference=np.exp(reference-reference.max(axis=1,keepdims=True))
                reference/=reference.sum(axis=1,keepdims=True)
                x=torch.from_numpy(host).cuda()
                names=['pytorch','online','online_registers']
                outputs=[torch.empty_like(x) for _ in names]
                ptr=x.data_ptr(); ptrs=[v.data_ptr() for v in outputs]
                def functions(stream):
                    def custom(fn, out):
                        code=fn(ptr,out,rows,dims,stream)
                        if code: raise RuntimeError(f'CUDA launch failed: {code}')
                    return [lambda:torch.softmax(x,dim=-1,out=outputs[0]),
                            lambda:custom(lib.launch_online,ptrs[1]),
                            lambda:custom(lib.launch_registers,ptrs[2])]
                direct=functions(torch.cuda.current_stream().cuda_stream)
                for name,fn,y in zip(names,direct,outputs):
                    y.fill_(float('nan')); fn();torch.cuda.synchronize()
                    checks.append(dict(rows=rows,dims=dims,variant=name,max_abs_error=validate(y,reference)))
                    for _ in range(WARMUP):fn()
                    torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
                start.record();stop.record();stop.synchronize()
                for mode in ['direct','graph']:
                    graphs=[]
                    if mode=='graph':
                        capture_stream=torch.cuda.Stream()
                        capture_stream.wait_stream(torch.cuda.current_stream())
                        funcs=functions(capture_stream.cuda_stream)
                        for fn in funcs:
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=capture_stream):
                                for _ in range(REPEATS):fn()
                            graphs.append(graph)
                        torch.cuda.current_stream().wait_stream(capture_stream)
                        for graph in graphs:graph.replay()
                        torch.cuda.synchronize()
                    for round_id in range(ROUNDS):
                        for k in range(3):
                            v=(k+round_id)%3
                            start.record()
                            if mode=='direct':
                                for _ in range(REPEATS):direct[v]()
                            else:graphs[v].replay()
                            stop.record();stop.synchronize()
                            us=start.elapsed_time(stop)*1000/REPEATS
                            validate(outputs[v],reference)
                            record=[mode,names[v],rows,dims,round_id,REPEATS,us]
                            writer.writerow(record);records.append(record)
                    f.flush()
                    del graphs
                print(f'PASS {rows}x{dims}: correctness and both timing modes',flush=True)
                del direct, outputs, x, reference, host
    summary=[]
    for mode in ['direct','graph']:
        for rows in ROWS:
            for dims in selected_dims:
                for variant in ['pytorch','online','online_registers']:
                    values=[r[-1] for r in records if r[:4]==[mode,variant,rows,dims]]
                    assert len(values)==ROUNDS
                    summary.append(dict(mode=mode,variant=variant,rows=rows,dims=dims,mean_us=statistics.mean(values),median_us=statistics.median(values),min_us=min(values),max_us=max(values)))
    with (dest/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (dest/'metadata.json').write_text(json.dumps(dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),kernel_template='softmax_online_registers<N, dims>',threads_per_block=128,torch=torch.__version__,revision=torch.version.git_version,device=torch.cuda.get_device_name(),dtype='float32',rows=ROWS,dims=selected_dims,repeats=REPEATS,rounds=ROUNDS,warmup=WARMUP,baseline='current softmax_online with specialized conditional update',checks=checks),indent=2)+'\n')

if __name__=='__main__':main()
