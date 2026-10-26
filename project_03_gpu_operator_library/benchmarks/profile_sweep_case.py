"""Capture one matched sweep kernel; validate independently before and after."""
import argparse
import ctypes
from pathlib import Path
import numpy as np
import torch

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,required=True)
    parser.add_argument('--cols',type=int,choices=[128,512,1024,2048,4096,8192],required=True)
    parser.add_argument('--variant',choices=['pytorch','online_registers'],required=True)
    args=parser.parse_args()
    if args.rows<=0:parser.error('rows must be positive')
    root=Path(__file__).resolve().parents[2]
    host=((np.arange(args.rows*args.cols,dtype=np.int64)%41-20)*0.25).astype(np.float32).reshape(args.rows,args.cols)
    ref=host.astype(np.float64)
    ref=np.exp(ref-ref.max(axis=1,keepdims=True));ref/=ref.sum(axis=1,keepdims=True)
    x=torch.from_numpy(host).cuda();y=torch.full_like(x,float('nan'))
    if args.variant=='pytorch':
        def launch():torch.softmax(x,dim=-1,out=y)
    else:
        lib=ctypes.CDLL(str(root/'artifacts/project03-2026-10-26/bin/softmax_sweep.so'))
        fn=lib.launch_registers
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
        xp,yp,stream=x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream
        def launch():
            code=fn(xp,yp,args.rows,args.cols,stream)
            if code:raise RuntimeError(f'CUDA launch error {code}')
    def validate():
        actual=y.cpu().numpy().astype(np.float64)
        assert np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
        assert (np.abs(actual-ref)<=1e-7+1e-5*np.abs(ref)).all()
        assert (np.abs(actual.sum(axis=1)-1)<=1e-5).all()
    launch();torch.cuda.synchronize();validate()
    for _ in range(50):launch()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    launch();torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    validate()
    print(f'PASS {args.variant} {args.rows}x{args.cols}: independent FP64 reference and invariants')

if __name__=='__main__':main()
