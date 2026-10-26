"""Inspect eager CUDA softmax dispatch; diagnostic captures, not benchmarks."""
import csv
import json
from pathlib import Path
import torch

@torch.no_grad()
def main():
    dest = Path(__file__).resolve().parents[1] / 'results/2026-10-27-pytorch-dispatch-survey'
    dest.mkdir(parents=True, exist_ok=True)
    # Common feature/vocabulary widths, with boundary neighbors.
    widths = [32, 64, 128, 129, 256, 512, 768, 1024, 2048, 2049, 4096, 8192, 9216, 10240, 16384, 32768, 50257]
    cases = [(256, width, dtype) for dtype in [torch.float32, torch.float16, torch.bfloat16] for width in widths]
    # Check whether row count affects the selected kernel for two representative widths.
    cases += [(rows, width, torch.float32) for rows in [1, 37, 4096] for width in [1024, 4096]]
    results = []
    for rows, cols, dtype in cases:
        x = torch.linspace(-5, 5, rows * cols, dtype=torch.float32, device='cuda').reshape(rows, cols).to(dtype)
        for _ in range(5):
            y = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            y = torch.softmax(x, dim=-1)
            torch.cuda.synchronize()
        tag = f'{str(dtype).split(".")[-1]}_{rows}x{cols}'
        path = dest / (tag + '.trace.json')
        prof.export_chrome_trace(str(path))
        kernels = [e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat') == 'kernel']
        assert kernels, tag
        assert torch.isfinite(y).all().item(), tag
        for e in kernels:
            a = e['args']
            result = dict(dtype=str(dtype), rows=rows, cols=cols, kernel=e['name'], grid=a.get('grid'), block=a.get('block'), registers_per_thread=a.get('registers per thread'), shared_memory_bytes=a.get('shared memory'), trace=path.name)
            results.append(result)
            print(tag, e['name'].split('(')[0], 'grid',result['grid'],'block',result['block'],flush=True)
    (dest/'dispatch.json').write_text(json.dumps(dict(torch=torch.__version__, revision=torch.version.git_version, device=torch.cuda.get_device_name(), results=results),indent=2)+'\n')
    with (dest/'dispatch.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)

if __name__ == '__main__':
    main()
