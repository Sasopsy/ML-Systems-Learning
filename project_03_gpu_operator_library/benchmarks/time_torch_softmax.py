"""Eager PyTorch comparison; CUDA-event intervals include Python submission gaps."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch


def validate(y, expected):
    actual = y.cpu().numpy().astype(np.float64)
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    error = np.abs(actual - expected)
    assert (error <= 1e-7 + 1e-5 * np.abs(expected)).all()
    assert (np.abs(actual.sum(axis=1) - 1) <= 1e-5).all()
    return float(error.max())


@torch.no_grad()
def main():
    destination = Path(__file__).resolve().parents[1] / "results/2026-10-27-pytorch-softmax"
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    errors = {}
    for rows, cols in [(37, 130), (4096, 1024)]:
        host = ((np.arange(rows * cols, dtype=np.int64) % 41 - 20) * 0.25).astype(np.float32).reshape(rows, cols)
        ref = host.astype(np.float64)
        ref = np.exp(ref - ref.max(axis=1, keepdims=True))
        ref /= ref.sum(axis=1, keepdims=True)
        x = torch.from_numpy(host).cuda()
        out = torch.empty_like(x)
        functions = [lambda: torch.softmax(x, dim=-1), lambda: torch.softmax(x, dim=-1, out=out)]
        names = ["eager_default", "eager_preallocated"]
        for name, fn in zip(names, functions):
            errors[f"{rows}x{cols}/{name}"] = validate(fn(), ref)
            for _ in range(50):
                y = fn()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        # Initialize events outside measured batches.
        start.record()
        stop.record()
        stop.synchronize()
        for round_id in range(9):
            for k in range(2):
                v = (k + round_id) % 2
                fn = functions[v]
                start.record()
                for _ in range(500):
                    y = fn()
                stop.record()
                stop.synchronize()
                us = start.elapsed_time(stop) * 1000 / 500
                validate(y, ref)
                samples.append([names[v], rows, cols, round_id, 500, us])
    with (destination / "rounds.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "rows", "cols", "round", "repeats", "us_per_call"])
        writer.writerows(samples)
    summary = []
    for rows, cols in [(37, 130), (4096, 1024)]:
        for name in names:
            values = [s[-1] for s in samples if s[:3] == [name, rows, cols]]
            summary.append(dict(variant=name, rows=rows, cols=cols, median_us=statistics.median(values), min_us=min(values), max_us=max(values)))
    metadata = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), dtype="float32", warmup=50, rounds=9, repeats=500, max_abs_errors=errors, summary=summary)
    (destination / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
