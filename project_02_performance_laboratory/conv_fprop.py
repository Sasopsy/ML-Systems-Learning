import torch
import torch.nn.functional as F
import time
import argparse
import json
from statistics import median
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudnn-benchmark", choices=["off", "on"], default="off")
    parser.add_argument("--cudnn-allow-tf32", choices=["off", "on"], default="off")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--mode", choices=["benchmark", "trace", "ncu"], default="benchmark")
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--save-tensors", type=Path)
    args = parser.parse_args()
    result_directory = Path(__file__).resolve().parent / "results" / "conv_fprop"
    run_name = f"{args.mode}_{args.cudnn_benchmark}_tf32_{args.cudnn_allow_tf32}_{args.run_id:03d}"
    if args.results_json is None:
        category = "benchmarks" if args.mode == "benchmark" else "diagnostics"
        args.results_json = result_directory / category / (run_name + ".json")
    for destination in (args.results_json, args.save_tensors):
        if destination is not None and destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)

    trace_path = result_directory / "traces" / (
        f"conv_fprop_benchmark_{args.cudnn_benchmark}_tf32_{args.cudnn_allow_tf32}"
        f"_{args.run_id:03d}.trace.json"
    )
    if args.mode == "trace" and trace_path.exists():
        raise FileExistsError(f"Trace already exists; choose a new run ID: {trace_path}")
    if args.mode == "trace":
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This exercise requires a CUDA device.")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float32
    benchmark_blocks = 5
    benchmark_repetitions = 100


    input_size = (32, 64, 16, 16)
    weight_size = (64, 64, 3, 3)
    conv_kwargs = dict(stride=1, padding=1, dilation=1, groups=1)

    # Configure cuDNN algorithm search and TF32 permission.
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "on"
    torch.backends.cudnn.allow_tf32 = args.cudnn_allow_tf32 == "on"
    torch.backends.cudnn.deterministic = False

    input = torch.randn(input_size, device=device, dtype=dtype)
    weight = torch.randn(weight_size, device=device, dtype=dtype)
    output = F.conv2d(input, weight, **conv_kwargs)

    # CPU reference implementation
    input_cpu = input.cpu().to(dtype=torch.float64)
    weight_cpu = weight.cpu().to(dtype=torch.float64)
    reference_output = torch.nn.functional.conv2d(input_cpu, weight_cpu, **conv_kwargs)

    # Warm up after reference computation, outside the measured blocks.
    for _ in range(10):
        output = F.conv2d(input, weight, **conv_kwargs)

    block_times_us = []
    if args.mode == "benchmark":
        for _ in range(benchmark_blocks):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_repetitions):
                output = F.conv2d(input, weight, **conv_kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            block_times_us.append(elapsed / benchmark_repetitions * 1e6)
    elif args.mode == "trace":
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(3):
                with torch.profiler.record_function("fprop_only"):
                    output = F.conv2d(input, weight, **conv_kwargs)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))
        print(f"Trace: {trace_path}")
    elif args.mode == "ncu":
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("conv_fprop_target"):
            output = F.conv2d(input, weight, **conv_kwargs)
        torch.cuda.synchronize()

    # Validate the final output outside timing/profiling.
    assert output.shape == (32, 64, 16, 16)
    assert output.dtype == dtype
    assert output.device == device
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(reference_output).all().item()

    # Compare in float64 while retaining our explicit float32 error criterion.
    output_cpu = output.cpu().to(dtype=torch.float64)
    absolute_error = (output_cpu - reference_output).abs()
    max_abs_error = absolute_error.max().item()
    atol, rtol = 1e-5, 1.3e-6
    failed = absolute_error > atol + rtol * reference_output.abs()
    failure_count = failed.sum().item()
    total_count = failed.numel()
    reference_norm = torch.linalg.vector_norm(reference_output).item()
    error_norm = torch.linalg.vector_norm(output_cpu - reference_output).item()
    relative_l2_error = error_norm / reference_norm if reference_norm else None

    print(f"Run ID: {args.run_id}")
    print(f"Mode: {args.mode}")
    print(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CUDNN allow_tf32: {torch.backends.cudnn.allow_tf32}")
    print(f"Maximum absolute error: {max_abs_error:.10g}")
    print(f"Relative L2 error: {relative_l2_error}")
    print(f"Criterion: atol={atol}, rtol={rtol}")
    print(
        f"Elements outside tolerance: {failure_count} / {total_count} "
        f"({100 * failure_count / total_count:.2f}%)"
    )
    print(f"Elementwise criterion: {'PASS' if failure_count == 0 else 'FAIL'}")
    # A diagnostic run completes even if the reported criterion fails.


    for i, time_us in enumerate(block_times_us, start=1):
        print(f"Block {i} time: {time_us:.2f} microseconds per call")
    if block_times_us:
        print(f"Median block time: {median(block_times_us):.2f} microseconds per call")

    if args.results_json is not None:
        result = dict(
            run_id=args.run_id, mode=args.mode, seed=42,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            input_shape=list(input.shape), weight_shape=list(weight.shape),
            input_stride=list(input.stride()), weight_stride=list(weight.stride()),
            convolution=conv_kwargs, dtype=str(dtype), device=str(device),
            warmup_calls=10, block_repetitions=benchmark_repetitions,
            block_times_us=block_times_us,
            median_us=median(block_times_us) if block_times_us else None,
            max_abs_error=max_abs_error, relative_l2_error=relative_l2_error,
            atol=atol, rtol=rtol, failure_count=failure_count,
            total_count=total_count, criterion_passed=failure_count == 0,
        )
        args.results_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_tensors is not None:
        torch.save(dict(input=input.cpu(), weight=weight.cpu(), output=output.cpu()), args.save_tensors)


if __name__ == "__main__":
    main()
