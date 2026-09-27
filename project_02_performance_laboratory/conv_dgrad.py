"""Check an isolated convolution input gradient against PyTorch autograd."""

import torch
import torch.nn.functional as F
from torch.profiler import record_function
from pathlib import Path
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudnn-benchmark", choices=["off", "on"], default="off")
    parser.add_argument("--cudnn-allow-tf32", choices=["off", "on"], default="off")
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This exercise requires a CUDA device.")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float32
    input_size = (32, 512, 2, 2)
    weight_size = (512, 512, 3, 3)
    warmup_steps = 10
    conv_kwargs = dict(stride=1, padding=1, dilation=1, groups=1)
    profiling_steps = 3
    benchmark_blocks = 5
    benchmark_repetitions = 100

    # Set the CUDNN backend to use the best algorithm for the operation
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "on"
    # Disable TF32 for the operation
    torch.backends.cudnn.allow_tf32 = args.cudnn_allow_tf32 == "on"

    weight = torch.randn(weight_size, device=device, dtype=dtype)
    grad_output = torch.randn(input_size, device=device, dtype=dtype)

    # dX needs W, dY, and the input shape, but not the original input values.
    for _ in range(warmup_steps):
        grad_input = torch.nn.grad.conv2d_input(
            input_size, weight, grad_output, **conv_kwargs
        )

    # Reference through forward convolution and autograd, using the same W/dY.
    x = torch.randn(input_size, device=device, dtype=dtype, requires_grad=True)
    y = F.conv2d(x, weight, **conv_kwargs)
    assert y.shape == grad_output.shape
    (grad_input_reference,) = torch.autograd.grad(
        y, x, grad_outputs=grad_output
    )

    for name, result in (
        ("isolated dX", grad_input),
        ("autograd dX", grad_input_reference),
    ):
        assert tuple(result.shape) == input_size, f"Unexpected {name} shape"
        assert result.dtype == dtype, f"Unexpected {name} dtype"
        assert result.device == device, f"Unexpected {name} device"
        assert torch.isfinite(result).all().item(), f"Non-finite values in {name}"

    # Both routes can use the same backend: this checks API consistency.
    torch.testing.assert_close(grad_input, grad_input_reference)
    max_abs_error = (grad_input - grad_input_reference).abs().max().item()

    print(f"dX shape: {tuple(grad_input.shape)}")
    print(f"dtype: {grad_input.dtype}, device: {grad_input.device}")
    print(f"Maximum absolute error: {max_abs_error:.8g}")
    print("PASS: finite input gradients match the autograd reference.")
    print(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CUDNN allow_tf32: {torch.backends.cudnn.allow_tf32}")
    print(f"Run ID: {args.run_id}")

    # Benchmark the operation
    # Note that time is measured before and after the whole sequence of operation
    # and not just the operation itself.
    # If each operation was measured separately, it would include the time taken to synchronize the device
    times = []
    for i in range(benchmark_blocks):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for j in range(benchmark_repetitions):
            grad_input = torch.nn.grad.conv2d_input(
                input_size, weight, grad_output, **conv_kwargs
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
    
    for i, elapsed in enumerate(times):
        print(f"Time taken per call in block {i}: {elapsed/benchmark_repetitions * 1_000_000:.2f} microseconds")

    # Timed calls overwrite grad_input. Check that result, outside the interval.
    torch.testing.assert_close(grad_input, grad_input_reference)
    timed_max_abs_error = (grad_input - grad_input_reference).abs().max().item()
    print(f"Timed-call maximum absolute error: {timed_max_abs_error:.8g}")

    benchmark_label = "on" if args.cudnn_benchmark == "on" else "off"
    tf32_label = "tf32" if args.cudnn_allow_tf32 == "on" else "no_tf32"
    tensor_path = Path(__file__).with_name(
        f"conv_dgrad_{benchmark_label}_{tf32_label}_{args.run_id:03d}.pt"
    )
    torch.save(
        {
            "weight": weight.detach().cpu(),
            "grad_output": grad_output.detach().cpu(),
            "grad_input": grad_input.detach().cpu(),
        },
        tensor_path,
    )
    print(f"Saved tensors: {tensor_path.name}")

    # Profiling the operation
    torch.cuda.synchronize()
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(profiling_steps):   
            with record_function("dgrad_only"):
                grad_input = torch.nn.grad.conv2d_input(
                input_size, weight, grad_output, **conv_kwargs
            )
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)) 
    if torch.backends.cudnn.benchmark:
        if torch.backends.cudnn.allow_tf32:
            trace_path = Path(__file__).with_name(f"conv_dgrad_benchmark_on_tf32_{args.run_id:03d}.trace.json")
        else:
            trace_path = Path(__file__).with_name(f"conv_dgrad_benchmark_on_no_tf32_{args.run_id:03d}.trace.json")
    else:
        if torch.backends.cudnn.allow_tf32:
            trace_path = Path(__file__).with_name(f"conv_dgrad_benchmark_off_tf32_{args.run_id:03d}.trace.json")
        else:
            trace_path = Path(__file__).with_name(f"conv_dgrad_benchmark_off_no_tf32_{args.run_id:03d}.trace.json")
    
    prof.export_chrome_trace(str(trace_path))

if __name__ == "__main__":
    main()
