import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
from pathlib import Path
from torch.profiler import record_function
import argparse


def train_step(model, optimizer, loss_fn, input, targets):
    optimizer.zero_grad()
    outputs = model(input)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss


def train_loop(model, optimizer, loss_fn, input, targets, steps):
    for _ in range(steps):
        loss = train_step(model, optimizer, loss_fn, input, targets)
    return loss


def benchmark(model, optimizer, loss_fn, input, targets, steps):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    loss = train_loop(model, optimizer, loss_fn, input, targets, steps)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return loss, end_time - start_time


def profiled_train_step(model, optimizer, loss_fn, input, targets):
    with record_function("zero_grad"):
        optimizer.zero_grad()
    with record_function("forward_pass"):
        outputs = model(input)
    with record_function("loss"):
        loss = loss_fn(outputs, targets)
    with record_function("loss_backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudnn-benchmark", choices=["off", "on"], default="off")
    parser.add_argument("--cudnn-allow-tf32", choices=["off", "on"], default="off")
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This exercise requires a CUDA device.")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "on"
    torch.backends.cudnn.allow_tf32 = args.cudnn_allow_tf32 == "on"

    batch_size = 32
    image_size = 64
    num_classes = 10
    seed = 42
    torch.manual_seed(seed)
    model = models.resnet18(weights=None,num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    warmup_steps = 10
    steps = 50
    benchmark_runs = 5
    profiling_steps = 3

    input = torch.randn(batch_size,3,image_size,image_size).to(device)
    targets = torch.randint(0,num_classes, (batch_size,)).to(device)

    # warmup
    for _ in range(warmup_steps):
        _ = train_step(model, optimizer, loss_fn, input, targets)

    times = []

    for i in range(benchmark_runs):
        loss, duration = benchmark(model, optimizer, loss_fn, input, targets, steps)
        times.append(duration)       # in seconds

    for i in range(benchmark_runs):
        print(f"Average time (ms) per step for run {i}: {times[i] / steps * 1000}")
        print(f"Images per second for run {i}: {batch_size * steps / times[i]}")

    print(f"Final loss: {loss.item()}")

    # Mean, median and mode of times
    mean_time = sum(times) / len(times)
    median_time = sorted(times)[len(times) // 2]
    minimum_time = min(times)
    maximum_time = max(times)
    print(f"Mean time (ms) per step (mean of {benchmark_runs} runs): {mean_time * 1000/steps}")
    print(f"Median time (ms) per step (mean of {benchmark_runs} runs): {median_time * 1000/steps}")
    print(f"Minimum time (ms) per step (mean of {benchmark_runs} runs): {minimum_time * 1000/steps}")
    print(f"Maximum time (ms) per step (mean of {benchmark_runs} runs): {maximum_time * 1000/steps}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN allow TF32: {torch.backends.cudnn.allow_tf32}")
    print(f"run_id: {args.run_id}")


    # # Profiling

    # torch.cuda.synchronize()
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,
    #     torch.profiler.ProfilerActivity.CUDA,
    # ]
    # with torch.profiler.profile(activities=activities) as prof:
    #     for _ in range(profiling_steps):
    #         loss = train_step(model, optimizer, loss_fn, input, targets)
    #     torch.cuda.synchronize()
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # trace_path = Path(__file__).with_name("resnet18.trace.json")
    # prof.export_chrome_trace(str(trace_path))


    # Profiling with phases
    torch.cuda.synchronize()
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(profiling_steps):
            with record_function("train_step"):
                loss = profiled_train_step(
                    model, optimizer, loss_fn, input, targets
                )
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


    trace_path = Path(__file__).with_name(
        f"resnet18_benchmark_{args.cudnn_benchmark}_{args.cudnn_allow_tf32}_{args.run_id:03d}.trace.json")
    prof.export_chrome_trace(str(trace_path))


if __name__ == "__main__":
    main()
