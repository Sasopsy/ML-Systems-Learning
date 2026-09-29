# ML Systems / Infrastructure Mastery Roadmap

## Goal

Become an **architecture-literate ML systems engineer** who can take almost any modern deep-learning architecture and reason through:

```text
Model mathematics
    ↓
Tensor computation
    ↓
Autograd / training
    ↓
GPU execution
    ↓
Compilation
    ↓
Distributed execution
    ↓
Inference runtime
    ↓
Deployment infrastructure
```

You should eventually be able to pick up something unfamiliar—an SSM, new attention variant, diffusion world model, MoE architecture, or new recurrent model—and independently determine:

* how it trains;
* what state it maintains;
* what its bottlenecks are;
* how it maps onto GPUs;
* how it should be distributed;
* how inference differs from training;
* whether compilation or custom kernels help;
* what assumptions existing frameworks make that no longer hold;
* and how to deploy it reliably.

The roadmap is project-first. Courses and papers support the projects rather than replacing them.

## Scope and depth

* **Core:** complete a bounded implementation, measurement, and independent
  explanation for the mechanism. Reuse the same models and benchmark harness.
* **Survey:** explain the idea, assumptions, and tradeoffs; full implementation
  is not required for every named method.
* **Optional extension:** pursue when a later workload, interest, or role needs it.
* Build practical competence in **CUDA C++, Triton, and TileLang**. Extreme
  language/compiler/kernel specialization follows concrete project needs.
* Complete one architecture-transfer project first; other architecture ports,
  advanced serving, and the full platform are specialization branches.
* Existing project IDs remain stable for progress records. Project **4C** adds
  quantization; the recommended order below specifies dependencies and branches.

---

# Core Method

Use the same loop everywhere:

```text
Build a correct baseline
    ↓
Measure it
    ↓
Form a hypothesis
    ↓
Study the relevant theory
    ↓
Change one thing
    ↓
Verify correctness
    ↓
Measure again
    ↓
Explain the result
```

For every optimization, answer:

1. What was the bottleneck?
2. Why should this change improve it?
3. Did it preserve correctness/model quality?
4. Did it improve the whole workload?

---

# Phase 1 — Deep-Learning Internals

## Project 1: Build a Mini Deep-Learning Framework

Build a small framework inspired by CMU's Needle.

Implement:

* tensor storage;
* shapes and strides;
* broadcasting;
* views versus copies;
* reverse-mode autodiff;
* computational graphs;
* modules;
* optimizers;
* minimal CPU tensor backend.

**First pass:** train an MLP using tensor autograd, then trace the complete step.
The existing scalar MLP is a stepping stone, not completion of tensor autograd.

**Later extensions:** add a small CUDA backend alongside Project 3; add a CNN or
recurrent model when it exercises new tensor, gradient, or state semantics.

Understand:

* `loss.backward()`;
* gradient accumulation;
* saved tensors;
* broadcasting backward rules;
* tensor aliasing;
* memory lifetime;
* parameter updates;
* forward versus backward computation.

Intentionally introduce incorrect gradients, accidental copies, bad gradient accumulation, and excessive activation retention, then diagnose them.

**Exit:** You can manually trace a training step from input tensor to optimizer update.

**Course:** CMU 10-414/714 — Deep Learning Systems.

---

# Phase 2 — Performance Engineering

## Project 2: Build an ML Performance Laboratory

Create a reusable benchmarking framework.

Start with:

* CNN;
* transformer encoder;
* small decoder-only transformer.

Later add:

* SSM;
* diffusion denoiser;
* world model.

Record:

* configuration;
* model version;
* hardware/software;
* shapes;
* dtype;
* batch size;
* throughput;
* GPU/CPU memory;
* latency;
* quality metrics;
* profiler traces.

Measure separately:

```text
Input preparation
Host → GPU transfer
Forward
Backward
Optimizer
Synchronization
Compilation
Inference
```

For LLMs additionally measure:

* prefill latency;
* TTFT;
* inter-token latency;
* end-to-end latency;
* tokens/sec.

Use:

* PyTorch Profiler;
* Nsight Systems;
* Nsight Compute;
* CUDA events.

Create bad workloads—CPU preprocessing bottlenecks, unnecessary synchronization, tiny kernels, pathological padding, repeated allocation—and diagnose them from measurements.

### Numerical accuracy and performance reasoning

* Compare FP32, TF32, FP16, and BF16 where supported; distinguish storage,
  multiplication, and accumulation precision.
* Study rounding, overflow/underflow, cancellation, reduction order, absolute
  versus relative error, and the limitations of higher-precision references.
* Set numerical tolerances and held-out model-quality requirements explicitly.
  Agreement with a reference using the same backend is not independent proof.
* Separate nondeterminism, algorithm selection, and numerical error through
  controlled experiments; preserve failed checks as evidence.
* Use Amdahl's law and rough compute/bandwidth bounds to predict useful targets.
* Account for live tensor memory, allocator-reserved memory, temporary workspace,
  and peaks; do not equate total device memory use with live activations.
* Revisit these checks in Project 4C before accepting lower-precision execution.

**Exit:** Given an unfamiliar workload, you can identify where time and memory go and design an experiment to test competing explanations.

**Course:** Stanford CS149 — Parallel Computing.

---

# Phase 3 — GPU Programming

## Project 3: Build a GPU Operator Library

Build a small representative operator collection. The categories below are a
coverage menu, not a requirement to implement every operator in every language.

### Dense

* tiled GEMM;
* GEMM + bias + activation fusion.

### Reduction

* RMSNorm / LayerNorm;
* softmax.

### Spatial

* convolution;
* fused image preprocessing.

### Sequence

* naive attention followed by tiled attention with online softmax; compare
  intermediate memory, correctness, and execution time;
* associative scan (extension for the SSM project).

### Irregular memory access

* embedding gather/reduction.

### Practical coverage of all three languages

Use **CUDA C++, Triton, and TileLang** with comparable first-pass depth:

1. In each, write a small elementwise kernel, a reduction, and a tiled GEMM.
2. Handle boundary tiles/non-divisible shapes; validate against an independent
   reference and debug one deliberate indexing or synchronization error.
3. Benchmark shared shapes/dtypes with the same measurement boundary; explain
   memory access, parallel work assignment, and one measured tuning change.
4. Implement a fused operator and the attention exercise in whichever language
   suits the task. Reuse the others in later projects; avoid three full libraries.

**Defer:** exhaustive autotuning, production GEMM/attention parity, compiler
internals for each DSL, and architecture-specific instruction tuning. Return to
these when a measured project bottleneck justifies the depth.

Study:

* warps;
* coalescing;
* tensor cores;
* shared memory;
* registers;
* occupancy;
* synchronization;
* tiling;
* arithmetic intensity;
* numerical stability.

Build rough performance models:

```text
Ideal runtime ≳ max(
    operations / compute throughput,
    bytes moved / memory bandwidth
)
```

Implement backward for at least one nontrivial operator and integrate at least one kernel into a real model.

### GPU concurrency exercise

Use streams and events to express dependencies between transfer and computation.
Compare serialized execution with a double-buffered pipeline; verify data and
buffer lifetime correctness before interpreting overlap. Inspect the timeline
and explain when shared resources or dependencies prevent overlap.

**Exit:** In each language, you can write, validate, modify, and profile a small
kernel without copying a complete solution. Explain performance across shapes
and integrate at least one operator into a model. Peak hardware performance is
not a first-pass requirement.

---

# Phase 4 — PyTorch Systems Internals

Understand this stack:

```text
Python model
    ↓
PyTorch Dispatcher
    ↓
ATen operators
    ↓
Autograd
    ↓
TorchDynamo
    ↓
FX / ATen graph
    ↓
AOTAutograd
    ↓
Functionalization / decompositions
    ↓
TorchInductor
    ↓
Generated Triton/C++/CUDA
    ↓
CUDA runtime
```

## Project 4A: Build a Proper PyTorch Custom Operator

Take one of your kernels and integrate it fully.

Implement:

* operator schema;
* CPU/CUDA behavior where appropriate;
* fake/meta implementation;
* autograd;
* shape/dtype validation;
* mutation contract;
* compiler compatibility.

Test:

```text
Eager inference
Eager training
Compiled inference
Compiled training
```

Use `torch.library`, `opcheck`, numerical references, and gradient checks.

Investigate whether your custom op helps the whole compiled graph or becomes an opaque optimization boundary.

**Exit:** Your operator works correctly through training and compilation paths, and you can explain end-to-end performance.

---

# Phase 5 — torch.compile

## Project 4B: Build a Torch Compile Laboratory

Do not treat:

```python
torch.compile(model)
```

as a magic flag.

### Graph capture

Study:

* TorchDynamo;
* FX graphs;
* guards;
* graph breaks;
* recompilation;
* dynamic shapes;
* Python control flow;
* data-dependent behavior.

Use diagnostics such as:

```bash
TORCH_LOGS="graph_breaks,recompiles,guards"
```

Create controlled graph breaks and recompilation cases.

### Training compilation

Study:

* AOTAutograd;
* forward/backward partitioning;
* saved tensors;
* functionalization;
* decompositions;
* compiled optimizers;
* later, Compiled Autograd.

### TorchInductor

Trace:

```text
PyTorch operations
    ↓
Captured graph
    ↓
Inductor IR
    ↓
Fusion decisions
    ↓
Generated kernels
    ↓
GPU execution
```

Inspect generated code and identify what fused, what materialized, and what stayed as library calls.

### Compilation economics

Measure:

* compile time;
* warm execution;
* recompilation;
* cache reuse;
* startup overhead.

Estimate:

```text
break-even calls ≈
extra setup cost /
(baseline runtime - optimized runtime)
```

when that denominator is positive.

**Exit:** When `torch.compile` performs poorly, you can determine whether the issue is capture, guards, recompilation, dynamic shapes, autograd, generated kernels, launch overhead, or unsupported ops.

**Supporting resource:** Tianqi Chen's Machine Learning Compilation / MLC.

---

# Phase 5B — Quantization and Low-Precision Systems

## Project 4C: Build a Quantization Laboratory

**Prerequisites:** Project 2 measurement/numerical reasoning and basic Project 3
kernels/Project 4A integration. Compiler comparisons reuse Project 4B. Use a
small pretrained model; completing LLM pretraining is not a prerequisite.

**Goal:** understand the major practical quantization families, implement the
fundamentals, and evaluate selected methods through actual model execution.
Begin with a small vision model or MLP, then a decoder model. Reuse the same
evaluation and benchmark infrastructure rather than creating one per algorithm.

### Foundations — implement and explain

* Reference quantize/dequantize; scale, zero point, rounding, clipping, saturation.
* Symmetric/asymmetric schemes; per-tensor, per-channel, per-token, and per-group
  scaling; static calibration versus dynamic activation scaling.
* Weight-only versus weight-and-activation quantization; separately account for
  KV state, optimizer state, and accumulation precision.
* Uniform INT8/INT4 versus nonuniform codebooks and floating-point formats.
* Outliers, calibration representativeness, layer sensitivity, and mixed precision.
* Packed storage, effective bits per weight including metadata, layout conversion,
  fused dequantization, and fallback to higher-precision computation.
* PTQ (post-training quantization), QAT (quantization-aware training), and actual
  low-precision training. Fake quantization alone does not demonstrate acceleration.

### Algorithm and ecosystem coverage

**Coverage snapshot: 2026-09-30.** This is a broad practical syllabus, not a
measured popularity ranking or an exhaustive list of papers. Survey every row;
the depth column specifies the required experiments. Refresh maintained tooling
and hardware support when starting, using the
[Transformers overview](https://huggingface.co/docs/transformers/main/quantization/overview)
and [vLLM support matrix](https://docs.vllm.ai/en/latest/features/quantization/).

| Method / family | Mechanism to understand | Depth |
| --- | --- | --- |
| Round-to-nearest (RTN), calibrated INT8 PTQ | Scale/range selection, clipping, granularity, static/dynamic activations | **Core:** implement a reference and quantize a small non-LLM model |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Approximate second-order information to compensate weight-quantization error | **Core:** explain a small worked example and run a maintained implementation |
| [AWQ](https://arxiv.org/abs/2306.00978) | Activation-informed weight scaling to protect important channels | **Core:** compare against GPTQ and RTN on the same decoder |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | Move activation quantization difficulty into weights through equivalent scaling | **Core:** test W8A8 and calibration sensitivity |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | Mixed-precision handling of activation outliers | Required conceptual comparison with SmoothQuant; optional separate benchmark |
| [NF4 / QLoRA](https://arxiv.org/abs/2305.14314) | Nonuniform weight representation, double quantization, adapters over a frozen quantized base | Survey here; one bounded fine-tuning experiment in Project 11. QLoRA is a fine-tuning method, not synonymous with QAT |
| [AutoRound](https://github.com/intel/auto-round) and [HQQ](https://github.com/dropbox/hqq) | Optimized rounding versus half-quadratic quantization without calibration data | Survey both; optional experiment with either |
| [OmniQuant](https://arxiv.org/abs/2308.13137) | Learnable clipping and equivalent transformations during calibration | Optional algorithm extension |
| [QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406) | Rotations to manage outliers; fixed/random versus learned rotations | Survey both; optional comparison of rotation cost and quality |
| [AQLM](https://arxiv.org/abs/2401.06118), [VPTQ](https://github.com/microsoft/VPTQ), [SpQR](https://arxiv.org/abs/2306.03078) | Additive/vector representations versus sparse outlier exceptions | Optional extreme-compression branch; include decoding and metadata cost |
| [FP8, MXFP8, MXFP4, NVFP4](https://docs.pytorch.org/ao/stable/workflows/inference.html) | Floating-point formats and scaling recipes, not interchangeable algorithms | **Core:** one FP8 experiment where supported; survey microscaling/FP4, hardware-dependent runs optional |
| [llama.cpp K/IQ schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes) | Block encodings and importance-aware quantization; GGUF is the container | Survey; optional CPU/Apple Silicon deployment experiment with [importance calibration](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) |
| [QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html) | Fake quantization, straight-through gradient estimation, conversion to executable quantized operators | **Core:** small-model PTQ versus QAT quality comparison |
| [BitNet](https://github.com/microsoft/BitNet) | Models designed/trained for very low-bit weights | Optional training specialization; do not assume arbitrary pretrained models can be cast to ternary weights |
| [KIVI](https://arxiv.org/abs/2402.02750), runtime FP8 KV, TurboQuant | KV-cache-specific representation, scaling, and attention integration | Survey here; implementation belongs after the basic Project 10 KV runtime, then the capstone |

Algorithms, data formats, kernel backends, and checkpoint containers are distinct
choices. Record all four for an experiment. A framework exposing a configuration
does not establish fast execution for the selected device, model, and shape.

### Bounded experiment sequence

1. Build RTN and a small-model calibrated PTQ baseline; vary clipping and group
   size and predict which tensors lose accuracy before measuring.
2. On one decoder, compare RTN, GPTQ, and AWQ with matched weight bit width and
   group size where supported; record differences when exact matching is impossible.
3. Evaluate SmoothQuant W8A8 and a supported FP8 recipe against the same floating
   baseline. Include activation quantization overhead and separate prefill/decode.
4. Compare PTQ and QAT on a small model using a held-out evaluation set; separate
   the benefit of extra training from quantization choices where possible.
5. Integrate one quantized operator into the profiling/compiler workflow. Compare
   fake-quantized reference, actual packed execution, and floating baseline.
6. Choose optional methods only when their mechanism or deployment target adds a
   useful comparison. Survey coverage does not require implementing every paper.

For each method record:

* identical starting model/tokenizer and evaluation inputs; calibration data kept
  separate from held-out evaluation; calibration/optimization time and memory;
* weight/activation/KV formats, group sizes, scales/zero points, exceptions,
  accumulation dtype, backend, versions, and verified device support;
* layer error plus model quality (task metric and, for a decoder, perplexity);
* checkpoint size, loaded/peak memory, metadata, packing and conversion overhead;
* unprofiled latency/throughput across batch sizes and sequence lengths; prefill,
  decode, warmup/compilation, and quantize/dequantize costs distinguished;
* a quality/memory/latency comparison showing where compression helps or hurts.

Set acceptance criteria before tuning. Unsupported native formats may be studied
in a reference implementation, but such runs do not establish native speedups.

**Exit:** choose and defend a quantization recipe for a specified model/device
and quality/latency budget, including a case where quantization is not beneficial.
Demonstrate independent transfer to a new layer shape or calibration distribution.

**Later extensions:** sparsity, pruning, and distillation; quantized distributed
training; KV compression; new low-bit methods prompted by deployment needs.

---

# Phase 6 — CUDA Graphs

## Project 5: Build a CUDA Graph Runtime

Keep this conceptually separate from compilation:

```text
torch.compile
→ changes/optimizes computation

CUDA Graphs
→ changes how GPU work is launched/replayed
```

### Manual capture

Build:

```text
Warmup
Capture
Persistent input buffers
Persistent output buffers
Replay
```

Understand stable device addresses and persistent buffers.

### Shape management

Build a graph cache keyed by:

* shape;
* dtype;
* model;
* execution mode.

Compare:

* capture per shape;
* bucketing;
* padding;
* eager fallback.

Dynamic-shape compilation does **not** imply one CUDA Graph works for every shape.

### Training graphs

Investigate:

* optimizer state;
* randomness;
* changing learning rates;
* mutable buffers;
* warmup state.

### CUDAGraph Trees

Study:

* graph recording;
* shared memory pools;
* multiple shapes;
* output lifetime;
* mutation handling.

### Reuse across architectures

Test:

* LLM decoding;
* SSM recurrent state;
* diffusion timesteps/conditioning.

Run this benchmark matrix:

| Mode                             | Purpose           |
| -------------------------------- | ----------------- |
| Eager                            | Baseline          |
| Eager + CUDA Graphs              | Replay benefit    |
| `torch.compile`, graphs disabled | Compiler benefit  |
| `torch.compile` + CUDA Graphs    | Combined behavior |

**Exit:** You can explain when CUDA Graphs help and when shape variability, copying, memory overhead, or capture constraints make them unattractive.

---

# Phase 7 — Data and Training Infrastructure

## Project 6: High-Throughput Data + Training Pipeline

Build pipelines for:

* images;
* variable-length audio;
* language-model data.

Compare:

```text
Naive file loading
Preprocessed/sharded datasets
Asynchronous prefetched pipeline
```

Investigate:

* multiprocessing;
* worker count;
* pinned memory;
* host-to-device transfer;
* mmap;
* shuffling;
* bucketing;
* preprocessing;
* decoding;
* filesystem/page-cache behavior.

For LLMs add:

* tokenization;
* document boundaries;
* sequence packing;
* filtering;
* tokenizer versions.

Checkpoint:

* model;
* optimizer;
* scheduler;
* RNG state;
* training step;
* relevant data/sampler state.

Test corrupted samples, slow workers, worker crashes, interrupted checkpoints, and restart behavior.

### Training-memory milestone (before distributed training)

* Build a memory budget for parameters, gradients, optimizer state, saved
  activations, and temporary workspace; compare estimates with measured peaks.
* Compare activation checkpointing/recomputation with the unchanged baseline.
* Study gradient accumulation and effective batch size, loss normalization,
  optimizer-step frequency, and stateful-layer caveats.
* Run a bounded CPU-offload experiment when useful; measure transfer and
  synchronization cost rather than assuming saved device memory is free.
* Distinguish activation checkpointing from durable recovery checkpoints.

**Exit:** Explain input stalls and memory peaks, demonstrate one measured
memory/time tradeoff, and recover training without silently resetting state.

---

# Phase 8 — LLM Pretraining

## Project 7: End-to-End Language Model Training

Use Stanford CS336 heavily.

Build:

```text
Raw text
    ↓
Filtering
    ↓
Tokenizer
    ↓
Packed sequences
    ↓
Transformer
    ↓
Optimizer
    ↓
Training
    ↓
Evaluation
```

Understand:

* causal masking;
* positional methods;
* attention;
* normalization;
* initialization;
* LR schedules;
* gradient clipping;
* mixed precision;
* packing;
* training stability.

Run controlled scaling experiments and compare quality against:

* tokens processed;
* training FLOPs;
* wall-clock time.

**Exit:** You can reproduce a run, diagnose instability, and explain why changes helped or hurt held-out performance.

**Course:** Stanford CS336 — Language Modeling from Scratch.

---

# Phase 9 — Distributed Training

## Project 8: Build a Distributed Training Laboratory

Use both a CNN/ViT and your language model.

Start by implementing synchronous data parallelism using explicit collectives.

Then study:

* DDP;
* NCCL;
* gradient buckets;
* communication overlap;
* FSDP2;
* DeviceMesh;
* DTensor.

Implement small educational examples of:

* tensor parallelism;
* pipeline parallelism;
* sequence/context parallelism where relevant.

Add an MoE experiment for load balancing and all-to-all communication.

Treat MoE as an extension after the basic dense-model distributed experiments.
Before scaling across nodes, inspect GPU/NIC topology and relate PCIe/NVLink
and inter-node links to communication paths. Benchmark representative collectives
across message sizes and compare standalone bandwidth with communication exposed
on the training critical path. Multi-node runs depend on available hardware;
do not infer their scaling from a single-node test.

Measure:

* fixed-global-batch scaling;
* fixed-per-GPU-batch scaling;
* communication exposure;
* GPU utilization;
* memory;
* checkpoint time;
* recovery time.

Then compare:

```text
Distributed eager
Distributed + torch.compile
```

Test rank failure and recovery.

**Exit:** You can explain why scaling stops being linear and whether the limit is compute, communication, synchronization, input, memory, or imbalance.

---

# Phase 10 — TensorRT

## Project 9: TensorRT Deployment Laboratory

Learn TensorRT as a **general DL inference system**, not only TensorRT-LLM.

Use:

* vision model;
* encoder;
* later diffusion denoiser.

Compare:

```text
PyTorch eager
torch.compile
AOTInductor
TensorRT
```

Study:

### Export

* `torch.export`;
* static/dynamic shapes;
* unsupported operators;
* export constraints.

### TensorRT

* engine compilation;
* tactic selection;
* optimization profiles;
* workspace;
* buffers;
* streams.

### Precision

* FP32;
* FP16/BF16;
* supported FP8 paths;
* INT8 where appropriate.

Reuse the quantization recipes and acceptance criteria from **Project 4C**.
Compare the actual supported export/runtime paths rather than treating a precision
flag as proof that every operator executes in that precision.

### Dynamic shapes

Build several optimization profiles and benchmark min/typical/max cases.

### Custom plugin

Integrate one of your own GPU kernels.

**Exit:** Another engineer can reproduce your engine, validate model quality, and understand its supported shapes, precision, hardware, and runtime constraints.

---

# Phase 11 — LLM Inference

## Project 10: Build an LLM Runtime

Support one decoder architecture.

**First pass:** complete V1–V3 with correct request cancellation and state cleanup.
Then add V4 as a bounded memory-management milestone. V5 and full integrations
with multiple serving backends are optional extensions driven by measurements.

### V1

* prefill;
* autoregressive decode;
* KV cache.

### V2

* batching across different sequence lengths.

### V3

* continuous batching.

### V4

* block-based KV allocation;
* free lists;
* block tables;
* cancellation;
* prefix sharing.

### V5 — Optional advanced serving

Investigate:

* chunked prefill;
* prefix caching;
* speculative decoding;
* quantized KV;
* CPU/off-device KV;
* disaggregated prefill/decode.

Compare against:

* vLLM;
* TensorRT-LLM.

Integrate:

* `torch.compile`;
* CUDA Graphs;
* custom kernels;
* quantization;
* streams;
* allocator strategies.

Measure separately:

* prefill;
* decode;
* queueing;
* scheduling;
* kernel time;
* memory overhead.

**Exit:** You can explain a request's full lifetime from arrival through scheduling, cache management, execution, streaming, cancellation, and cleanup.

---

# Phase 12 — LLM Post-Training

## Project 11: SFT + Preference Training + Rollouts

Use a suitably sized pretrained model.

**First pass:** bounded SFT and preference-training runs with reproducible data,
evaluation, and checkpoint recovery. Revisit Project 4C with one small NF4/QLoRA
fine-tuning experiment. Full rollout infrastructure is a later specialization.

Implement:

### SFT

Understand formatting, masking, packing, and parameter-efficient methods where useful.

### Preference optimization

Implement something such as DPO and understand the objective directly.

### Rollout/reward/update loop — Optional systems extension

```text
Model
    ↓
Generate rollouts
    ↓
Score/reward
    ↓
Training examples
    ↓
Update model
    ↓
New model version
```

Track:

* model versions;
* rollout provenance;
* sampled probabilities;
* worker/learner synchronization;
* checkpoint state.

**Core exit:** explain the SFT/preference objectives, evaluate a small run, and
account for adapter/quantized-base behavior. **Extension exit:** explain and
recover the repeated rollout/reward/update process, including model provenance.

---

# Phase 13 — Architecture Transfer: SSMs

## Project 12: Replace the Transformer with an SSM

**Architecture-transfer option:** choose this or Project 13 first. Complete the
other later if it serves your goals; neither is a prerequisite for serving work.

Use Mamba/Mamba-2 as a reference.

Start from the equations.

Understand:

* recurrent state;
* selective updates;
* scan formulations;
* parallel training;
* incremental inference.

Build:

1. sequential reference;
2. efficient sequence implementation;
3. training integration;
4. incremental inference runtime.

Compare:

* training memory;
* throughput;
* sequence-length scaling;
* recurrent-state size;
* prompt-processing behavior;
* generation latency.

Ask:

> Which assumptions from my transformer infrastructure are now wrong?

**Exit:** Produce an architecture-port report explaining what infrastructure was reusable and what needed replacement.

---

# Phase 14 — Architecture Transfer: Diffusion

## Project 13: Build a Diffusion / Flow-Matching System

**Architecture-transfer option:** choose this or Project 12 first. Start with
one small denoiser; a second architecture and all deployment backends are extensions.

Implement a small model yourself.

Understand:

* corruption/noising;
* denoising objectives;
* score/velocity/noise prediction;
* sampling;
* relevant ODE/SDE intuition.

Start with a U-Net; optionally transfer to a transformer-based denoiser such as
a DiT-like architecture after completing the first system.

Profile:

```text
Encoder
Conditioning
Denoiser
Repeated sampling
Decoder
```

Reuse:

* distributed training;
* `torch.compile`;
* CUDA Graphs where applicable;
* TensorRT.

Keep systems optimizations separate from algorithmic changes such as reducing sampling steps.

**Course:** MIT flow-matching/diffusion material.

---

# Phase 15 — Architecture Transfer: World Models

## Project 14: Build an Action-Conditioned World Model

**Optional specialization:** revisit after one architecture-transfer project;
it is not required before retrieval, serving, or the platform milestone.

Use a small simulated environment.

Train on:

```text
Observation history + Actions
    ↓
Predict future observation/state
```

Start with supervised prediction, then multi-step rollouts.

Evaluate:

* prediction quality;
* temporal drift;
* response to changed actions;
* rollout consistency;
* inference cost.

Eventually use the world model for a small planning/control problem.

**Exit:** The model demonstrates useful temporal behavior, not merely attractive frames.

---

# Phase 16 — Recommendation and Embedding Systems

## Project 15: Retrieval + Ranking

**Breadth option:** retain for sparse-access and embedding-system experience;
it can follow the core training/serving work independently of world models.

Build:

```text
Interactions
    ↓
Features
    ↓
Two-tower model
    ↓
Embedding index
    ↓
Candidate retrieval
    ↓
Ranking model
```

Learn:

* large embedding tables;
* sparse memory access;
* embedding sharding;
* feature freshness;
* training-serving skew;
* index versioning.

Compare with TorchRec.

This exposes systems problems very different from dense transformer execution.

---

# Phase 17 — Multi-Model Serving

## Project 16: Build a General Inference Service

Serve:

* TensorRT vision model;
* encoder;
* stateful audio model;
* LLM.

Build a simple scheduler yourself, then compare with NVIDIA Triton Inference Server.

Understand:

* dynamic batching;
* sequence batching;
* model instances;
* priorities;
* deadlines;
* admission control;
* backpressure;
* cancellation;
* health checks.

Measure:

* latency percentiles;
* queue time;
* throughput;
* rejection rate;
* resource utilization.

Compare cost per successful request (or token) **subject to fixed model-quality
and latency targets**. Record hardware price assumptions and actual utilization;
distinguish offline throughput from goodput that meets the service target.
Sweep arrival rates and request lengths to expose queueing and overload behavior.

Test overload, worker failure, client disconnects, slow models, and malformed inputs.

**Exit:** You understand the difference between a fast model executable and a reliable inference service.

---

# Phase 18 — ML Platform

## Project 17: Build an ML Lifecycle Platform

**First pass:** one containerized train → evaluate → deploy → rollback path,
with versioned artifacts, durable metadata, and a recovery test. Support one
workload first. The general platform below is an optional expansion, including
Kubernetes, cluster GPU scheduling, and multiple training/serving backends.

Build:

```text
Dataset version
    ↓
Training job
    ↓
Checkpoint
    ↓
Evaluation
    ↓
Registered model
    ↓
Compiled artifact
    ↓
Canary deployment
    ↓
Promotion / rollback
```

Support:

* vision training;
* LLM training;
* batch jobs;
* online inference.

Use:

* containers;
* object storage;
* durable metadata;
* Kubernetes;
* GPU scheduling.

Understand:

* control plane versus workers;
* idempotency;
* retries;
* quotas;
* artifact lineage;
* reproducibility;
* monitoring;
* rollback.

Later add:

* distributed training;
* TensorRT builds;
* compilation caches;
* CUDA Graph warmup;
* per-model runtime policies.

**Core exit:** another engineer can reproduce the chosen workload's training,
deployment, and rollback from the recorded configuration. **Extension exit:**
they can use the broader multi-workload platform without manual intervention.

---

# Phase 19 — TurboQuant Capstone

Your TurboQuant project becomes a strong cross-layer capstone:

Prerequisites: Project 4C quantization foundations and a working Project 10 KV
runtime. Treat this as a later specialization; it does not replace general
quantization coverage. Begin with one representation/runtime integration before
adding connectors or distributed serving.

```text
Mathematical representation
    ↓
Reference implementation
    ↓
CUDA/TileLang kernels
    ↓
PyTorch integration
    ↓
torch.compile compatibility
    ↓
KV cache representation
    ↓
vLLM integration
    ↓
KV connectors
    ↓
Distributed serving
```

Keep the design boundary:

> TurboQuant handles representation and computation. Existing systems such as vLLM/LMCache should handle scheduling and transfer where possible.

Measure:

* KV memory;
* metadata overhead;
* quant/dequant overhead;
* transfer bytes;
* attention error;
* downstream model quality;
* end-to-end serving latency.

---

# University Courses

Use these selectively alongside the relevant projects.

| Course                                               | Main role                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| **CMU 10-414/714 — Deep Learning Systems**           | Autograd, tensor libraries, GPU backends, framework internals         |
| **Stanford CS149 — Parallel Computing**              | CUDA, memory hierarchy, scans, scheduling, parallel performance       |
| **Stanford CS336 — Language Modeling from Scratch**  | LLM pretraining, systems, scaling, data, post-training                |
| **MIT 6.5940 — Efficient Deep Learning**             | Quantization, sparsity, efficient architectures, distributed training |
| **Machine Learning Compilation — MLC / Tianqi Chen** | IRs, graph transformations, compiler optimization                     |
| **MIT Flow Matching / Diffusion material**           | Diffusion, flow matching, sampling                                    |
| **Stanford CS329S — ML Systems Design**              | Data, evaluation, deployment, monitoring                              |
| **MIT 6.5840 — Distributed Systems**                 | Fault tolerance, replication, consistency                             |

Do not complete all of them before building. The project determines which material you study.

---

# How to Use AI

Use AI heavily, but separate **project progress** from **personal understanding**.

## When the mechanism is new

Use:

```text
My current mental model
    ↓
AI critiques it
    ↓
AI explains missing prerequisite
    ↓
I attempt implementation
    ↓
AI reviews/debugs
```

Good prompts:

> Here is my understanding of CUDA Graph replay. Identify the first important misconception.

> I think this kernel is memory-bound. Here are the profiler results. Give me competing explanations and an experiment that distinguishes them.

## When you understand the mechanism

Delegate aggressively.

Give AI:

* interfaces;
* invariants;
* acceptance criteria;
* test requirements.

Let it implement:

* scaffolding;
* bindings;
* experiment runners;
* alternative kernels;
* deployment configuration.

Then review and test it.

## Use AI adversarially

Ask:

> Assume this implementation is wrong. Construct the smallest case that exposes the bug.

Useful targets:

* KV sharing + cancellation;
* distributed gradient normalization;
* CUDA Graph output lifetime;
* shape-induced recompilation;
* SSM state resets;
* diffusion timestep handling.

Turn those cases into tests.

---

# Personal Mastery Tests

For each major project, assess yourself separately from the codebase.

### Explain

Can you explain the mechanism without reading AI's explanation?

### Modify

Can you change a meaningful constraint—shape, dtype, batching policy, architecture, distributed strategy?

### Diagnose

Can you investigate a new failure before asking AI?

---

# Architecture Transfer Checklist

Whenever a new architecture appears, answer:

1. What is the learning objective?
2. What computation happens?
3. What tensor shapes and dependencies exist?
4. What persistent state exists?
5. What does training need?
6. What does inference need?
7. How does cost scale with sequence length, resolution, batch size, rollout length, etc.?
8. What can be parallelized?
9. What proves correctness?
10. Which assumptions in the existing stack no longer hold?

That last question is especially important.

---

# Recommended Order

```text
1. Mini DL framework
2. Performance/profiling laboratory
3. Practical CUDA C++ + Triton + TileLang operators (Project 3)
4. PyTorch custom ops (Project 4A)
5. torch.compile internals (Project 4B)
6. Quantization foundations, survey, and core experiments (Project 4C)
7. CUDA Graphs (Project 5)
8. Data pipeline + training-memory optimization (Project 6)
9. LLM pretraining (Project 7)
10. Distributed training (Project 8)
11. TensorRT deployment (Project 9)
12. LLM runtime V1–V4 (Project 10; V5 optional)
13. Bounded post-training + QLoRA (Project 11; rollout platform optional)
14. One architecture transfer: SSM or diffusion (Project 12 or 13)
15. General inference service and cost/latency evaluation (Project 16)
16. One reproducible lifecycle path (Project 17)

Optional branches, when relevant:
- Additional architecture transfer and world model (Projects 12–14)
- Retrieval/embedding systems (Project 15)
- Advanced serving, full rollout system, general ML platform
- Quantization algorithm extensions and KV experiments
- TurboQuant capstone
- Deep kernel/DSL/compiler specialization driven by a project bottleneck
```

These should overlap. Once you learn `torch.compile`, CUDA Graphs, profiling, or distributed execution, reuse them in every later project.

The order is a learning path, not a requirement to finish every extension before
continuing. Quantization's KV and fine-tuning experiments resume with their runtime
and post-training prerequisites; unsupported hardware formats remain survey topics.

---

# The Four Expertise Pillars

## GPU / Performance

```text
CUDA
TileLang
Triton
Profiling
Memory hierarchy
Kernel optimization
CUDA Graphs
```

## Framework / Compiler

```text
PyTorch internals
Autograd
Dispatcher
Custom ops
TorchDynamo
AOTAutograd
TorchInductor
torch.export
AOTInductor
```

## Training Systems

```text
Data pipelines
Mixed precision
Distributed training
DDP
FSDP
Tensor parallelism
Checkpointing
LLM pretraining
Post-training
```

## Inference / Infrastructure

```text
TensorRT
TensorRT-LLM
vLLM
KV management
Continuous batching
Triton Inference Server
Kubernetes
Observability
ML lifecycle
```

Across all four:

```text
CNNs
Transformers
LLMs
SSMs
Diffusion
World models
Recommendation systems
Future architectures
```

---

# What “Finished” Looks Like

The goal is not:

> I know CUDA.

or:

> I know vLLM.

or:

> I completed CS336.

It is:

> **Give me an unfamiliar deep-learning architecture. I can read the paper, understand its computation and state, implement a correct baseline, train it, profile it, determine where it wastes resources, integrate or write the necessary kernels, reason about compilation, choose an appropriate distributed strategy, design its inference runtime, and deploy it reliably.**

Then when the next Mamba, FlashAttention, diffusion architecture, training algorithm, or inference paradigm appears, you do not need a new roadmap—you have the systems foundations to figure it out.
