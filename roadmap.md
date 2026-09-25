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
* CPU backend;
* small CUDA backend.

Train:

* MLP;
* CNN;
* small recurrent model.

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

**Exit:** Given an unfamiliar workload, you can identify where time and memory go and design an experiment to test competing explanations.

**Course:** Stanford CS149 — Parallel Computing.

---

# Phase 3 — GPU Programming

## Project 3: Build a GPU Operator Library

Implement operations with different computational structures.

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

* attention;
* associative scan.

### Irregular memory access

* embedding gather/reduction.

Use CUDA C++, Triton, and TileLang.

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

**Exit:** You can explain performance differences across shapes and workloads instead of merely reporting speedups.

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

**Exit:** You can account for GPU idle time caused by the input pipeline and recover training without silently resetting state.

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

Also investigate `torchao`.

### Dynamic shapes

Build several optimization profiles and benchmark min/typical/max cases.

### Custom plugin

Integrate one of your own GPU kernels.

**Exit:** Another engineer can reproduce your engine, validate model quality, and understand its supported shapes, precision, hardware, and runtime constraints.

---

# Phase 11 — LLM Inference

## Project 10: Build an LLM Runtime

Support one decoder architecture.

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

### V5

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

Implement:

### SFT

Understand formatting, masking, packing, and parameter-efficient methods where useful.

### Preference optimization

Implement something such as DPO and understand the objective directly.

### Rollout/reward/update loop

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

**Exit:** You understand both the learning objective and the systems required to repeatedly generate data and update models.

---

# Phase 13 — Architecture Transfer: SSMs

## Project 12: Replace the Transformer with an SSM

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

Implement a small model yourself.

Understand:

* corruption/noising;
* denoising objectives;
* score/velocity/noise prediction;
* sampling;
* relevant ODE/SDE intuition.

Start with a U-Net, then use a transformer-based denoiser such as a DiT-like architecture.

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

Test overload, worker failure, client disconnects, slow models, and malformed inputs.

**Exit:** You understand the difference between a fast model executable and a reliable inference service.

---

# Phase 18 — ML Platform

## Project 17: Build an ML Lifecycle Platform

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

**Exit:** Another engineer can train and deploy models through your infrastructure without manual intervention from you.

---

# Phase 19 — TurboQuant Capstone

Your TurboQuant project becomes a strong cross-layer capstone:

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
3. CUDA + TileLang operators
4. PyTorch custom ops
5. torch.compile internals
6. CUDA Graphs
7. Data pipeline
8. LLM pretraining
9. Distributed training
10. TensorRT
11. LLM inference engine
12. LLM post-training
13. SSM architecture port
14. Diffusion
15. World model
16. Recommendation/embedding systems
17. Multi-model serving
18. ML platform
19. TurboQuant capstone
```

These should overlap. Once you learn `torch.compile`, CUDA Graphs, profiling, or distributed execution, reuse them in every later project.

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
