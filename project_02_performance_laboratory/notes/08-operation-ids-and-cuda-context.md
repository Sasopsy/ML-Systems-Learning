# Operation IDs, CUDA contexts, and queue timestamps

## External id: which PyTorch operation invocation?

- **External** means an ID supplied from outside NVIDIA's CUPTI profiler;
  here PyTorch supplies it through Kineto.
- Each recorded CPU operation invocation gets an ID. Repeated calls to the same
  operator get different IDs; IDs do not encode layer number or nesting depth.
- PyTorch 2.11 allocates ranges for event-storage blocks, then uses
  **range start + event position**. Nearby calls on one thread often have
  consecutive IDs; do not infer global ordering across threads from the numbers.
- Entering an operation pushes its ID onto a per-thread external-correlation
  stack; exiting pops it. CUDA calls made inside the innermost active scope are
  linked to that scope; the profiler carries the link to their GPU work.

Actual nested CPU scopes in our saved trace:

```text
aten::conv2d                 389
  aten::convolution         390
    aten::_convolution      391
      aten::cudnn_convolution 392
```

## Correlation: which CUDA launch produced this kernel?

- **CUPTI correlation** links a CUDA API invocation on the CPU to the GPU activity
  it caused, even when execution happens later.
- Separate ID system from PyTorch's External id; numbers need not match.
- For our selected convolution, CPU `cudaLaunchKernelExC` and the 11.808 us GPU
  kernel both have correlation **5036** and External id **392**.
- One PyTorch operation can issue multiple CUDA launches:

| GPU activity | External id | Correlation shared with its CPU launch |
| --- | --- | --- |
| NCHW to NHWC conversion | 392 | 5026 |
| Another NCHW to NHWC conversion | 392 | 5030 |
| Convolution | 392 | 5036 |
| NHWC to NCHW conversion | 392 | 5040 |

- Match IDs rather than guessing from the nearest CPU event or kernel name.

## CUDA context: resource and execution environment

- A **device** is the GPU; a **context** holds CUDA resources/state associated
  with a device, including its address space, allocations, loaded code, and streams.
- A **stream** orders submitted work within that environment.
- Our event: **device 0 → context 1 → stream 7 → kernel invocation**.
- Context 1 is an identifier, not GPU 1 or one context per operation.
- Normal runtime use initializes a primary context as needed for a device;
  CPU threads in the application can share it.

## Plain-language prerequisites and deep-learning example

- **Execution state:** information needed to carry out or continue computation.
  Examples: the current instruction and temporary values held by running threads.
  In a running sum, the current position and partial sum are execution state.
  Context management also tracks settings/resources used by that execution;
  exact save/restore details depend on hardware and scheduling.
- **Virtual address:** the memory address a program uses. Hardware translates it
  to the actual storage location using mappings managed by the driver/system.
- Analogy: a program uses a locker label, while a lookup table identifies the
  actual locker. Separate programs can have separate label-to-location mappings.
- **Context:** a program's CUDA workspace on a GPU, containing its resources
  and the information CUDA needs to manage its work.
- **Stream:** an ordered to-do list within that workspace.
- Illustrative example, **not executed**: independently launch `train_resnet.py`
  and `train_transformer.py` in two terminals, both using the same GPU 0.
  Each Python process normally initializes its own primary context on that GPU.
- Context A holds ResNet allocations/work; context B holds Transformer
  allocations/work. Their tensors are separate by default; both consume the
  physical GPU's memory and compete for its compute resources.
- Two model objects in **one ordinary PyTorch process on the same GPU** normally
  share a context. A model object does not automatically create a context.

## How the GPU distinguishes contexts

- A context is implemented through **driver bookkeeping plus GPU-side resources
  and execution state**; the numeric profiler ID is only a label for that context.
- **Memory mappings:** page tables translate virtual addresses used by kernels
  into physical memory locations and determine accessible mappings.
- **Work submission:** the driver associates GPU work queues with the context
  whose code, mappings, and execution state apply to that work.
- Conceptual example with separate processes: virtual address `0x1000` in
  context A can map to physical page X, while `0x1000` in context B maps to Y
  or is unmapped. Raw pointer equality does not establish shared memory.
- Above a stream means **resource ownership and shared environment**. Streams
  in one ordinary context share its address space; they provide separate ordering
  of operations. Correct synchronization is still required for shared data.
- CUDA streams are software queues mapped by the driver to GPU work queues;
  neither a stream nor an ordinary context means a permanently dedicated SM.
- Ordinary separate contexts typically time-share compute execution; switching
  selects the appropriate execution/address-space state. Existing allocations
  need not be copied out of VRAM at every switch.
- Special modes such as **MPS** change concurrency across clients. These modes
  were not inspected for this trace. Exact submission formats and hardware state
  layouts depend on architecture/driver and were not reverse-engineered here.

## queued: when the driver recorded the command

- **Timestamp**, in nanoseconds: when the kernel launch was queued in the
  driver's command buffer. It is neither a duration nor the queue length.
- Command buffers carry launch/copy commands from the driver to the GPU.

| Timestamp | Milestone |
| --- | --- |
| queued | Kernel command queued in driver command buffer |
| submitted | Command buffer submitted to GPU |
| start | Kernel execution starts |
| end | Kernel execution ends |

- When valid timestamps are available, **start − queued** measures the interval
  from enqueue to execution; **end − start** measures kernel duration.
- Submitted time can further separate pre-submission and post-submission delay;
  these intervals alone do not diagnose why the kernel waited.
- CUPTI does **not collect queued/submitted timestamps by default**.
  Our `queued: 0` supplies no usable enqueue timing and does not mean zero wait.
- Perfetto's 1.430 us flow delay in our example uses **CPU launch end → GPU start**;
  it is not a measurement derived from `queued`.

## Sources

- [PyTorch 2.11 event ID allocation and stack push](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/profiler/collection.cpp#L295-L354)
- [CUPTI external-correlation mechanism](https://docs.nvidia.com/cupti/main/main.html)
- [CUPTI kernel correlation and timestamps](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html)
- [CUDA 13 programming guide: context](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#context)
- [NVIDIA MPS architecture: streams, work queues, and context scheduling](https://docs.nvidia.com/deploy/mps/architecture.html)
- [NVIDIA virtual memory management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
