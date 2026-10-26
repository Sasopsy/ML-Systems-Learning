# 2026-10-22 — Performance-focused operator plan

## Context

- Project/milestone: Project 3 roadmap revision.
- Mode: TUTOR planning; authorized documentation changes, no learning-target code.
- Learning target: planned optimized softmax, GEMM, attention, and backward across CUDA C++, Triton, TileLang.
- Starting commit: `f6d6002`; previous kernel/notes/progress work already local and uncommitted.
- Environment: local macOS; GPU, compiler, and framework versions not inspected in this session.

## Work and evidence

- Read learning instructions, current status, active project record, latest session follow-ups, and roadmap Project 3.
- User requested online/optimized softmax, naive/shared/MMA/WGMMA GEMM, direct MMA/WGMMA attention, and efficient forward/backward operators.
- Revised roadmap scope, performance acceptance, backward requirements, and proposed language roles. Updated status and project record; no kernel source changed.
- Official references checked:
  - [Online normalizer paper](https://arxiv.org/abs/1805.02867).
  - [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
  - [Triton fused attention including backward](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
  - [Triton Gluon WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html).
  - [TileLang GEMM interfaces](https://www.tilelang.com/autoapi/tilelang/language/gemm_op/index.html): default synchronous GEMM versus explicit asynchronous WGMMA scheduling.
  - [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608).
- Documentation capabilities are not verification of locally/remote installed versions. Target Hopper/H100 and backend versions remain to be checked before execution.

## Experiment

- None. No compilation, installation, GPU session, benchmark, or performance result.
- Workload suites and numerical targets will be fixed before tuning; no universal performance advantage or library-parity claim.

## Learning and handoff

- Learner request: increase performance depth and include backward; asked for Triton/TileLang plan. Detailed sequencing is assistant-proposed.
- No new mastery assessment. Prior source-review evidence remains unchanged.
- Next task: stable fused softmax contract and normalization mechanism offline; existing-kernel GPU batch remains deferred at user request. Advanced tuning will require measured iterations.
- Changes local and uncommitted; no push or remote synchronization.

## Follow-up — inline PTX wrappers

- User proposed custom CUDA `asm volatile` wrappers, as in SageAttention, for GEMM/MMA. Updated roadmap and progress to make this explicit.
- Inspected [SageAttention mma.cuh](https://github.com/thu-ml/SageAttention/blob/main/csrc/mma.cuh), [wgmma.cuh](https://github.com/thu-ml/SageAttention/blob/main/csrc/wgmma.cuh), and [NVIDIA inline PTX guidance](https://docs.nvidia.com/cuda/inline-ptx-assembly/).
- Plan includes operand/register constraints and instruction layouts, minimal tile validation, then reusable primitives inside GEMM/attention. Existing PTX instructions are wrapped, not newly defined. Compiler volatile/clobbers do not substitute for hardware synchronization.
- Documentation only; no kernel edits, compile/run, dependency install, or remote activity. Next softmax lesson unchanged.

## Follow-up — low-precision outputs

- User requested FP16/BF16 outputs for MMA/WGMMA. Updated roadmap and progress to specify matching input/output dtype with FP32 accumulation and epilogue conversion.
- Accumulation precision was not changed; numerical validation and library baselines must account for the output conversion. No implementation or GPU action.

## Follow-up — low-precision accumulation requested

- User explicitly requested FP16/BF16 accumulation. Verified [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), mma and wgmma.mma_async instruction syntax: FP16 inputs allow FP16 accumulator types; BF16 input forms require FP32 accumulators.
- Updated canonical roadmap to include native FP16 accumulation and explain unsupported native BF16 accumulation. Periodic BF16 rounding of FP32 partial sums is a distinct possible experiment, not a native substitute or accepted implementation decision.
- Earlier FP32-only plan superseded. No kernel implementation, execution, or GPU performance claim.

## Confirmed precision decision

- User accepted native FP16 accumulation for FP16 inputs and FP32 accumulation for BF16 inputs; output remains the matching input dtype.
- Replaced the unresolved choice in the roadmap with this agreed contract. Matrix accumulation settings do not implicitly prescribe softmax statistics/reduction precision.
- Documentation only; no compile/run, commit, or push. Next lesson unchanged.
