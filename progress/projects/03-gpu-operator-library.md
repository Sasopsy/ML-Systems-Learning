# Project 3: GPU Operator Library

## Status

- **Active planning**, requested 2026-10-20. Scope: [roadmap](../../roadmap.md#project-3-build-a-gpu-operator-library).
- User writes learning-target kernels; assistant teaches, reviews, and maintains authorized notes/progress.
- No kernel implementation, local execution environment verification, remote connection, or GPU measurement yet.
- Project 2 review remains paused at Q21; profiling continues within kernel experiments.

## Proposed development loop

1. **Mac:** specify shapes, dtype, layout, operation, reference, edge cases, and numerical criteria.
2. **Mac:** learner derives work assignment and indices, writes one small kernel, and checks indexing with a CPU model. CPU checks do not validate GPU races, compilation, or performance.
3. **GPU:** compile and run correctness cases, including non-divisible sizes; use relevant memory/race diagnostics before performance claims.
4. **GPU:** measure warmed-up execution with explicit synchronization and baseline boundaries; collect a small separate diagnostic profile.
5. **Mac:** inspect saved results, form a hypothesis, predict one change, implement it, and repeat.
6. **Transfer:** learner handles an unfamiliar shape or mapping change and explains correctness/performance without a complete supplied solution.

## Proposed sequence

- **First milestone:** FP32 contiguous `y = a*x + b`; CUDA C++ first, then Triton and TileLang versions of the same contract.
- **Next:** row reduction, then tiled GEMM, with practical coverage in all three languages.
- **Later roadmap coverage:** fusion, normalization/softmax, attention, representative spatial/irregular operators, one nontrivial backward, real-model integration, streams/events exercise. This is coverage planning, not a commitment to implement every operator in every language.
- First learning mechanism: mapping logical elements to parallel work and handling the final partial block.
- First independent check: derive and validate indexing for an unseen non-divisible size; no kernel solution supplied yet.

## Machine handoff proposal

- Keep source, small tests, environment specifications, notes, and compact evidence in Git. Commit/push when authorized; use the same recorded revision remotely.
- Keep large profiler reports separate; copy them back with an artifact manifest/checksums before losing access to the instance.
- Record GPU model, driver/toolkit/framework/compiler versions, command, shapes, dtype, correctness outcome, timing boundary, and code revision (plus any local diff).
- Separate CPU reference imports from GPU-only imports/builds so local reference checks can run independently.
- Check the existing GPU environment before choosing/installing versions. Use separate environments where language dependencies require them.
- Use short GPU validation loops for early kernels; batch planned experiments only once the execution path works.
- Timing and profiling remain separate runs; cross-language comparisons use equivalent contracts and boundaries.

## Current next task

- Agree the first operator contract, then ask the learner to derive work assignment and boundary handling.
- [Notes index](../../project_03_gpu_operator_library/notes/README.md).
- Latest session: [Mac/GPU workflow planning](../sessions/2026-10-20-project-03-workflow.md).
- Changes local and uncommitted; no remote synchronization.
