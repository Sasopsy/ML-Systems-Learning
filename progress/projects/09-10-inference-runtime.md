# Projects 9–10: Connected Inference Runtime Track

## Status — planned, 2026-10-20

- User approved the proposal; canonical scope is the [runtime track in the roadmap](../../roadmap.md#connected-runtime-track--scheduling-memory-and-gpu-execution).
- Milestones: basic executor → execution backends → request scheduler → decoder/KV runtime → vLLM source trace and bounded comparison.
- Preserve Projects 9 and 10 IDs and their existing TensorRT/LLM coverage. Reuse profiling, kernels, compiler, and CUDA Graph work from earlier projects.
- First version uses one GPU, one small model at a time, and in-process requests. Start with existing tensor operations; the learner implements scheduling, ownership, and lifecycle mechanisms.
- TensorRT-LLM comparison and broader serving integrations are extensions.

## Evidence and next task

- Planning approved; implementation and independent understanding are **not assessed**.
- No runtime code, GPU runs, installations, or benchmarks performed.
- When this track starts: choose a small model and define the executor's input/output, completion, and buffer-ownership contract before implementation.
- Current learning remains Project 2, paused at unanswered Q21. No new project lesson notes created before lessons begin.
- Roadmap and progress changes local; no commit or push in this update.
