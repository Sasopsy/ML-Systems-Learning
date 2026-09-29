# 2026-09-30 — Adopt roadmap scope and quantization coverage

## Context

- **Scope:** user accepted the review and requested practical coverage of all
  three GPU languages, with extreme specialization deferred, and broad coverage
  of current quantization methods with optional extensions.
- **Mode:** BUILD for roadmap/progress documentation only.
- **Learning target:** define future milestones; no new lesson or assessment.
- **Starting commit:** `9772fbfa1fe61493cb8ebabba4bb819eaf890a82`.
- **Environment:** supplied local macOS workspace context; Python/framework/GPU
  versions not revalidated. No hardware experiments or package installations.
- Existing uncommitted review notes from 2026-09-29 were preserved.

## Work and evidence

- Updated `roadmap.md` with core/survey/optional scope and stable project IDs.
- Project 3 now requires elementwise, reduction, and tiled GEMM exercises in
  CUDA C++, Triton, and TileLang; shared validation/profiling and a tuning change.
  Full operator libraries and extreme backend specialization are deferred.
- Added Project 4C with quantization foundations, a dated method/format survey,
  core RTN/GPTQ/AWQ/SmoothQuant/FP8/QAT experiments, and optional algorithm paths.
- Survey includes LLM.int8(), NF4/QLoRA, AutoRound, HQQ, OmniQuant, QuaRot,
  SpinQuant, AQLM, VPTQ, SpQR, microscaling/FP4, llama.cpp encodings, BitNet,
  and KV methods. This is broad coverage, not an empirical popularity ranking.
- Deferred QLoRA execution to post-training and KV execution to the LLM runtime.
  Distinguished algorithms, numeric formats, containers, and runtime kernels.
- Added numerical accuracy, memory optimization, GPU concurrency, tiled attention,
  hardware topology, and serving-cost milestones. Reduced mandatory framework,
  architecture-transfer, rollout, advanced-serving, and general-platform scope.
- Updated `progress/STATUS.md` and the Project 1/2 notes without changing
  implementation/mastery status. Earlier session history retained.
- Commands: `cat`, `head`, `sed`, `rg`, `git status --short`,
  `git rev-parse HEAD`, `git diff -- roadmap.md`, and `git diff --check`.
- Validation: reviewed the document diff for prerequisites, stable IDs, core
  versus optional requirements, and continuity with current learning progress.
  No learning code modified; no benchmark or model-quality result claimed.

## Sources

- Checked official [Transformers quantization overview](https://huggingface.co/docs/transformers/main/quantization/overview),
  [vLLM quantization support](https://docs.vllm.ai/en/latest/features/quantization/),
  and [torchao inference](https://docs.pytorch.org/ao/stable/workflows/inference.html).
- Method rows link original papers or official repositories; llama.cpp rows
  link its encoding and importance-calibration documentation.
- Tooling coverage informed the syllabus; local compatibility and performance
  remain untested. Recheck versions and hardware before running each milestone.

## Learning and handoff

- **User preference:** practical competence across all three GPU languages;
  broad quantization understanding with optional deeper experiments.
- **Understanding evidence:** no new technical mastery assessment this session.
- **Next task:** resume Project 2 accumulation error and numerical versus
  application-level acceptance using the existing float64 comparison.
- Roadmap/progress edits remain local. No commit, push, or remote sync check.
- No new project folder or lesson notes created: quantization is planned,
  not started, and this session revised the curriculum rather than teaching it.
