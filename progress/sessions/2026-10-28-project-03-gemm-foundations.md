# 2026-10-28 — Naive GEMM foundations

## Context

- Project 3, TUTOR, local macOS, HEAD b664fcb from earlier verified pull.
- User requested a short start on GEMM. Kernel body remains learner work; scaffolding/reference/measurement support can be delegated.

## Work and evidence

- Reviewed learning instructions, active progress, latest Mac handoff, note index and agreed GEMM roadmap/precision contract.
- Established first exercise: C=AB for M37/K130/N65, contiguous row-major FP32 with FP32 accumulation. One output/thread and 16×16-thread blocks. This scalar baseline does not alter later FP16/BF16 Tensor Core precision choices.
- Introduced dense-layer interpretation and per-output dot product. Next learner task: grid dimensions, output guard and flat A/B/C offsets.
- Added concise note 08, section/root indexes and current progress. No learning-target code or implementation solution supplied.

## Learning and handoff

- Learner indexing answer pending; no mastery or performance claim. No compilation/GPU execution this lesson.
- Continue with indexing review, then learner kernel implementation and delegated host validation when requested.
- Softmax crossover interpretation, backward and language ports remain incomplete. Changes local/uncommitted; no new Git synchronization.
