# 2026-11-04 — PyTorch register softmax walkthrough

## Context

- Project3 softmax blog, TUTOR; user requested the next kernel in the same LaTeX
  algorithm format with a legend. Starting commit `b664fcb`, local Mac.
- Target: one-block-per-row mapping, retained logits and hierarchical reductions.

## Work and evidence

- Added `project_03_gpu_operator_library/notes/05-softmax-comparison/12-pytorch-register-softmax.md`
  and section index link. Read recorded SoftMax.cu and pinned upstream
  SoftMax.cu/block_reduce.cuh at revision70d99e998b4955e0049d13a98d77ae1b14db1f45.
- Parsed six saved PyTorch profile CSVs: rows37/256/4096 × widths4096/8192.
  All show block1024, grid equal to rows, Reg template4/8 respectively.
- Compared Reg kernel bodies in the Oct27 and Oct31 source snapshots; identical.
- Explained explicit per-thread reductions, shared partials, three barriers per
  block reduction, and the output epilogue's second source exponential evaluation.
  Did not infer actual exponential instruction counts or performance causality.
- Checked note step numbering, LaTeX structure, local links and `git diff --check`.
  No CUDA execution, visual rendering, new benchmark or source implementation.

## Learning and handoff

- Worked explanation supplied; independent understanding not assessed.
- Next: discuss Reg's block-wide reductions and retained-input behavior, then General.
- Existing changes and GEMM resume point preserved; no commit or push.
