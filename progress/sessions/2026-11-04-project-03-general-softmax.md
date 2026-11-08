# 2026-11-04 — PyTorch general softmax walkthrough

## Context

- Project3, TUTOR; user requested adding the remaining PyTorch kernel in the same
  compact algorithm format. Local Mac; starting repository head `b664fcb`.
- Target: vectorized three-pass softmax versus retained-register softmax.

## Work and evidence

- Added `project_03_gpu_operator_library/notes/05-softmax-comparison/13-pytorch-general-softmax.md`
  and section index link; updated status/project record.
- Inspected recorded `SoftMax.cu`: kernel, `ilpReduce`, vector/scalar output,
  alignment handling and epilogue. Inspected benchmark allocation source.
- Parsed all12 extended PyTorch captures with Python CSV reader: rows37/256/4096
  × widths16384/32768/65536/131072, all template4, block1024, grid equal to rows.
- Algorithm assumes finite inputs and aligned full vector batches, with separate
  notes on scalar prefixes/tails and mismatched output alignment. Kernel symbol
  alone does not establish executed internal branches or instruction counts.
- Checked numbered steps, LaTeX brace balance, local links and `git diff --check`.
  No visual rendering, CUDA execution, new measurements or code changes.

## Learning and handoff

- Explanation supplied; independent understanding not assessed.
- Next: discuss General, then connect all three PyTorch families to profiles.
- Source/data remain as downloaded; existing local work and GEMM resume preserved.
  No commit or push.
