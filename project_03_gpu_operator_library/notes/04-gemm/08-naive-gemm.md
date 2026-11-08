# 08 — Naive GEMM: layout and output ownership

## Initial contract

- **C = A B**, with A[M,K], B[K,N], C[M,N]. Separate contiguous row-major FP32 arrays; FP32 accumulation. Positive dimensions, no transpose, bias or activation.
- This is the alpha=1, beta=0 case of general GEMM; existing C values are not read.
- Initial teaching shape: M=37, K=130, N=65. FP16/BF16 matrix-instruction precision remains a later, separately specified milestone.

## Deep-learning interpretation

- A holds 37 input vectors with 130 features each; B maps those features to 65 outputs.
- Each output C[r,c] is a dot product: sum over k of A[r,k] * B[k,c].
- Each valid thread owns one output and accumulates its full dot product privately; different threads do not combine partial results in this first version.
- Proposed launch: 16×16 threads/block, x maps to output columns and y maps to output rows. Edge threads must not access invalid outputs.
- Row-major storage places adjacent columns next to one another. Matrix dimensions determine the row stride.

## Status

- Indexing/launch exercise is pending; no kernel, correctness tests or performance measurements yet.
- [GEMM notes](README.md)
