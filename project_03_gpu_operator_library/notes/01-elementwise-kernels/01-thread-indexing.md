# 01 — Thread indexing and boundaries

## Contract

- **Operation:** `y[i] = a * x[i] + b`; FP32 scalars and contiguous FP32 arrays.
- **Ownership:** one thread per output; separate output array; input unchanged.
- **Launch model:** one-dimensional grid and blocks.

## Mapping

| Quantity | Expression |
| --- | --- |
| Element index | `blockIdx.x * blockDim.x + threadIdx.x` |
| Access guard | `i < N` |
| Blocks, positive integers | `(N + B - 1) // B` in Python; integer `/` in C++ |

- `blockIdx.x`: zero-based block index; `blockDim.x`: threads per block; `threadIdx.x`: zero-based position inside the block.
- **Example:** N=1000, B=128 requires 8 blocks (1024 threads).
- Last block starts at element 896. Threads 0–103 handle elements 896–999; threads 104–127 skip array accesses.

## Pitfalls

- `ceil(1000 // 128)` equals 7: integer division discards the fraction before ceiling. Use `ceil(1000 / 128)` or the integer formula.
- Guard the input read and output write, not just the write.
- These formulas assume arithmetic fits the chosen integer types; the first kernel uses small sizes and an `int` index.
- **Source review is not GPU validation:** compilation, runtime correctness, and performance still need device checks.
