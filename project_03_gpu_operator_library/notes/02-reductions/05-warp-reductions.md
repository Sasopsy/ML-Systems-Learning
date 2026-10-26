# 05 — Warp reductions and shared-result lifetime

## Two levels of reduction

- **128 threads = four warps.** Keep each thread's strided input accumulation.
- Within each warp, exchange register values with `__shfl_down_sync` at offsets **16, 8, 4, 2, 1**. Lane 0 obtains the complete warp reduction.
- Each warp's lane 0 writes one shared partial; all threads then call **`__syncthreads()`**.
- Warp 0 loads the four partials and reduces again. Its remaining lanes contribute **0 for sum**, **negative infinity for maximum**.
- With a full mask, all 32 lanes of that warp execute each shuffle, including lanes without input data. A shuffle does not synchronize different warps or replace shared-memory ordering barriers.

## Row sum versus softmax

- **Row sum:** only warp 0's lane 0 needs the final total to write the output.
- **Softmax:** every thread needs the row maximum and denominator for its assigned elements.
- Warp 0's lane 0 publishes the final result in shared memory; a block barrier precedes other threads reading it.
- Before reusing the maximum's shared slot for sum partials, every thread must have copied the maximum privately. Add a block barrier **after the reads, before overwriting**.
- Without that barrier, a faster warp can overwrite the slot while a slower warp still needs to read the maximum.
- No read-before-overwrite barrier is needed after the last use if no subsequent write reuses the storage.

## Experiment boundary

### Source comparison at 128 threads per block

| Kernel | Shared-tree floats | Warp-version floats | Shared-tree block barriers | Warp-version block barriers |
| --- | --- | --- | --- | --- |
| Row sum | 128 | 4 | 8 | 1 |
| Softmax | 128 | 4 | 17 | 5 |

- Softmax's five barriers protect: warp maximum publication, row maximum publication, reads before shared reuse, warp sum publication, and denominator publication.
- These counts describe source structure, not measured runtime or compiler-generated instruction counts.

- Preserve shared-tree baselines and keep input, block size, numerical checks, and softmax exponential recomputation unchanged.
- **Hypothesis:** fewer shared-memory operations and block barriers improve performance. No GPU validation or speedup measurement yet.
- **Correctness harnesses:** both variants use identical inputs and independent CPU references. Reset output to NaN before each launch so a previous correct result cannot conceal missing writes; check each variant separately rather than only comparing them to one another.
- Row sum checks exact results for bounded integer fixtures; softmax uses its existing fixed tolerances and probability/row-sum checks. Both harnesses now cover ten cases per variant; GPU execution remains pending.
- [Reduction notes](README.md)
