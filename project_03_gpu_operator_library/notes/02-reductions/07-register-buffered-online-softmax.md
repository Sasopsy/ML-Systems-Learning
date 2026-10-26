# Register-buffered online softmax

- **Algorithm and storage are separate choices:** online normalization can consume values retained in a private buffer.
- **Original experiment:** fixed width 1024, 128 threads/row, eight FP32 values/thread. The sweep now specializes compile-time widths 128–8192; keep the existing online kernel for comparison.

## Steps

1. Load each thread's eight strided values into a fixed-size private array.
2. Compute its online maximum/sum from those saved inputs.
3. Merge summaries across warps as before.
4. Normalize saved inputs using the final row maximum/sum and write outputs.

## Pitfalls

- Retain **original inputs**, not provisional exponentials whose normalization maximum can change.
- A private array is not automatically register-resident. Constant indexing/unrolling helps; check compiler allocation and spills.
- More retained values need not produce a simple additive increase in register count: instruction scheduling and live ranges can change too.
- Fixed-width specialization/unrolling can change generated code beyond removing loads. Do not attribute every speed difference to memory traffic alone.
- Keep mapping, input, tolerances and timing boundaries comparable. Profile to test whether fewer loads actually reduce exposed dependency latency.
- Current contract: **128 threads/block and a compile-time positive width divisible by 128**. The default width is 1024; sweep dispatch explicitly supports 128/512/1024/2048/4096/8192. It is not an arbitrary runtime-width fallback.

## Why buffering does not remove every overhead

- Registers retain each thread's inputs; **shared memory still exchanges row summaries between warps**. Register shuffles communicate within a warp, not directly across warps.
- Four warps/row: each publishes its pair → block barrier → warp0 merges pairs/publishes final pair → block barrier → all warps read the final pair.
- Online pair reduction costs more than scalar max/add reductions: two summary fields, exponentials for rescaling and additional control flow.
- Conditional single-value updates save an exponential per participating lane at source level, but lanes can choose different cases, creating divergent machine branches. Measure the tradeoff.
- Warp-instruction counts count issued operations even when only some lanes participate; divergence can reduce useful lane participation while adding branch-path work.
