# Register-buffered online softmax

- **Algorithm and storage are separate choices:** online normalization can consume values retained in a private buffer.
- **Original experiment:** fixed width 1024, 128 threads/row, eight FP32 values/thread. The sweep now specializes compile-time widths 128–8192; keep the existing online kernel for comparison.

## Steps

1. Load each thread's `dims / N` strided values into a fixed-size private array.
2. Compute its online maximum/sum from those saved inputs.
3. Merge summaries across warps as before.
4. Normalize saved inputs using the final row maximum/sum and write outputs.

## Pitfalls

- Retain **original inputs**, not provisional exponentials whose normalization maximum can change.
- A private array is not automatically register-resident. Constant indexing/unrolling helps; check compiler allocation and spills.
- More retained values need not produce a simple additive increase in register count: instruction scheduling and live ranges can change too.
- Fixed-width specialization/unrolling can change generated code beyond removing loads. Do not attribute every speed difference to memory traffic alone.
- Keep mapping, input, tolerances and timing boundaries comparable. Profile to test whether fewer loads actually reduce exposed dependency latency.
- **Template API:** `softmax_online_registers<N, dims>` (default `dims = 1024`); `N` is the number of threads per block. Example: `softmax_online_registers<256, 2048><<<rows, 256>>>(x, y)`. The former restriction to `N == 128` is removed.
- Current contract: **32 ≤ N ≤ 1024, N divisible by 32, positive dims divisible by N**, and exactly N launched threads. The benchmark bridge still uses N = 128 and dims 128/512/1024/2048/4096/8192/16384/32768/65536/131072. Supporting other runtime widths needs dispatch or boundary handling.

## Why buffering does not remove every overhead

- Registers retain each thread's inputs; **shared memory still exchanges row summaries between warps**. Register shuffles communicate within a warp, not directly across warps.
- At N = 128, four warps/row: each publishes its pair → block barrier → warp0 merges pairs/publishes final pair → block barrier → all warps read the final pair.
- Online pair reduction costs more than scalar max/add reductions: two summary fields, exponentials for rescaling and additional control flow.
- Conditional single-value updates save an exponential per participating lane at source level, but lanes can choose different cases, creating divergent machine branches. Measure the tradeoff.
- Warp-instruction counts count issued operations even when only some lanes participate; divergence can reduce useful lane participation while adding branch-path work.

## Choosing `N`

- **`N` means threads per block**, not row width. One block handles one row; `N / 32` warps cooperate, and each thread retains `dims / N` values.
- The algorithm is not inherently limited to 128 threads. This configurable version requires `32 <= N <= 1024`, `N % 32 == 0`, positive `dims`, and `dims % N == 0`, with exactly `N` threads launched and resources within device limits.
- These conditions give complete warps, equal work per thread, and at most 32 warp summaries for the final warp to reduce. Nondivisible widths need additional boundary handling.
- The old `static_assert(N == 128)` restricted an experiment configuration. It has been replaced by general complete-warp and divisibility checks; choosing the fastest valid N requires measurement.
- Example at width 1024: 32 threads retain 32 values each; 128 retain 8; 256 retain 4. Changing block size changes register demand, work per thread and reduction overhead, so it needs its own benchmark.

## Larger dims and spills

- At N = 128, dims = 131072 means **1024 retained values per thread**. A private array can overflow the available register allocation and spill to device-backed local memory.
- Observed build: **16384 dims → 168 registers/thread, no spills**; **32768–131072 dims → 255 registers/thread, spills**. Read compiler resource reports and runtime local-memory counters; the variant name is not a residency guarantee.
- PyTorch already beat retained online at 16384 dims in the 4096-row graph test, before spills appeared. Spills alone do not explain every performance difference.
- [Timing tables through 128K dims](../../results/2026-10-31-softmax-128k/README.md) and [large-dims paired profiles](../../results/2026-10-31-offline-profiles-128k/README.md) contain the evidence. N stayed 128; this was not a block-size tuning sweep.
