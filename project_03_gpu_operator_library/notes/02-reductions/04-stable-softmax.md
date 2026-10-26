# 04 — Stable softmax

## First contract

- Contiguous FP32 input/output [37,130], separate output, finite inputs, no masking.
- FP32 intermediate calculations; one 128-thread block per row.
- Every row produces 130 probabilities; all share that row's denominator.

## Shift invariance and stability

- Direct exp on large logits such as [1000,1001,1002] overflows in FP32; subsequent infinity/infinity yields NaN.
- Subtracting a common constant m multiplies every numerator and the denominator by exp(-m), which cancels mathematically. Do not implement this by computing overflowing exp(x) first.
- Choose m=max(x). Shifted inputs are nonpositive; exponentials are at most 1 and at least one is 1. Very small probabilities can still underflow.
- Compute Y[c] = exp(X[c]-m) / sum_j exp(X[j]-m). In floating point, equivalence and a row sum of 1 are approximate.
- **Kernel phases:** row maximum → shifted exponentials → row sum → normalization/output.
- Stable baseline implemented and source-reviewed; online normalization and tuning remain future lessons. No compilation or GPU validation yet.

## Maximum reduction and shared-array reuse

- Initialize private maxima to **negative infinity**, including for threads with no assigned elements.
- Combine with `fmaxf(partials[t], partials[t + offset])`; every thread reaches each block barrier.
- After the final maximum-reduction barrier, each thread copies `partials[0]` into its private `row_max`.
- **Barrier after that read:** prevents an early writer from overwriting shared slot 0 with its exponential sum before another warp reads the maximum.
- Then compute private exponential sums, store them in the shared array, and synchronize before the sum reduction.
- These barriers protect different dependencies: finishing old reads versus making new writes available.
- First baseline recomputes exponentials for output. Retaining values on-chip is a later optimization to compare.
- After the final sum-reduction barrier, reading the denominator needs no additional barrier when the shared array is never written again. This differs from reading the maximum before reusing the array.

## Correctness harness

- [Host scaffold and kernel](../../kernels/cuda/softmax.cu) use a [CPU-only reference/validator](../../kernels/cuda/softmax_test_support.h).
- **Reference:** promote stored FP32 inputs to FP64, subtract the row maximum, compute exponentials and sequential sum, then normalize.
- **Elementwise criterion:** abs(actual-reference) <= 1e-7 + 1e-5*abs(reference).
- **Invariants:** all outputs finite and in [0,1]; each row sum within 1e-5 of 1.
- **Cases:** equal/negative/large positive/large negative/mixed logits, tied maxima, dominant tail; widths around 128 and longer rows. NaN output initialization exposes missing writes.
- Row sums alone are insufficient: a wrongly permuted probability vector can still sum to 1.
- [CPU tests](../../kernels/cuda/softmax_reference_test.cpp) check known results, shift invariance, and rejection of corrupted outputs. These passed on the Mac; CUDA compilation/device testing remains pending.
