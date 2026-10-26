# 06 — Online softmax

## Why change the normalizer?

- Current baseline/warp kernels traverse each thread's assigned inputs three times: maximum, shifted exponential sum, output.
- **Online normalization** maintains the maximum and shifted sum together during one traversal; a second traversal produces standalone softmax outputs.
- These are logical input traversals, not guaranteed DRAM transactions. Caches and compiler behavior affect actual traffic.
- Fewer traversals may help, but rescaling adds arithmetic/dependencies. Compare the same cases and fixed correctness criteria; no speedup measured.

## Running state

- **m:** maximum of the elements processed so far.
- **s:** sum of their exponentials shifted by that same m; not the sum of logits or normalized probabilities.
- Example: after [2,3], m=3 and s=exp(-1)+1.
- If the next value is 5, the old contributions must now be exp(2-5)+exp(3-5), before adding exp(5-5). Updating m without adjusting s mixes two different shifts.

## Online update and pair merge

- For a new finite input x, compute **m_new = max(m_old, x)**.
- Update **s_new = s_old * exp(m_old - m_new) + exp(x - m_new)**, then replace the old state. Preserve the old maximum until its rescaling factor is computed.
- For [2,3] followed by 5, rescale the old sum by exp(3-5)=exp(-2), then add 1. A following 4 leaves m=5, rescales by 1, and adds exp(-1).
- Compute `expf(x - m)` directly; separately computing exp(x)/exp(m) can overflow.

To merge nonempty summaries A and B:

```text
mC = max(mA, mB)
sC = sA * exp(mA - mC) + sB * exp(mB - mC)
```

- Both sums must use the **same shift** before addition. Pair merging supports thread → warp → block reduction without rereading summarized inputs.
- At each shuffle step, fetch both partner fields before updating the local pair. Floating-point reduction order can change rounding.

## Empty groups and output

- Empty-group state: **(-INFINITY, 0)**. Under the finite-input contract, a nonempty group's shifted sum includes exp(0)=1; zero identifies an empty summary.
- Handle empty groups before evaluating rescaling: if A is empty, return B; if B is empty, return A. Two empty groups remain empty.
- Blindly merging two empty groups evaluates -INFINITY - (-INFINITY), producing NaN; multiplying by zero does not remove NaN.
- All named lanes must still execute both shuffles. Apply empty-group handling after fetching the partner pair; do not skip shuffles for no-data lanes or return early from the kernel.
- Once the final row pair is broadcast to all threads, output **y[i] = exp(x[i] - m) / s** in a second traversal.
- Online kernel is implemented and source-reviewed after guided corrections. The host suite runs baseline, warp and online on the same ten cases, with unchanged references/tolerances and a fresh NaN output sentinel per launch. GPU compilation/validation and speedup measurement remain pending.

## References

- [Online normalizer calculation for softmax](https://arxiv.org/html/1805.02867v2), Milakov and Gimelshein.
- [Reduction notes](README.md)
