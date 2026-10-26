# 02 — Bias broadcasting and ReLU order

## Contract and indexing

- **Operation:** Y[r,c] = ReLU(X[r,c] + bias[c]).
- **Layout:** contiguous row-major FP32 X/Y, shape [37,130]; bias has 130 elements. Separate output, finite inputs.
- **Flat element count:** rows * cols = 4810; guard `idx < rows * cols`.
- **Bias index:** `idx % cols`; column bias repeats for every row.
- Flat indices 129, 130, 131 select bias entries 129, 0, 1.
- X and Y use the same flat index; computing the row is unnecessary for this kernel.

## Operation order

1. Load the input and corresponding column bias.
2. Add them.
3. Apply ReLU to the sum and store the result.

- **Pitfall:** ReLU(x) + bias differs from ReLU(x + bias).
- Example: x=-3, bias=2 gives required output 0; applying ReLU first gives 2.
- **Review status:** corrected source reviewed; GPU compilation and validation pending. Initial test does not define general NaN/signed-zero behavior.
