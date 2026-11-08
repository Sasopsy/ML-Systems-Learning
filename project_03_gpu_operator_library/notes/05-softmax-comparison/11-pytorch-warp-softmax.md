# 11 — Algorithm: PyTorch warp softmax

Ordinary, unmasked softmax for finite, contiguous inputs; `1 <= ncols <= 2048`.
All three types are `float` in our measurements. This is source-derived algorithm
notation, not compilable CUDA or a statement of exact instruction scheduling.

## Legend

| Name | Meaning |
|---|---|
| `X`, `Y`; `nrows`, `ncols` | Input/output matrices; their row and column counts |
| `cap`, `lgcap` | Power-of-two row capacity; its base-2 logarithm |
| `nlanes`, `nvals` | Lanes cooperating on a row; retained values per lane per row |
| `batch`, `ngrp` | Independent rows per lane group; lane groups per block |
| `vals`, `rmax`, `rsum` | Private retained values, row maxima and row denominators |
| `lmax`, `lsum` | This thread's local maximum and partial sum, both in `acc_t` |
| `lane`, `b`, `j` | Lane index, row slot within the group, value slot within a lane |
| `row`, `col`, `row0` | Global row/column index; the group's first row |
| `input_t`, `output_t`, `acc_t` | Input, output and accumulator types (PyTorch names) |

`vals[b,j]` is value slot `j` for row slot `b` **inside this thread**.
`X[row,col]` denotes flat offset `row*stride+col`, with `stride=ncols`.

$$
\begin{array}{l}
\textbf{Algorithm 1: PyTorch warp softmax} \\[3pt]
\textbf{Runtime: }X,Y,\mathrm{nrows},\mathrm{ncols},\mathrm{stride},\mathrm{stream}\\
\textbf{Template: }\mathrm{input\_t},\mathrm{output\_t},\mathrm{acc\_t},\mathrm{lgcap}\\
\qquad \mathrm{is\_log\_softmax}=\mathrm{false},\quad\mathrm{is\_masked}=\mathrm{false}\\[6pt]
\textbf{Host: choose a compiled kernel and launch}\\
1.\quad \mathrm{lgcap}\gets\lceil\log_2(\mathrm{ncols})\rceil;\quad\mathrm{cap}\gets2^{\mathrm{lgcap}}\\
2.\quad \mathrm{nlanes}\gets\min(\mathrm{cap},32);\quad\mathrm{nvals}\gets\mathrm{cap}/\mathrm{nlanes}\\
3.\quad \mathrm{batch}\gets\begin{cases}2,&\mathrm{cap}\leq128\\1,&\text{otherwise}\end{cases};\quad
\mathrm{ngrp}\gets128/\mathrm{nlanes}\\
4.\quad \mathrm{block}\gets(\mathrm{nlanes},\mathrm{ngrp},1)\\
5.\quad \mathrm{grid}\gets(\lceil\mathrm{nrows}/(\mathrm{ngrp}\cdot\mathrm{batch})\rceil,1,1)\\
6.\quad\text{Launch the specialization for the types, lgcap and flags.}\\[6pt]
\textbf{Device: per-thread computation}\\
7.\quad\mathrm{lane}\gets\mathrm{threadIdx.x}\\
8.\quad\mathrm{row0}\gets(\mathrm{blockIdx.x}\cdot\mathrm{ngrp}+\mathrm{threadIdx.y})\cdot\mathrm{batch}\\
9.\quad\text{Private acc\_t arrays: }\mathrm{vals}[\mathrm{batch},\mathrm{nvals}],\ \mathrm{rmax}[\mathrm{batch}],\ \mathrm{rsum}[\mathrm{batch}]\\[5pt]
\textit{Load once; retain logits.}\\
10.\quad\textbf{for }b=0,\ldots,\mathrm{batch}-1:\\
11.\qquad\mathrm{row}\gets\mathrm{row0}+b\\
12.\qquad\textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
13.\qquad\quad\mathrm{col}\gets\mathrm{lane}+j\cdot\mathrm{nlanes}\\
14.\qquad\quad\mathrm{vals}[b,j]\gets-\infty\\
15.\qquad\quad\textbf{if }\mathrm{row}<\mathrm{nrows}\ \land\ \mathrm{col}<\mathrm{ncols}:\\
16.\qquad\qquad\mathrm{vals}[b,j]\gets\operatorname{cast}_{\mathrm{acc\_t}}(X[\mathrm{row},\mathrm{col}])\\[5pt]
\textit{Find the local maximum, then reduce across lanes.}\\
17.\quad\textbf{for each }b:\\
18.\qquad\mathrm{lmax}\gets-\infty\\
19.\qquad\textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
20.\qquad\quad\mathrm{lmax}\gets\max(\mathrm{lmax},\mathrm{vals}[b,j])\\
21.\qquad\mathrm{rmax}[b]\gets\operatorname{warpReduce}(\mathrm{lmax},\max)\\[5pt]
\textit{Retain exponentials; sum locally, then reduce across lanes.}\\
22.\quad\textbf{for each }b:\\
23.\qquad\mathrm{lsum}\gets0\\
24.\qquad\textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
25.\qquad\quad\mathrm{vals}[b,j]\gets\exp(\mathrm{vals}[b,j]-\mathrm{rmax}[b])\\
26.\qquad\quad\mathrm{lsum}\gets\mathrm{lsum}+\mathrm{vals}[b,j]\\
27.\qquad\mathrm{rsum}[b]\gets\operatorname{warpReduce}(\mathrm{lsum},+)\\[5pt]
\textit{Normalize and store.}\\
28.\quad\textbf{for each }b,j:\\
29.\qquad\mathrm{row}\gets\mathrm{row0}+b;\quad\mathrm{col}\gets\mathrm{lane}+j\cdot\mathrm{nlanes}\\
30.\qquad\textbf{if }\mathrm{row}<\mathrm{nrows}\ \land\ \mathrm{col}<\mathrm{ncols}:\\
31.\qquad\quad Y[\mathrm{row},\mathrm{col}]\gets\operatorname{cast}_{\mathrm{output\_t}}(\mathrm{vals}[b,j]/\mathrm{rsum}[b])\\[6pt]
\textbf{Helper: }\operatorname{warpReduce}(\mathrm{value},\mathrm{op})\\
32.\quad\textbf{for }\mathrm{offset}=\mathrm{nlanes}/2,\mathrm{nlanes}/4,\ldots,1:\\
33.\qquad\mathrm{peer}\gets\operatorname{shuffleXor}(\mathrm{value},\mathrm{offset},\mathrm{nlanes})\\
34.\qquad\mathrm{value}\gets\mathrm{op}(\mathrm{value},\mathrm{peer})\\
35.\quad\textbf{return }\mathrm{value}\qquad\text{(complete result in every lane)}
\end{array}

$$

## Reading the algorithm

- **Notation:** the per-row reductions above group independent work for readability;
  PyTorch's helper interleaves the rows at each shuffle offset.
- **Compile-time versus runtime:** the host chooses `lgcap` from the runtime
  `ncols` and selects an existing compiled kernel. Inside it, `cap`,
  `nlanes`, `nvals` and `batch` are fixed, allowing fixed-size
  private arrays and unrolled loops. Actual dimensions, pointers and stride remain
  runtime arguments. There is no per-call compilation or autotuning.
- **Retained storage:** `vals` is private to each thread. It holds original
  logits through the maximum reduction, then the same slots hold exponentials
  through the denominator reduction and output. Thus the algorithm loads each
  valid input once and computes its exponential once. These private values can
  occupy registers, but may spill to local memory; they are not shared memory.
- **Local then warp reduction:** each thread first scans its own `nvals` values
  into `lmax`; the warp combines these local maxima into `rmax[b]`. Each thread
  similarly accumulates its exponentials into `lsum`, then the warp combines
  those partial sums into `rsum[b]`. The maximum loop leaves `vals` unchanged;
  the exponential loop replaces logits in those same slots.
- **Reduction:** `shuffleXor(value, offset, nlanes)` exchanges with a peer in
  the same lane group. All participating lanes execute each exchange. At32 lanes,
  offsets16,8,4,2,1 leave the result in every lane. No lane0 broadcast or
  shared-memory/block-wide reduction is needed. Each `b` is independent.
- **Padding:** width900 selects capacity1024. The124 nonexistent positions receive
  negative infinity, become zero after exponentiation and are not stored. No
  padded global tensor is allocated. Invalid tail rows never load/store data;
  their unused internal arithmetic can produce NaNs.

| Private retained state | During maximum reduction | During sum reduction | At output |
|---|---|---|---|
| `vals[b][slot]` | Original logit in accumulator type | Exponential in accumulator type | Divide the same exponential by the denominator |

## Three ideas behind the algorithm

- **Capacity and log:** `cap` is the smallest power of two covering `ncols`;
  `lgcap = ceil(log2(ncols))` selects that compiled capacity. For 900 columns,
  `lgcap=10`, `cap=1024`; the 124 unused slots hold negative infinity, then zero
  after exponentiation. No padded global tensor is allocated.
- **Interleaving for coalescing:** `col = lane + j*32`. Lane 0 owns columns
  `0,32,64,...`, but at the same `j`, all 32 lanes access adjacent columns
  (`0–31`, then `32–63`, etc.). Those simultaneous adjacent accesses coalesce;
  stores use the same mapping.
- **Local then warp reduction:** at width 1024, each thread first reduces its
  own 32 values to `lmax`; the warp combines 32 local maxima into the row maximum.
  Each thread then computes exponentials and their local sum `lsum`; the warp
  combines 32 partial sums into the denominator. Both results reach every lane.

## Concrete instantiations

| Columns | Capacity log2 | Lanes per row | Values per lane per row | Rows per group | Private retained array | Rows per block |
|---:|---:|---:|---:|---:|---|---:|
| 128 | 7 | 32 | 4 | 2 | `float[2][4]` | 8 |
| 512 | 9 | 32 | 16 | 1 | `float[1][16]` | 4 |
| 1024 | 10 | 32 | 32 | 1 | `float[1][32]` | 4 |
| 2048 | 11 | 32 | 64 | 1 | `float[1][64]` | 4 |

- **Shape `[37,1024]`:** selects
  `softmax_warp_forward<float,float,float,10,false,false>`, with block `(32,4,1)`
  and grid `(10,1,1)`. Each warp processes one row. A lane owns columns
  `lane, lane+32, ..., lane+992`, retaining32 values plus other private state.
  This is not a claim of exactly32 total registers per thread.
- **Width128:** a warp processes two independent rows, each with its own maximum
  and denominator. They are never normalized together.
- Below width32, `nlanes` describes logical subgroups; hardware warps remain32
  lanes. All measured warp-family widths use32 lanes per row.

## Source names and other template settings

| Descriptive name | PyTorch source name |
|---|---|
| `input_t`, `output_t`, `acc_t` | `input_t`, `output_t`, `acc_t` |
| `lgcap` | `log2_elements` |
| `cap` | `next_power_of_two` |
| `nlanes` | `WARP_SIZE` |
| `nvals` | `WARP_ITERATIONS` |
| `batch` | `WARP_BATCH` |
| `vals`, `rmax`, `rsum` | `elements`, `max_value`, `sum` |
| `nrows`, `ncols`, `stride` | `batch_size`, `element_count`, `stride` |

- **Types:** input/output types control global storage, while accumulator type
  controls retained storage and intermediate arithmetic. FP16 input/output with
  FP32 accumulation would load/cast into FP32 retained values and cast back on
  output.
- **`is_log_softmax=true`:** retain original logits instead of overwriting them
  with exponentials. Accumulate exponentials into the denominator, then output
  `retained - rmax - log(rsum)`.
- **`is_masked=true`:** additional runtime mask inputs exclude masked elements
  from the maximum and sum. With ordinary softmax and valid finite unmasked data,
  masked positions contribute/output zero. This differs from bounds padding;
  full mask addressing and exceptional cases are in the source.

## Comparison to inspect next

At width1024, PyTorch assigns one warp per row and retains32 values per thread;
our retained Online assigns four warps per row and retains8 inputs per thread.
PyTorch keeps exponentials for output, whereas ours keeps original inputs and
computes output exponentials later. Our kernel also performs online state updates
and shared pair merging with block barriers. These are differences to check
against profiles, not yet a demonstrated cause of the timing difference.

## Source references and scope

- Recorded environment: **H100, PyTorch 2.11.0+cu130**, FP32 unmasked last-dimension
  forward softmax. Recorded revision: `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
- [Pinned upstream kernel and dispatch helper](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/PersistentSoftmax.cuh):
  `softmax_warp_forward`, `warp_reduce`, `dispatch_softmax_forward`.
- [Pinned upstream operator dispatch](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu).
- Offline copies: [kernel snapshot](../../results/2026-10-31-offline-profiles-128k/source/pytorch/PersistentSoftmax.cuh)
  and [operator dispatch](../../results/2026-10-31-offline-profiles-128k/source/pytorch/SoftMax.cu).
  These require the separate results download. Navigate by symbol names because
  wrapper/formatting differences can change line numbers between copies.
- Our recorded warp-family widths are 128/512/1024/2048 at every tested row count.
  This FP32 path checks `columns <= 2048` and `columns*sizeof(float) <= 8192`.
  The older source comment mentioning 1024 is not the actual dispatch limit.
- The walkthrough describes this selected path, not all versions or dtypes. The
  source guards tail rows/columns; shuffle exchanges provide within-warp
  cooperation, and there is no shared-memory reduction or block-wide barrier here.
- [Results](09-results-and-kernel-inventory.md) · [Plots](10-runtime-plots.md) ·
  [Section index](README.md).
