# 12 — Algorithm: PyTorch register softmax

**Measured path:** `cunn_SoftMaxForwardReg` for contiguous FP32 rows of width 4096
or 8192. Each row uses **one 1024-thread block (32 warps)**. The algorithm retains
original logits and performs block-wide maximum and sum reductions.

## Legend

| Name | Meaning |
|---|---|
| `X`, `Y`; `nrows`, `ncols` | Input/output; row and column counts |
| `nt`, `nvals` | Threads per block; retained values per thread |
| `tid`, `lane`, `warp` | Thread index in block; lane and warp indices |
| `row`, `col`, `j` | Matrix row/column; private value slot |
| `vals` | Private array holding original logits throughout |
| `lmax`, `lsum` | This thread's maximum and exponential sum |
| `rmax`, `rsum` | Full row maximum and denominator, available to every thread |
| `smem` | Shared scratch: one FP32 partial result per warp |

All value/accumulator/output types are FP32 here; `index_t` is `int64_t`.
`nvals` is the source template parameter `reg_cnt`; the output template
`Epilogue` is `SoftMaxForwardEpilogue` (the final exponential/division).
`X[row,col]` denotes flat offset `row*ncols+col`. This is source-derived
pseudocode for finite inputs, not an instruction-count prediction.

$$
\begin{array}{l}
\textbf{Algorithm 2: cunn\_SoftMaxForwardReg}\\[3pt]
\textbf{Host inputs: }X,Y,\mathrm{nrows},\mathrm{ncols},\mathrm{stream}\\
\textbf{Kernel runtime: }X,Y,\mathrm{ncols}\\
\textbf{Template: }\mathrm{scalar\_t},\mathrm{accscalar\_t},\mathrm{outscalar\_t},\\
\qquad \mathrm{Epilogue},\mathrm{index\_t},\mathrm{nvals}\\[6pt]
\textbf{Host: launch for the measured widths 4096 and 8192}\\
1.\quad \mathrm{nt}\gets1024;\quad\mathrm{nvals}\gets\lceil\mathrm{ncols}/\mathrm{nt}\rceil\\
2.\quad \mathrm{block}\gets(\mathrm{nt},1,1);\quad\mathrm{grid}\gets(\mathrm{nrows},1,1)\\
3.\quad \text{Select the compiled nvals specialization; launch with 128 bytes shared scratch.}\\[6pt]
\textbf{Device: one block per row}\\
4.\quad \mathrm{row}\gets\mathrm{blockIdx.x};\quad\mathrm{tid}\gets\mathrm{threadIdx.x}\\
5.\quad \mathrm{lane}\gets\mathrm{tid}\bmod32;\quad\mathrm{warp}\gets\lfloor\mathrm{tid}/32\rfloor\\
6.\quad \text{Private: }\mathrm{vals}[\mathrm{nvals}];\quad\text{shared: }\mathrm{smem}[32]\\[5pt]
\textit{Load once and find the local maximum.}\\
7.\quad \mathrm{lmax}\gets-\infty\\
8.\quad \textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
9.\qquad \mathrm{col}\gets\mathrm{tid}+j\cdot\mathrm{nt}\\
10.\qquad \textbf{if }\mathrm{col}<\mathrm{ncols}:\\
11.\qquad\quad \mathrm{vals}[j]\gets X[\mathrm{row},\mathrm{col}]\\
12.\qquad\quad \mathrm{lmax}\gets\max(\mathrm{lmax},\mathrm{vals}[j])\\
13.\quad \mathrm{rmax}\gets\operatorname{blockReduce}(\mathrm{lmax},\max)\\[5pt]
\textit{Sum exponentials locally; keep the original logits.}\\
14.\quad \mathrm{lsum}\gets0\\
15.\quad \textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
16.\qquad \textbf{if }\mathrm{tid}+j\cdot\mathrm{nt}<\mathrm{ncols}:\\
17.\qquad\quad \mathrm{lsum}\gets\mathrm{lsum}+\exp(\mathrm{vals}[j]-\mathrm{rmax})\\
18.\quad \mathrm{rsum}\gets\operatorname{blockReduce}(\mathrm{lsum},+)\\[5pt]
\textit{Apply the output epilogue using retained logits.}\\
19.\quad \textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
20.\qquad \mathrm{col}\gets\mathrm{tid}+j\cdot\mathrm{nt}\\
21.\qquad \textbf{if }\mathrm{col}<\mathrm{ncols}:\\
22.\qquad\quad Y[\mathrm{row},\mathrm{col}]\gets\exp(\mathrm{vals}[j]-\mathrm{rmax})/\mathrm{rsum}\\[6pt]
\textbf{Helper: }\operatorname{blockReduce}(\mathrm{val},\mathrm{op})\\
23.\quad \mathrm{val}\gets\operatorname{warpFold}(\mathrm{val},\mathrm{op})\\
24.\quad \operatorname{syncBlock}()\qquad\text{(protect scratch reuse)}\\
25.\quad \textbf{if }\mathrm{lane}=0:\quad\mathrm{smem}[\mathrm{warp}]\gets\mathrm{val}\\
26.\quad \operatorname{syncBlock}()\\
27.\quad \textbf{if }\mathrm{warp}=0:\\
28.\qquad \mathrm{val}\gets\mathrm{smem}[\mathrm{lane}]\\
29.\qquad \mathrm{val}\gets\operatorname{warpFold}(\mathrm{val},\mathrm{op})\\
30.\qquad \textbf{if }\mathrm{lane}=0:\quad\mathrm{smem}[0]\gets\mathrm{val}\\
31.\quad \operatorname{syncBlock}();\quad\textbf{return }\mathrm{smem}[0]\\[6pt]
\textbf{Helper: }\operatorname{warpFold}(\mathrm{val},\mathrm{op})\\
32.\quad \textbf{for }\mathrm{offset}=16,8,4,2,1:\\
33.\qquad \mathrm{peer}\gets\operatorname{shuffleDown}(\mathrm{val},\mathrm{offset})\\
34.\qquad \mathrm{val}\gets\mathrm{op}(\mathrm{val},\mathrm{peer})\\
35.\quad \textbf{return }\mathrm{val}\qquad\text{(full warp result in lane0)}
\end{array}
$$

## Key points

- **More threads per row:** width 4096 uses `nvals=4`; width 8192 uses `nvals=8`.
  Thread 0 owns columns `0,1024,2048,...`; at each slot, neighboring threads still
  access adjacent columns. The source's `reg_cnt` counts retained input slots,
  not all physical registers used by the thread. Private values can spill.
- **Three reduction levels:** private values → one scalar per thread → one scalar
  per warp → one scalar per block. Warp 0 reduces the 32 shared partials; a final
  barrier makes the result available to all 1024 threads. Shared memory stores
  reduction scratch, not the entire row.
- **Retained logits:** `vals` is never overwritten with exponentials. The source
  evaluates `exp(vals[j]-rmax)` in the sum loop and again in the output epilogue.
  Generated-code reuse and actual exponential instruction counts require SASS;
  two source evaluations alone do not establish a performance cause.

## Launch and template choices

| Measured width | Threads/row | Warps/row | `reg_cnt` / private input slots | Blocks |
|---:|---:|---:|---:|---|
| 4096 | 1024 | 32 | 4 | `nrows` |
| 8192 | 1024 | 32 | 8 | `nrows` |

- The enclosing dispatch first excludes the small-row warp path. In the measured
  branch, `SoftMaxForward_getBlockSize` caps threads at 1024 and rounds to a whole
  warp. `potential_register_count` computes `ceil(ncols/nt)`; values 1–9 select
  precompiled Reg variants. Thus the final template 4/8 is a value-slot count,
  not a logarithm or the exact row width. Actual `classes` (`ncols`) is runtime.
- Exact measured template forms:
  `cunn_SoftMaxForwardReg<float,float,float,SoftMaxForwardEpilogue,int64_t,4>`
  and the same with final argument 8. The epilogue template selects the output
  formula; `index_t` selects the runtime column-count type.
- Dynamic shared memory is `32 * sizeof(float) = 128 bytes` for the 32 warp
  partials. Each reduction uses three block barriers in the inspected source.
- The helper above is scoped to the measured 32-warp blocks. For smaller blocks,
  the source supplies identity values to unused lanes during the final reduction.
  The source initializes maxima with the lowest finite accumulator value; the
  pseudocode's negative infinity gives the same maximum for finite valid rows.

## Difference from warp softmax

| Property | Previous warp kernel | This Reg kernel |
|---|---|---|
| Row ownership, measured widths | One warp handles one or two rows | One block handles one row |
| Reduction communication | Warp shuffles | Warp shuffles + shared scratch + block barriers |
| Retained data after maximum | Replaced by exponentials | Remains original logits |
| Output formula's input | Retained exponential | Retained logit |

These are algorithm differences to investigate against profiles. They do not
by themselves explain the observed faster/slower cases.

## Sources and evidence

- Recorded H100 / PyTorch 2.11.0+cu130, revision
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
- [Recorded SoftMax.cu](../../results/2026-10-27-offline-profiles/source/pytorch/SoftMax.cu):
  `cunn_SoftMaxForwardReg`, `SoftMaxForwardEpilogue`, `blockReduceWarp`,
  `SoftMaxForward_getBlockSize`, `potential_register_count`, `host_softmax`.
  The Reg kernel body also matches the later downloaded source snapshot.
- [Pinned upstream SoftMax.cu](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu).
- [Pinned block-reduction helper](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/block_reduce.cuh):
  generic `BlockReduce` / `WarpReduce`; `blockReduceWarp` adds the final broadcast.
- Checked all six saved PyTorch captures at rows 37/256/4096 × widths 4096/8192:
  block `(1024,1,1)`, grid `(nrows,1,1)`, final template 4/8. Example captures:
  [37×4096](../../results/2026-10-27-offline-profiles/37x4096/pytorch_raw.csv),
  [37×8192](../../results/2026-10-27-offline-profiles/37x8192/pytorch_raw.csv).
  Offline sources/captures require the separate results download.
- [Previous warp algorithm](11-pytorch-warp-softmax.md) · [Section index](README.md).
