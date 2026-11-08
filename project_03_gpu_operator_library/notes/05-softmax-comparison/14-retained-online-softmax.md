# 14 — Algorithm: our retained online softmax

**Recorded kernel:** `softmax_online_registers<128, ncols>`, FP32, one 128-thread
block (four warps) per row. Each thread loads its inputs once, keeps the original
logits, and computes an online **(maximum, shifted sum)** summary.

## Legend

| Name | Meaning |
|---|---|
| `X`, `Y`; `nrows`, `ncols` | Input/output matrices; dimensions |
| `nt`, `nvals` | Threads per block; retained values per thread |
| `tid`, `lane`, `warp` | Thread index; lane index; warp index |
| `row`, `j`, `x` | Row assigned to the block; private value slot; next logit |
| `vals` | Private retained original logits |
| `p`, `a`, `b`, `peer` | Summary pairs, each with fields `max` and `sum` |
| `hi`, `total` | Maximum and shifted sum of a merged pair |
| `smem` | Four shared summary pairs, one per warp |

For any group of processed values, `p.max` is their maximum and
`p.sum` is the sum of `exp(value - p.max)`. The sum must be rescaled whenever
its reference maximum changes. The empty group is `(-infinity, 0)`.

**Scope:** finite inputs; `ncols` is a positive multiple of 128 and a supported
benchmark specialization. Both `nt` and the exact `ncols` are compile-time
parameters. The host uses runtime `nrows` for the grid; kernel arguments are
only `X,Y`. Matrix indexing means flat offset `row*ncols+col`.

$$
\begin{array}{l}
\textbf{Algorithm 4: retained online softmax}\\[3pt]
\textbf{Host inputs: }X,Y,\mathrm{nrows},\mathrm{ncols},\mathrm{stream}\\
\textbf{Kernel runtime: }X,Y\qquad\textbf{Template: }\mathrm{nt},\mathrm{ncols}\\[6pt]
\textbf{Host: select the exact-width specialization}\\
1.\quad\mathrm{nt}\gets128;\quad\mathrm{nvals}\gets\mathrm{ncols}/\mathrm{nt}\\
2.\quad\mathrm{block}\gets(128,1,1);\quad\mathrm{grid}\gets(\mathrm{nrows},1,1)\\
3.\quad\text{Launch softmax\_online\_registers}\langle128,\mathrm{ncols}\rangle.\\[6pt]
\textbf{Device: retain inputs and build a local online pair}\\
4.\quad\mathrm{row}\gets\mathrm{blockIdx.x};\quad\mathrm{tid}\gets\mathrm{threadIdx.x}\\
5.\quad\mathrm{lane}\gets\mathrm{tid}\bmod32;\quad\mathrm{warp}\gets\lfloor\mathrm{tid}/32\rfloor\\
6.\quad\text{Private: }\mathrm{vals}[\mathrm{nvals}];\quad\text{shared pairs: }\mathrm{smem}[4]\\
7.\quad\textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
8.\qquad\mathrm{vals}[j]\gets X[\mathrm{row},\mathrm{tid}+j\cdot\mathrm{nt}]\\
9.\quad p\gets(\mathrm{vals}[0],1)\\
10.\quad\textbf{for }j=1,\ldots,\mathrm{nvals}-1:\\
11.\qquad x\gets\mathrm{vals}[j]\\
12.\qquad\textbf{if }x\leq p.\mathrm{max}:\\
13.\qquad\quad p.\mathrm{sum}\gets p.\mathrm{sum}+\exp(x-p.\mathrm{max})\\
14.\qquad\textbf{else}:\\
15.\qquad\quad p.\mathrm{sum}\gets p.\mathrm{sum}\cdot\exp(p.\mathrm{max}-x)+1\\
16.\qquad\quad p.\mathrm{max}\gets x\\[5pt]
\textit{Merge thread pairs into one pair per warp, then one per row.}\\
17.\quad p\gets\operatorname{warpPair}(p)\\
18.\quad\textbf{if }\mathrm{lane}=0:\quad\mathrm{smem}[\mathrm{warp}]\gets p\\
19.\quad\operatorname{syncBlock}()\\
20.\quad\textbf{if }\mathrm{warp}=0:\\
21.\qquad p\gets\begin{cases}\mathrm{smem}[\mathrm{lane}],&\mathrm{lane}<4\\(-\infty,0),&\text{otherwise}\end{cases}\\
22.\qquad p\gets\operatorname{warpPair}(p)\\
23.\qquad\textbf{if }\mathrm{lane}=0:\quad\mathrm{smem}[0]\gets p\\
24.\quad\operatorname{syncBlock}()\\[5pt]
\textit{Normalize retained logits using the complete row pair.}\\
25.\quad p\gets\mathrm{smem}[0]\\
26.\quad\textbf{for }j=0,\ldots,\mathrm{nvals}-1:\\
27.\qquad Y[\mathrm{row},\mathrm{tid}+j\cdot\mathrm{nt}]\gets\exp(\mathrm{vals}[j]-p.\mathrm{max})/p.\mathrm{sum}\\[6pt]
\textbf{Helper: }\operatorname{warpPair}(p)\\
28.\quad\textbf{for }\mathrm{offset}=16,8,4,2,1:\\
29.\qquad\mathrm{peer}.\mathrm{max}\gets\operatorname{shuffleDown}(p.\mathrm{max},\mathrm{offset})\\
30.\qquad\mathrm{peer}.\mathrm{sum}\gets\operatorname{shuffleDown}(p.\mathrm{sum},\mathrm{offset})\\
31.\qquad p\gets\operatorname{merge}(p,\mathrm{peer})\\
32.\quad\textbf{return }p\qquad\text{(complete warp summary in lane 0)}\\[6pt]
\textbf{Helper: }\operatorname{merge}(a,b)\\
33.\quad\textbf{if }b.\mathrm{sum}=0:\quad\textbf{return }a\\
34.\quad\textbf{if }a.\mathrm{sum}=0:\quad\textbf{return }b\\
35.\quad\mathrm{hi}\gets\max(a.\mathrm{max},b.\mathrm{max})\\
36.\quad\mathrm{total}\gets a.\mathrm{sum}\cdot\exp(a.\mathrm{max}-\mathrm{hi})+b.\mathrm{sum}\cdot\exp(b.\mathrm{max}-\mathrm{hi})\\
37.\quad\textbf{return }(\mathrm{hi},\mathrm{total})
\end{array}
$$

## Key points

- **Online update:** if the next value stays below the maximum, add its shifted
  exponential. If it raises the maximum, rescale the old sum and add 1 for the
  new maximum's contribution. The old maximum must be used before updating it.
- **Pair reduction:** maxima and sums travel together. Sums from different
  groups have different reference maxima, so `merge` rescales both to `hi`
  before adding. Each shuffle stage exchanges two floats, then merges the pair.
  Only lane 0's complete warp result is consumed.
- **Retained:** `vals` always holds original logits, including through both
  block barriers. Output reuses them without loading `X` again in the source.
  This is private storage, not a guarantee of physical registers; wide rows can
  spill. Shared storage is only four pairs (32 bytes), not a row buffer.

## Small update example

Suppose `p=(5, 1+exp(-1))`, summarizing values 5 and 4:

- Next value 3: maximum stays 5; add `exp(3-5)` to the sum.
- Next value 7 instead: new maximum 7; old sum becomes
  `(1+exp(-1))*exp(5-7)`, then add 1.

`merge` applies the same rescaling to two groups. Empty checks avoid evaluating
`exp(-infinity - (-infinity))` when the final warp is padded with empty pairs.

## Mapping and compile-time choices

| Width | Threads/row | Values/thread | Shared pairs |
|---:|---:|---:|---:|
| 128 | 128 | 1 | 4 |
| 1024 | 128 | 8 | 4 |
| 4096 | 128 | 32 | 4 |
| 16384 | 128 | 128 | 4 |
| 131072 | 128 | 1024 | 4 |

- At width 1024, thread 0 keeps columns `0,128,...,896`; neighboring lanes load
  neighboring columns at each slot. All 128 threads collectively cover one row.
- Width is **exactly specialized**, not a rounded capacity or runtime loop limit.
  The bridge selects a precompiled `softmax_online_registers<128,dims>` case.
  Supported widths are 128/512/1024/2048/4096/8192/16384/32768/65536/131072;
  none needs column bounds guards because each is divisible by 128.
- No separate local maximum pass followed by a local denominator pass: the
  conditional recurrence updates both together. After that, the warp/block
  reductions also combine both quantities together, using two block barriers.
- Unlike the non-retained `softmax_online<128>` baseline, this kernel keeps its
  inputs for output. Do not transfer claims about one variant to the other.

## Comparison points for profiling

| Property | PyTorch Warp | PyTorch Reg | PyTorch General | Our retained Online |
|---|---|---|---|---|
| Threads cooperating on a row, measured cases | 32 | 1024 | 1024 | 128 |
| Retained data | Logits, then exponentials | Logits | Small reused chunks | Logits |
| Reduction state | Max and sum separately | Max and sum separately | Max and sum separately | Coupled max/sum pairs |
| Input passes in source | 1 | 1 | 3 | 1 |

The online recurrence and pair merges add rescaling arithmetic; retaining inputs
avoids rereading them but increases private storage. Those are profiling questions,
not a conclusion that online or higher occupancy must be faster. The current
comparison also retains the exact-width versus capacity/runtime-width distinction.

## Is online normalization redundant when inputs are retained?

- **Much of its memory advantage disappears.** Both PyTorch Warp and our
  retained Online load each input once in the source. Online normalization is
  useful for combining streaming groups, but retention already avoids rereading
  the full input here.
- **Warp reuses exponentials:** after finding the row maximum, it overwrites
  retained logits with exponentials, sums them, then divides those same values.
  Our kernel retains logits, performs online rescaling and pair merges, then
  computes exponentials again for output.
- **There is still a tradeoff:** one combined max/sum reduction can reduce
  synchronization stages versus two separate block reductions, at the cost of
  more arithmetic per merge. PyTorch Warp itself needs no block barriers.
- **The current wins do not isolate the recurrence.** Thread mapping and work
  per thread also differ. A useful future control keeps our 128 threads/row and
  retained storage, but uses max → exponential/sum → normalization and retains
  the exponentials. This control has not been implemented or measured.

## Sources and evidence

- [Recorded kernel and merge helper](../../results/2026-10-31-offline-profiles-128k/source/softmax.cu):
  `softmax_online_registers`, `merge_online`, `OnlineState`.
- [Recorded host bridge](../../results/2026-10-31-offline-profiles-128k/source/softmax_sweep_bridge.cu):
  `launch_registers`, exact-width switch and 128-thread launch.
- [Local learning source](../../kernels/cuda/softmax.cu), synchronized with the GPU
  commit on 2026-11-08, now uses `softmax_online_registers<N, dims>` and permits
  other valid whole-warp block sizes. Before that merge, the local copy used
  `Cols` and asserted `N==128`. The recorded benchmark bridge launches 128
  threads; this explanation follows that measured path.
- All 30 saved retained-online profiles show block `(128,1,1)`, grid `(nrows,1,1)`
  and matching exact-width template arguments. Examples:
  [37×1024](../../results/2026-10-27-offline-profiles/37x1024/online_registers_raw.csv),
  [37×16384](../../results/2026-10-31-offline-profiles-128k/37x16384/online_registers_raw.csv).
  Offline source/profile links require the separate results download.
- [Previous General algorithm](13-pytorch-general-softmax.md) · [Section index](README.md).
