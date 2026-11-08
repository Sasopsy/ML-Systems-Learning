# 13 — Algorithm: PyTorch general softmax

**Measured family:** `cunn_SoftMaxForward<4,float,float,float,SoftMaxForwardEpilogue>`
for widths 16384–131072. One1024-thread block handles each row. The source makes
**three input passes**, using a small vector buffer instead of retaining the row.

The algorithm shows finite FP32 inputs, aligned input/output row starts and widths
that are multiples of 4096, matching the main path for these contiguous allocations.
Alignment-prefix, remainder and mismatched-alignment output branches are described
below rather than expanded into the algorithm.

## Legend

| Name | Meaning |
|---|---|
| `X`, `Y`; `nrows`, `ncols` | Input/output matrices; dimensions |
| `nt`, `vec` | Threads per block; adjacent values per vector load/store |
| `tid`, `row` | Thread index in block; row assigned to the block |
| `q`, `j` | Vector-chunk index; element within that vector |
| `v`, `out` | Temporary private input/output vectors, each of length `vec` |
| `lmax`, `lsum` | Thread-local maximum and exponential sum |
| `rmax`, `rsum` | Full row maximum and denominator |
| `smem` | Shared reduction scratch: 32 FP32 entries |

`loadVec(X[row,vec*q])` loads columns `vec*q` through `vec*q+vec-1`.
`storeVec` writes the corresponding output vector. The template `vec` is named
`ILP` in the source. All data/accumulator/output types are `float` here, and
`Epilogue = SoftMaxForwardEpilogue`. `ncols` remains a runtime argument.

$$
\begin{array}{l}
\textbf{Algorithm 3: cunn\_SoftMaxForward — aligned FP32 path}\\[3pt]
\textbf{Host inputs: }X,Y,\mathrm{nrows},\mathrm{ncols},\mathrm{stream}\\
\textbf{Kernel runtime: }X,Y,\mathrm{ncols}\\
\textbf{Template: }\mathrm{vec},\mathrm{scalar\_t},\mathrm{accscalar\_t},\mathrm{outscalar\_t},\mathrm{Epilogue}\\[6pt]
\textbf{Host: after dispatch selects the General family}\\
1.\quad\mathrm{nt}\gets1024;\quad\mathrm{vec}\gets16/\operatorname{sizeof}(\mathrm{float})=4\\
2.\quad\mathrm{block}\gets(\mathrm{nt},1,1);\quad\mathrm{grid}\gets(\mathrm{nrows},1,1)\\
3.\quad\text{Launch with 128 bytes of shared reduction scratch.}\\[6pt]
\textbf{Device: one block per row}\\
4.\quad\mathrm{row}\gets\mathrm{blockIdx.x};\quad\mathrm{tid}\gets\mathrm{threadIdx.x}\\
5.\quad\text{Private: }v[\mathrm{vec}],\mathrm{out}[\mathrm{vec}];\quad\text{shared: }\mathrm{smem}[32]\\[5pt]
\textit{Pass 1: load input and accumulate the local maximum.}\\
6.\quad\mathrm{lmax}\gets-\infty\\
7.\quad\textbf{for }q=\mathrm{tid},\mathrm{tid}+\mathrm{nt},\ldots<\mathrm{ncols}/\mathrm{vec}:\\
8.\qquad v\gets\operatorname{loadVec}(X[\mathrm{row},\mathrm{vec}\cdot q])\\
9.\qquad\textbf{for }j=0,\ldots,\mathrm{vec}-1:\quad\mathrm{lmax}\gets\max(\mathrm{lmax},v[j])\\
10.\quad\mathrm{rmax}\gets\operatorname{blockReduce}(\mathrm{lmax},\max)\\[5pt]
\textit{Pass 2: reload input and accumulate the local exponential sum.}\\
11.\quad\mathrm{lsum}\gets0\\
12.\quad\textbf{for }q=\mathrm{tid},\mathrm{tid}+\mathrm{nt},\ldots<\mathrm{ncols}/\mathrm{vec}:\\
13.\qquad v\gets\operatorname{loadVec}(X[\mathrm{row},\mathrm{vec}\cdot q])\\
14.\qquad\textbf{for }j=0,\ldots,\mathrm{vec}-1:\\
15.\qquad\quad\mathrm{lsum}\gets\mathrm{lsum}+\exp(v[j]-\mathrm{rmax})\\
16.\quad\mathrm{rsum}\gets\operatorname{blockReduce}(\mathrm{lsum},+)\\[5pt]
\textit{Pass 3: reload input, apply the epilogue and store.}\\
17.\quad\textbf{for }q=\mathrm{tid},\mathrm{tid}+\mathrm{nt},\ldots<\mathrm{ncols}/\mathrm{vec}:\\
18.\qquad v\gets\operatorname{loadVec}(X[\mathrm{row},\mathrm{vec}\cdot q])\\
19.\qquad\textbf{for }j=0,\ldots,\mathrm{vec}-1:\\
20.\qquad\quad\mathrm{out}[j]\gets\exp(v[j]-\mathrm{rmax})/\mathrm{rsum}\\
21.\qquad\operatorname{storeVec}(Y[\mathrm{row},\mathrm{vec}\cdot q],\mathrm{out})\\[6pt]
\textbf{Helper: blockReduce — same implementation as Algorithm 2}\\
22.\quad\text{Shuffle-reduce within each warp; synchronize the block.}\\
23.\quad\text{Lane0 of each warp writes its partial result to smem; synchronize.}\\
24.\quad\text{Warp0 shuffle-reduces the 32 partials; thread 0 writes smem[0].}\\
25.\quad\text{Synchronize; every thread returns smem[0].}
\end{array}
$$

## Key points

- **Three passes, small storage:** the source reloads every input for maximum,
  sum and output. `v[4]` is reused for each chunk; it does not hold all values
  assigned to that thread. Three global-memory reads do **not** imply three DRAM
  reads: later accesses can hit caches. Actual loads/instructions require profiles.
- **Vector mapping:** thread 0 loads columns 0–3, thread 1 loads 4–7, etc. At its next
  iteration, thread 0 loads 4096–4099. Neighboring lanes access adjacent 16-byte
  chunks, allowing coalescing. `vec=4` is a load/store width, not `reg_cnt=4`.
- **Local → block:** local max/sum loops feed the same shared-scratch block
  reduction as [Algorithm 2](12-pytorch-register-softmax.md). Shared memory holds
  partials, not the entire row. The source evaluates exponentials in both the sum
  and output passes; this alone does not establish generated instruction counts.

## Width changes the loop count, not this template

| Width | `vec` | Vector chunks/thread/pass | Values/thread/pass |
|---:|---:|---:|---:|
| 16384 | 4 | 4 | 16 |
| 32768 | 4 | 8 | 32 |
| 65536 | 4 | 16 | 64 |
| 131072 | 4 | 32 | 128 |

- All 12 saved captures use the same template, block `(1024,1,1)` and grid
  `(nrows,1,1)`, for row counts 37/256/4096. `classes` (`ncols`) changes at runtime.
- The FP32 dispatch computes `ILP = sizeof(float4)/sizeof(float) = 4`.
  `ilpReduce` performs the first two passes; `WriteFpropResultsVectorized`
  performs the third using `SoftMaxForwardEpilogue`.
- The host tries the register path and eligible shared-memory path before this
  fallback in the inspected branch. Calling this the General family does not
  mean PyTorch always uses it for every shape/dtype/version.

## Scope of the simplified algorithm

- **Alignment:** the kernel computes input/output row-address offsets modulo 16
  bytes. The main vector loop needs aligned addresses. A scalar prefix handles
  an unaligned start; a scalar tail handles incomplete vector-loop batches.
- **Output fallback:** matching input/output alignment offsets permit the
  vectorized output helper; different offsets select the scalar output helper.
  Fresh contiguous benchmark allocations and these widths support the aligned
  case shown; the saved kernel name alone does not prove which branch ran.
- **Finite data:** source max initialization uses the lowest finite accumulator
  value; negative infinity is equivalent for these finite nonempty rows.
- **Storage:** arrays in the algorithm describe temporary source-level state,
  not an exact physical register count. No CUDA compilation or new GPU run is
  implied by this walkthrough.

## Compared with the previous kernels

| Property | Warp | Reg | General |
|---|---|---|---|
| Row owner in measured cases | Warp | Block | Block |
| Data retained between reductions | Logits, then exponentials | Original logits | No full-row retention |
| Input passes in source | 1 | 1 | 3 |
| Main input access per thread | Scalar | Scalar | Vector of 4 floats |
| Cross-warp reduction | No | Yes | Yes |

The extra loads and smaller retained state are a tradeoff to inspect in profiles;
they do not by themselves establish why a particular shape is faster/slower.

## Sources and evidence

- [Recorded SoftMax.cu](../../results/2026-10-31-offline-profiles-128k/source/pytorch/SoftMax.cu):
  `cunn_SoftMaxForward`, `ilpReduce`, `WriteFpropResultsVectorized`,
  `WriteFpropResults`, `SoftMaxForwardEpilogue`, `host_softmax`.
- [Pinned upstream source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu),
  recorded H100 / PyTorch 2.11.0+cu130.
- All 12 saved PyTorch captures at rows 37/256/4096 × widths 16384/32768/65536/131072
  have General template 4 and the block/grid dimensions stated above. Example:
  [37×16384 capture](../../results/2026-10-31-offline-profiles-128k/37x16384/pytorch_raw.csv).
  These offline files require the separate results download.
- [Previous Reg algorithm](12-pytorch-register-softmax.md) · [Section index](README.md).
