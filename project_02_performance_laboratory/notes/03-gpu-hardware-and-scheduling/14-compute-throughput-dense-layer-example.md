# Compute throughput through a dense layer

## Workload and scope

- **Illustrative forward layer:** `Y = X @ W`, with `X[32,128]`, `W[128,64]`, `Y[32,64]`; no bias or activation.
- Each output accumulates 128 products. Total: 2,048 outputs, 262,144 multiply-accumulates, conventionally 524,288 FLOPs.
- **Teaching implementation:** one thread per output, using scalar FMA operations. This is not a claim about PyTorch/cuBLAS or our cuDNN kernel's implementation.

## Work behind one output

1. Determine addresses of an input and a weight.
2. Load values into registers.
3. Accumulate `acc = x * w + acc`.
4. Repeat for the next input feature; store the final output.

| Activity | Relevant machinery |
| --- | --- |
| Address/index calculations | Integer/address instruction paths |
| Loads and final store | Load/store path and memory hierarchy |
| Multiply-accumulate | FMA arithmetic path in this example |
| Select an instruction to issue | Warp scheduler |
| Observe events or active cycles | Performance counters |

- **Dependencies:** an FMA needs both operands and its previous accumulator value.
- **Load latency is not bandwidth:** a warp can wait for a request while aggregate HBM traffic stays low; a request may also be served by cache.

## A deliberately simplified six-cycle window

- One toy scheduler, at most one instruction issued/cycle; two resident warps A/B.
- Input operands and prior accumulators are ready. Only each weight load is missing.
- Weight-load latency is exactly four cycles; no other independent instructions are ready during the waits. This is an invented timing model, not H100 latency.

| Cycle | Issued instruction |
| --- | --- |
| 0 | A loads weight, ready at cycle 4 |
| 1 | B loads weight, ready at cycle 5 |
| 2 | None: both next instructions wait for weights |
| 3 | None: both next instructions wait for weights |
| 4 | A issues FMA |
| 5 | B issues FMA |

- **Issue activity:** 4 issued instructions / 6 opportunities = 66.7%.
- **Arithmetic issue rate:** 2 FMA instructions / 6 cycles; if the toy FMA path accepts one/cycle, this is 33.3% of that rate. This is not the real report's FMA-active-cycle metric.
- The warps remain resident while waiting. **Residency and readiness differ.** Another eligible warp could fill an unused issue opportunity; having more resident warps alone does not guarantee one is ready.
- This window stops at instruction issue, not completion of the layer.

## Coverage and summary metrics

- With the teaching mapping and 128 threads/block, 2,048 output threads give 16 blocks. On the recorded 132-SM device, those blocks can occupy at most 16 SMs simultaneously.
- That is a coverage constraint, not a prediction of an exact throughput percentage.
- **Nsight SM throughput:** maximum of selected normalized constituent measurements. Each constituent has its own hardware-instance aggregation and denominator.
- Our real basic on008 report: issue 57.77%, FMA-active 35.19%, ALU-active 31.18%, LSU-instruction 17.34%; aggregate 57.77%. Do not sum them or equate the aggregate to useful FLOP/s.
- **Why maximum:** a 100%-used arithmetic path and unused tensor path can coexist. Averaging them obscures the saturated resource; the maximum exposes it for investigation without proving the bottleneck.

## Connection to the recorded convolution

- `conv_fprop.py`: `X[32,64,16,16]`, `W[64,64,3,3]`, padding 1, output `[32,64,16,16]`.
- Each interior output sums `64 * 3 * 3 = 576` products. Similar operand/accumulator dependencies exist, but cuDNN can tile, reuse values, and schedule differently.
- Off/on counter differences are observed; exact contributions from coverage, dependency waits, and execution-resource limits remain unproven.

## Sources and next check

## First warp-state labels: Wait and Dispatch Stall

### Issued versus executed instruction normalization

- **Warp-cycle:** one resident warp observed for one cycle; two resident warps over ten cycles contribute twenty warp-cycles.
- **Per issued instruction:** accumulated warp-cycles divided by issue count (on Hopper, single-issue scheduler issue-active cycles correspond to issue count).
- **Per executed instruction:** accumulated warp-cycles divided by executed instruction count. Neither ratio is an individual instruction's issue-to-completion latency.
- On035 raw ratios: `smsp__average_warp_latency_per_inst_issued.ratio = 1.531242`; `smsp__average_warps_active_per_inst_executed.ratio = 1.532910`. Both display **1.53** after rounding.
- Recorded warp-level counts: issued **18,116,865**, executed **18,097,152**. A small count difference is observed; exact cause not established.
- Issue can be retried: NVIDIA documents an instruction reissue after an immediate constant-cache miss. This is a possible mechanism, not an attribution of our count difference; do not equate hardware reissue with profiler kernel replay.
- [NVIDIA normalization explanation](https://forums.developer.nvidia.com/t/why-is-stall-reason-metric-calculated-based-on-issue-active-cycles-in-the-warp-state-statistics/259048).

- **On035 normalized breakdown:** Wait 0.179721; Dispatch Stall 0.101297. These are per-issue-active state ratios in the saved CSV, not percentages or raw PC-sampling counts. The Warp Cycles Per Issued Instruction presentation expresses normalized warp-cycle contributions.
- **Wait:** fixed-latency execution dependency. Example: an FMA needs the accumulator produced by a preceding FMA. This named category is not a general label for all memory waits.
- **Dispatch Stall:** an instruction is ready to issue, but dispatch is held back by another conflict or event. Exact underlying conflict is not established by the label alone.
- Earlier eligible/issued comparison motivates investigation; it does not exclude dispatch constraints.
- **Pitfall:** a stalled warp does not prove the whole scheduler was idle. Inspect evidence restricted to no-issue cycles before assigning critical performance impact.
- [NVIDIA warp-state definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).

- **First real scheduler reading (expanded on035):** active warps/scheduler 1.00, eligible warps/scheduler 0.65, issued warp/scheduler 0.65 (displayed rounded values).
- **Resident/active:** warp state/resources are present. **Eligible:** next instruction is ready for issue. **Issued:** selected to issue an instruction.
- These are averages, not fractional physical warps. Active-cycle scheduler metrics have a different denominator from elapsed-cycle SM throughput; compare like metrics and report IDs.
- The eligible and issued readings motivate investigating readiness, but do not identify a particular memory/dependency stall cause on their own.

- [Nsight scheduler descriptions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules).
- [Nsight metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure).
- [Previous counter and occupancy notes](13-nsight-introduction.md).
- **Next:** interpret actual resident, eligible and issued warp measurements; explanation here is not independent mastery evidence.
