# Profiling review — questions and answers

- **Paused 2026-10-20 after Q20. Resume at Q21 below.** Four core questions remain: Q21–Q24.
- These are concise **corrected revision answers**, not verbatim learner answers or proof of independent mastery. Assistance and corrections are recorded in the [review tracker](../../../progress/projects/02-profiling-review.md).
- Open notes were allowed. Q15 dependent-load reasoning and Q20 latency interpretation merit later transfer checks; throttle classification has not yet been tested.

## Q1 — More copy instructions, similar DRAM bytes

**Question:** The 512-block kernel executes 606,208 global-to-shared copies versus 378,880 for the 128-block kernel (+60%), but reads 2.282 MB versus 2.299 MB from DRAM. Give two explanations and discriminating measurements.

**Answer:** Different instruction widths/participating lanes can change bytes per instruction. Additional requests can also hit caches instead of fetching more DRAM data. Inspect SASS widths/participation and per-level request, hit/miss, and byte/sector counts. Total issued instructions alone cannot distinguish these explanations. These are hypotheses for the measured variants.

- **Follow-up:** three requests for the same 128-byte tile, first misses L2 and next two hit: 384 bytes requested from L2, 256 bytes in hit requests, 128 bytes fetched from DRAM under the simplified assumptions. Hit rate = 2/3 = 66.7%.
- **Pitfall:** requests include hits **and** misses; L2 traffic can rise while DRAM traffic stays flat.
- Reference: [note 18, instruction comparison](../04-performance-comparisons/18-off-on-performance-synthesis.md#checked-prediction-more-instructions).

## Q2 — Separate cache-hit denominators

**Question:** 1,000 sectors reach L1; 60% hit. Every miss reaches L2, where 75% hit. Every L2 miss reaches DRAM; no other traffic. How much reaches L2 and DRAM?

**Answer:** 400 L2 requests, 300 L2 hits, 100 sectors fetched from DRAM. L2's 75% applies to the 400 requests **reaching L2**, not the original 1,000.

- Reference: [note 17, cache reuse](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#cache-hit-rates-and-reuse).

## Q3 — Instructions versus useful floating-point work

**Question:** Same convolution shapes; 17% more executed warp instructions, similar DRAM bytes. Does this establish 17% more useful floating-point computation? What should be inspected?

**Answer:** No. Loads, stores, address arithmetic, branches, and synchronization also contribute instructions. Inspect executed counts by category/opcode, not only throughput percentages. More arithmetic instructions still need interpretation as useful versus redundant work. The saved comparison did not establish an increased accumulation count.

- **Follow-up:** higher bytes/second with identical bytes means a shorter matching measurement interval. More concurrency is a possible cause, not established by those two numbers.
- Reference: [note 18](../04-performance-comparisons/18-off-on-performance-synthesis.md).

## Q4 — Find counters; interpret missing data

**Question:** Which counters measure integer instructions and DRAM bytes? Does an absent integer counter mean zero instructions?

**Answer:**

| Quantity | Counter |
| --- | --- |
| Predicated-on integer **thread** instructions | `smsp__sass_thread_inst_executed_op_integer_pred_on.sum` |
| DRAM read bytes | `dram__bytes_read.sum` |
| DRAM write bytes | `dram__bytes_write.sum` |

- Include reads and writes for total DRAM traffic. Missing may mean uncollected/unsupported, not zero.
- Thread counts and warp counts are different units; do not assume a fixed factor of 32.
- Integer operations include address calculations but are not exclusively address calculations. Trace SASS register uses to determine purpose.
- Reference: [note 18, lookup and integer counters](../04-performance-comparisons/18-off-on-performance-synthesis.md#integer-instructions-versus-address-arithmetic).

## Q5 — Normalize activity

**Question:** Peak 4 packages/cycle; 480 packages over 200 elapsed cycles; at least one accepted in 150 cycles. Throughput relative to peak and active-cycle fraction?

**Answer:** `480/(200*4) = 60%`; `150/200 = 75%`. Active cycles need not use every slot. Average packages per active cycle = 3.2. Peak itself is 4 packages/cycle, not 60%.

- Reference: [note 13](../03-gpu-hardware-and-scheduling/13-nsight-introduction.md#counters-controllers-and-what-a-percentage-measures).

## Q6 — Maximum constituent

**Question:** Constituents are 80%, 40%, 10% of their own peaks. What does a maximum-based summary report? Is that useful FLOP/s utilization?

**Answer:** 80%. Averaging would dilute the busiest resource with less-utilized ones. Percentages have different capacity denominators and can overlap in time. The result is not useful FLOP/s utilization; the busiest constituent is a clue, not proof of the runtime bottleneck.

- Reference: [note 13](../03-gpu-hardware-and-scheduling/13-nsight-introduction.md#why-below-100-and-why-summarize-with-a-maximum).

## Q7 — Three memory summaries

**Question:** Match limits in memory-instruction processing, internal request service, and byte-transfer paths. Does high Mem Busy establish DRAM saturation?

**Answer:**

| Limit | Summary |
| --- | --- |
| SM memory-instruction processing | Mem Pipes Busy |
| Internal memory-unit service | Mem Busy |
| Data-transfer bandwidth | Max Bandwidth |

- No: relevant constituents can reflect activity elsewhere, including L1/TEX. Summaries select the largest normalized constituent rather than summing/averaging unrelated resources.
- A pipeline handles instructions issued by a scheduler; accepting an instruction does not mean completing its memory access that cycle.
- Reference: [note 17](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#three-memory-utilization-summaries).

## Q8 — Active/elapsed cycles and clocks

**Question:** Scheduler observed for 1,000 elapsed cycles, resident warps for 400, instruction issued in 300. What are active/elapsed issue percentages? Can DRAM cycles replace SM cycles?

**Answer:** 75% active and 30% elapsed. No direct substitution: clock domains can differ. Over 1 μs, hypothetical 1 GHz SM and 2 GHz DRAM clocks count 1,000 and 2,000 cycles respectively. Use matching activity, cycles, peak rate, and hardware scope.

- **Follow-up:** DRAM bytes normalized by peak bytes/DRAM cycle requires elapsed **DRAM-unit cycles**.
- Reference: [note 17, clock domains](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#different-hardware-clocks-match-the-denominator).

## Q9 — Resource-limited residency

**Question:** Per SM: 65,536 registers, 96 KiB shared, 64 warps, 16 blocks maximum. Each block needs 16,384 registers, 32 KiB shared, 8 warps. Requirements include rounding/overhead; enough blocks available. Residency and occupancy?

**Answer:** Limits are 4, 3, 8, and 16 blocks respectively. Minimum = **3 blocks**, limited by shared memory; 24 warps; **37.5% theoretical occupancy**.

- Reference: [note 13](../03-gpu-hardware-and-scheduling/13-nsight-introduction.md#compute-throughput-and-resource-accounting).

## Q10 — Capacity versus actual residency

**Question:** Same kernel on 120 SMs: one block per SM, equal duration, negligible startup/tail. Achieved versus theoretical occupancy? What if only 30 SMs get a block?

**Answer:** One block = 8 warps, so achieved = **12.5%** during active cycles. Theoretical stays **37.5%**. With 30 occupied SMs, achieved occupancy remains 12.5% under these assumptions; it does not reveal the 90 idle SMs.

- **Follow-up:** two blocks per active SM = 16 warps = 25% achieved occupancy.
- Reference: [note 13](../03-gpu-hardware-and-scheduling/13-nsight-introduction.md#same-achieved-occupancy-different-performance).

## Q11 — Eligibility distribution versus issue rate

**Question:** Scheduler issues at most one instruction/cycle and always issues when a warp is eligible. A has one eligible every cycle; B has four eligible for 25 cycles, zero for 75. Compare averages/rates.

**Answer:** Both average one eligible warp. A issues 100 instructions/100 cycles = **1/cycle**; B issues 25/100 = **0.25/cycle**. Four eligible warps offer choices, not four issue slots.

- **Follow-up:** two eligible for 60 cycles, zero for 40: average 1.2 eligible warps; 60 issued instructions = 0.6/cycle.
- Reference: [note 16, formulas](../03-gpu-hardware-and-scheduling/16-latency-hiding.md#formulas-readiness-versus-instruction-issue).

## Q12 — All versus Not-Issued samples

**Question:** Dispatch Stall has 220 All/5 Not-Issued samples; Wait has 380/370. What does the filter mean, and which suggests other warps often kept issuing?

**Answer:** Not-Issued means the **scheduler issued no instruction** that cycle. Dispatch Stall more strongly suggests other warps covered the sampled warp's wait. Wait deserves investigation for idle issue opportunities. Samples are not exact lost-cycle counts; separate passes prevent exact subtraction.

- Reference: [note 16, lookup](../03-gpu-hardware-and-scheduling/16-latency-hiding.md#not-selected-and-finding-stall-states-in-nsight).

## Q13 — Dependency versus actual stall

**Question:** Global load into R1, independent arithmetic, then consume R1. Must the consumer stall? If pending, which state, and must the scheduler idle?

**Answer:** No stall if R1 is ready when needed. If still pending from the L1TEX operation, expect Long Scoreboard. Another eligible warp can issue, so scheduler idleness is not necessary.

- Reference: [note 16, scoreboards](../03-gpu-hardware-and-scheduling/16-latency-hiding.md#qa-scoreboard-long-scoreboard-and-short-scoreboard).

## Q14 — State classification

**Question and answer:**

| Situation | State |
| --- | --- |
| Waiting for fixed-latency arithmetic result | Wait |
| Waiting for shared-load result through MIO | Short Scoreboard |
| Eligible but another warp chosen | Not Selected |
| Waiting for other warps at block synchronization | Barrier |
| Ready instruction held back by dispatch conflict | Dispatch Stall |

- **Not Selected** is a selection state, not an unavailable-operand stall. Throttle states were planned but not tested in this question.
- **Additional questions:** No Instructions = waiting for fetch selection or instruction-cache miss; IMC Miss = waiting for Immediate Constant Cache operand miss. Neither means the warp has finished.
- **Where:** Details → Warp State Statistics; Source/SASS → Warp Stall Sampling; Raw → `smsp__pcsamp_warps_issue_stalled_` when collected.
- Reference: [note 16](../03-gpu-hardware-and-scheduling/16-latency-hiding.md#no-instructions-versus-imc-miss), [note 15, constants](../03-gpu-hardware-and-scheduling/15-constants-and-instruction-reissue.md).

## Q15 — Low DRAM plus Long Scoreboard

**Question:** DRAM 3% of peak elapsed, frequent Long Scoreboard Not-Issued samples, often zero eligible warps, 25% occupancy. Leading hypothesis and alternative measurement?

**Answer:** Memory-operation latency with insufficient hiding is a leading hypothesis: resident warps may lack ready instructions and independent requests. Low DRAM bandwidth does not rule out other memory-system bottlenecks; inspect L1/TEX and L2 throughput breakdowns for alternative saturation. No exact root cause is established by these summaries.

- **Follow-up taught:** if each next-load address needs the previous load result, few requests may be in flight. Long waits can coexist with low bytes/second.
- **Dependent-load example:** load `token_id`, then load `embeddings[token_id, 0]`. Contrast independent `X[i]` and `Y[i]` loads. This prerequisite was introduced during the question and then explained; independent transfer remains unassessed.
- **Source/SASS:** trace load-result registers into later address calculations; consumer stall samples add waiting evidence. Static dependency alone does not prove every execution/warp stalled.
- **Resident:** admitted onto an SM with execution resources/state retained. **Eligible:** next instruction can issue now. Eligibility can change repeatedly without losing residency.
- **L1TEX in Long Scoreboard:** global/local/surface/texture operation path, not proof of an L1 hit or DRAM miss. Shared-load dependency is generally Short Scoreboard.
- Reference: [note 16, dependent loads](../03-gpu-hardware-and-scheduling/16-latency-hiding.md#dependent-loads-the-next-address-needs-the-previous-result), [note 17](../03-gpu-hardware-and-scheduling/17-dram-throughput.md).

## Q16 — Prefetching versus occupancy

**Question:** Same correct workload: registers/thread 64→96, occupancy 50%→25%, matching issue activity 40%→65%, reliable runtime 100→80 μs. Explain and decide whether to keep.

**Answer:** Early loads and independent computation can overlap, making remaining warps ready more often. More live register values can reduce residency. Keep for this workload: correct results with **20% lower runtime**. Prefetching may reduce waits but does not guarantee their elimination; increased activity alone would not prove a speedup.

- Reference: [note 16](../03-gpu-hardware-and-scheduling/16-latency-hiding.md).

## Q17 — Block allocation versus SRAM partition

**Question:** Toy 256 KiB SRAM split 128 shared/128 L1. Two blocks request 32 KiB each, ignoring overhead. Allocated, unused, and L1 effect?

**Answer:** 64 KiB allocated; 64 KiB unused within shared partition. L1 stays 128 KiB; unused shared capacity does not automatically become L1. Per-block allocation differs from per-SM configured split. Registers/warp limits affect resident block count, a separate question.

- Reference: [note 17](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#shared-sram-l1-cache-versus-shared-memory).

## Q18 — Reuse across SMs

**Question:** Two blocks read the same tile through ordinary cacheable loads, entries retained. Which cache can serve same-SM/cross-SM reuse? Does block 2 automatically read block 1's shared copy?

**Answer:** Same-SM reuse can hit L1; cross-SM reuse can hit shared L2. Ordinary shared allocations belong to blocks; the second block's global load does not automatically access the first block's allocation. It must populate its own shared tile if required.

- Reference: [note 17](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#cache-hit-rates-and-reuse).

## Q19 — Bank conflicts and broadcast

**Question:** 32 lanes, 32 banks, aligned float array, bank = index mod 32, one distinct 4-byte word/bank/service cycle. Compare `s[4*lane]` and all lanes reading `s[7]`.

**Answer:** Stride 4 uses **8 banks**, **4 distinct words per used bank**, **4 service steps**. Bank 0 gets indices 0,32,64,96 from lanes 0,8,16,24. All lanes reading the same `s[7]` can use **broadcast**, one service step in this model; no shuffle is required.

- **Follow-up:** `s[8*lane]` uses 4 banks, 8 distinct words each, 8 steps.
- Reference: [note 17](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#worked-example-one-warp-scalar-shared-loads).

## Q20 — Service rate versus latency

**Question:** Pipeline accepts one 128-byte package/cycle, latency four cycles, clock 1 GHz, enough independent work. Completion rate/throughput? Do two bank-service steps necessarily double total latency?

**Answer:** After fill, one completion/cycle; **128 GB/s**. No necessary doubling of whole-request latency: only the service work is doubled in the example.

| Separate toy latency example | Without conflict | With conflict |
| --- | ---: | ---: |
| Other stages | 6 cycles | 6 cycles |
| Bank service | 1 cycle | 2 cycles |
| Total | 7 cycles | 8 cycles |

- Assumes unchanged other stages and no overlap between these portions. Real timing also involves overlap/queueing. Not measured H100 timing.
- Reference: [note 17, added worked example](../03-gpu-hardware-and-scheduling/17-dram-throughput.md#extra-service-work-versus-total-completion-latency).

## Resume here — Q21, unanswered

**Question:** GPU has 132 SMs. Saved captures:

| Measurement | Off | On |
| --- | ---: | ---: |
| Blocks | 32 | 128 |
| Threads/block | 128 | 128 |
| Achieved occupancy | ~6.25% | ~6.25% |
| Active-cycle issue percentage | ~72.4% | ~65.1% |
| Profiled duration | ~74 μs | ~30 μs |

1. How can On be faster despite similar occupancy and lower active-cycle issue activity?
2. Do these observations prove increased SM coverage explains all the speedup? Why?

**Answer pending:** resume with the learner's answer, not a fresh review from Q1.

- Reference: [note 18](../04-performance-comparisons/18-off-on-performance-synthesis.md).
- Remaining planned questions: **Q22** 128/512-block work/runtime; **Q23** correctness and timing acceptance; **Q24** unfamiliar report diagnosis. Not yet asked or answered.
