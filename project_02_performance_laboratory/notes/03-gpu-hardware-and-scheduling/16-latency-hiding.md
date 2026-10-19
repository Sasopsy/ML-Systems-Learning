# Latency hiding

## Core idea

- **Latency hiding:** issue independent work while another instruction's result is pending, reducing unused issue opportunities.
- The original result can take just as long to arrive. Aggregate instruction throughput can improve because other work overlaps the wait.
- Already-resident warps keep their registers and execution state on chip; the scheduler can choose a different eligible warp without saving/restoring those registers to external memory.

## Toy arithmetic timeline

- One scheduler can issue one warp instruction/cycle; arithmetic pipeline accepts one/cycle; result latency is four cycles.
- Each warp has a dependent accumulator chain; operands are otherwise ready. These are illustrative assumptions, not timings measured from cuDNN.

| Cycle | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Only warp A available | A0 | — | — | — | A1 | — | — | — |
| Four independent warps | A0 | B0 | C0 | D0 | A1 | B1 | C1 | D1 |

- A1 still waits until cycle 4 in both cases. The other warps use earlier issue opportunities.
- In this repeating toy schedule, issue rate changes from 0.25 to 1 instruction/cycle. This is not a measured kernel speedup.
- For one thread lane, the dependency can be `acc = x0*w0 + acc`, followed by `acc = x1*w1 + acc`. Each warp works on its own outputs/registers.

## Requirements and pitfalls

- Another warp must be eligible on that scheduler; required dispatch/execution capacity must also be available.
- **Occupancy provides potential work; readiness determines whether it can issue.** All resident warps may wait together.
- A pipelined unit can accept independent work before earlier results finish. Result latency and instruction acceptance rate are different.
- Independent instructions within one warp can also cover waits; extra independent accumulators may require more registers.
- More warps cannot exceed a saturated resource's capacity. High occupancy alone does not guarantee improved throughput.
- On035's approximately one active warp/scheduler suggests limited alternatives, not proof that every wait causes an idle cycle.
- All stall samples can include waits hidden by another warp's issue. Not-Issued samples focus on cycles without scheduler issue; they still do not directly predict a removable fraction of kernel time.

## Sources and next exercise

## On035: all samples versus no-issue cycles

- **Find the counters:** open the expanded on035 report, select **Raw**, and filter by `smsp__pcsamp_warps_issue_stalled_wait` or `smsp__pcsamp_warps_issue_stalled_dispatch_stall`. Compare each base metric with its `_not_issued` version.
- **Metric help:** **Profile > Metric Details** shows descriptions and raw values; its search matches the start of the metric name.
- The default **Details > Warp State Statistics** chart and these raw PC-sampling counts are different presentations. Basic reports may lack the expanded counters.
- **Wait:** fixed-latency execution dependency, such as needing a previous accumulator result. **Dispatch Stall:** an instruction is ready to issue but dispatch is held back by another conflict/event; exact cause not established.

| Sampled state | All samples | Samples on cycles with no instruction issued |
| --- | ---: | ---: |
| Wait | 382 | 376 |
| Dispatch Stall | 223 | 5 |

- Re-read from saved on035 raw CSV; these are **sample counts**, not the earlier normalized warp-cycles/instruction values.
- The second column of counts filters by scheduler issue activity; it is not another kernel execution mode or an optimization result.
- Counts identify associations for investigation, not exact lost-cycle totals or predicted speedups. Learner interpretation requested before drawing the case conclusion.
- **"Base" in this lesson:** informal shorthand for the metric without `_not_issued`; it is not a baseline run. Prefer **all-samples metric**.
- `_not_issued` means the **scheduler issued no instruction** during that sampled cycle, not merely that the sampled warp did not issue. A Wait sample can count in the all-samples metric while another warp issues; it then does not count in the no-issue filter.
- **Dispatch explanation limit:** NVIDIA describes internal conflicts/events holding back dispatch but this label does not identify the precise conflict. Readiness does not guarantee successful dispatch. Do not attribute it specifically to a full arithmetic pipeline; other throttle categories exist.

### Concrete documented dispatch-stall cause

- **Register-file bandwidth/bank contention:** input values have been computed, but the hardware still must read them out of registers for execution. Limited register-file access capacity can delay dispatch.
- Example instruction: `R0 = R1 * R2 + R3`. Distinguish R2 not being produced yet (**fixed-latency dependency Wait**) from R1/R2/R3 being valid but competing for register read bandwidth (**possible Dispatch Stall**).
- NVIDIA engineers confirmed register-file read/write bandwidth as the cause in an RTX 3080 IMAD/IADD3 case; a separate explanation identifies register-bank contention during operand collection. These are documented mechanisms, not confirmation for our H100 capture.
- **Capacity versus bandwidth:** registers allocated per thread limit residency; speed of reading/writing register values can limit dispatch. They are different constraints.
- [Confirmed register-bandwidth case](https://forums.developer.nvidia.com/t/what-cause-dispatch-stall-how-to-avoid-it/233021/11).
- [NVIDIA operand-collection explanation](https://forums.developer.nvidia.com/t/understanding-tensor-pipe-throughput-and-throttle-stalls/355572/4).

### Why two bank conflicts can have different stall labels

- **Root cause versus observed state:** bank contention is a physical mechanism; the warp-state label describes the point preventing instruction progress. They are not a one-to-one taxonomy.
- **Register-file example:** arithmetic instruction's register values are valid, but operand collection is delayed by register-bank contention before dispatch to the execution unit: possible **Dispatch Stall**.
- **Shared-memory example:** `r1 = shared[index]` has already dispatched. Shared-memory bank conflicts serialize servicing that load. A following arithmetic instruction needs r1, so the scheduler tracks its pending load result through a scoreboard: possible **Short Scoreboard**.
- A **scoreboard** tracks outstanding instruction results and prevents dependent instructions from consuming values before they are ready. A shared-memory dependency is not generally the fixed-latency **Wait** category.
- The shared-memory bank conflict occurs inside the earlier load's memory service, while the sampled dependency stall can be associated with the later consumer instruction. Distinguish the cause from where its effect is observed.
- Shared-memory bank conflicts can have other downstream effects too; do not claim each physical cause has exactly one stall label.

- [CUDA execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html).
- [CUDA latency hiding and occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
- **Transfer:** with the same toy assumptions but only two independent warps, write the issue sequence for cycles 0–3. Answer not supplied here.

## Operand collection and the public definition

- **Operand collection:** obtain the actual input values from register storage for the execution unit. For one lane, `R0 = R1 * R2 + R3` with values 2, 3, 4 requires those numbers at the arithmetic inputs. Computed and stored does not mean already delivered.
- **Definition versus example:** NVIDIA defines Dispatch Stall broadly as dispatcher conflicts/events holding back a ready instruction. The definition does not restrict the category to registers or enumerate every cause. Register-bank contention is a documented example, not an exhaustive definition.

## Connecting stalls to memory throughput

- Low DRAM utilization does not ensure instructions are ready: arithmetic dependencies can leave issue opportunities unused even with spare memory bandwidth.
- On035's Wait samples (382 all / 376 no-issue), together with roughly one active warp/scheduler, motivate investigating dependency latency and limited independent work. They do not prove an exact instruction-level cause or removable runtime fraction.
- See the [combined memory conclusion](17-dram-throughput.md#current-case-supported-conclusion-versus-hypothesis).

## Q&A: why can a resident warp be ineligible?

- **Resident:** assigned to an SM sub-partition with execution state/resources retained. **Eligible:** its next instruction is ready to issue now. Eligibility can change cycle by cycle without changing residency.
- **Admitted onto an SM:** a resident warp has been assigned execution resources and has not finished. Its retained state includes thread register values, execution position, lane/control information, and bookkeeping for pending results. These are conceptual examples, not a literal hardware layout.
- **Residency persists while eligibility changes:** eligible → issues a load → next instruction needs its pending result, so ineligible → result arrives and other issue requirements are satisfied, so eligible again. The warp need not leave/re-enter the SM during these transitions.
- An outstanding load does **not automatically** make the warp ineligible: a ready independent next instruction may still issue.
- Checks include an available decoded instruction, resolved dependencies, synchronization permission, and an available execution/queue path. Data readiness alone is insufficient.

| Blocking condition | Example | Nsight category |
| --- | --- | --- |
| Next instruction unavailable | Waiting for instruction fetch/cache | No Instructions |
| Input not ready | Prior arithmetic or memory result | Wait / Short or Long Scoreboard |
| Synchronization incomplete | Warp reaches `__syncthreads()` before others | Barrier |
| Pipeline cannot accept work | All operands ready, execution pipe unavailable | Math Pipe Throttle |
| Instruction queue full | Cannot enqueue another memory operation | LG / MIO / TEX Throttle |
| Control-flow bookkeeping | Branch target/PC update pending | Branch Resolving |
| Explicitly blocked or sleeping | All threads blocked/yielded/asleep | Sleeping |

- **Not Selected is different:** the warp was eligible, but the scheduler chose another warp. Not issuing does not necessarily mean ineligible.
- Execution pipes can be pipelined: an older instruction being in flight does not automatically prevent accepting a new one.
- **Eligibility checklist:** next instruction available → inputs ready → synchronization permits progress → required execution/queue path available. These are conceptual checks, not a specification of hardware check order.
- **Scheduling distinction:** first ask whether a warp could issue (eligibility), then whether the scheduler chose it (selection). A ready value alone does not guarantee eligibility.
- Dispatch Stall remains the documented ready-to-issue-but-held-back case; do not replace distinct throttle categories with a generic dispatch explanation.
- [NVIDIA scheduler model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#hardware-model), [stall interpretation](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

### Deep-learning example: shared tile barrier

```cpp
tile[...] = input[...];  // cooperative loading by block threads
__syncthreads();
acc += tile[...] * weight;
```

- Warp A reaches the barrier before warp B finishes its part of the tile load.
- A remains **resident**, retaining its registers and execution state, but cannot proceed past the barrier until synchronization permits it.
- This can happen even when A's own preceding work is complete. It is a **Barrier** condition, not simply a missing arithmetic input or an unselected eligible warp.
- This sketch illustrates synchronization; it is not a complete kernel or a diagnosis of the captured convolution.

## Formulas: readiness versus instruction issue

For one scheduler over **C observed cycles**, let **E(t)** be its eligible-warp count in cycle t and **I(t)** the number of instructions it issues that cycle.

| Quantity | Formula | Units |
| --- | --- | --- |
| Eligible-warp cycles | `sum E(t)` | warp-cycles |
| Average eligible warps | `sum E(t) / C` | warps |
| Total issued instructions | `sum I(t)` | instructions |
| Average issue rate | `sum I(t) / C` | instructions/cycle |
| Issue-active percentage | `100 × cycles with I(t) > 0 / C` | % |

- **For these examples only:** at most one instruction/cycle, no other issue restrictions, and the scheduler issues whenever an eligible warp exists. Then `I(t) = 1` when `E(t) > 0`, otherwise `0`.
- Several warps can be **eligible simultaneously**, even though only one issues. Unselected eligible warps do not contribute extra issued instructions that cycle.
- With this one-instruction limit, issue-active percentage equals `100 × average issue rate`. This numerical equivalence is not general for hardware that can issue multiple instructions/cycle.

| Ready pool over 100 cycles | Average eligible warps | Issued instructions | Issue rate | Issue-active % |
| --- | --- | ---: | --- | ---: |
| 1 eligible in every cycle | `(1×100)/100 = 1` | 100 | 1 instruction/cycle | 100% |
| 4 eligible for 25 cycles, 0 for 75 | `(4×25 + 0×75)/100 = 1` | 25 | 0.25 instructions/cycle | 25% |
| 2 eligible for 50 cycles, 0 for 50 | `(2×50 + 0×50)/100 = 1` | 50 | 0.5 instructions/cycle | 50% |

- **Why multiply 4×25?** It counts readiness: four eligible warps observed in each of 25 cycles = 100 eligible-warp cycles. Issued work is only `1×25 = 25` instructions.
- **Pitfall:** `50 instructions / 100 cycles = 0.5 instructions/cycle`; the numerator is not “50 instructions per cycle.”
- **Takeaway:** an average ready-pool size of one does not mean ready work exists every cycle. Inspect issue activity and cycles with no eligible warps too.
- **Profiler denominator:** these examples use all 100 observed cycles. For `.per_cycle_active` metrics, use the relevant unit's active cycles instead; do not silently substitute elapsed cycles.

## Q&A: scoreboard, Long Scoreboard, and Short Scoreboard

- **Scoreboard:** dependency-tracking hardware/bookkeeping that prevents instructions from using results before they are ready. A pending/ready register table is a conceptual model, not the literal NVIDIA implementation.
- Example: `r = X[i]; acc = r*w + acc;`. The load starts, but the multiply-accumulate cannot use `r` until the load result arrives. The consumer can stall while the warp stays resident.
- **Long Scoreboard:** dependency on an outstanding L1TEX operation (global, local, surface, or texture memory). It does not prove a DRAM access/miss or bandwidth saturation.
- **Short Scoreboard:** dependency on an MIO operation, commonly a shared-memory load; also includes other operations such as special math.
- **Why long/short?** These labels distinguish dependency classes/paths associated with different latency expectations; they are not a threshold applied to measured wait duration. A prolonged shared-memory wait does not become Long Scoreboard merely because it lasts many cycles.
- **Wait:** separate category for fixed-latency execution dependencies. **Barrier:** synchronization condition; do not conflate these with every scoreboard wait.
- [NVIDIA stall definitions](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html), [consumer attribution and interpretation](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

### Scoreboard hardware versus the measured stall

- **Hardware scoreboard tracks readiness; profiler stall counters observe blocked warp states.** Long/Short Scoreboard counts do not count executed loads or dependency edges. The PC-sampling variant counts samples, not exact duration of each individual load.
- `R1 = global_X[i]` produces a register value; a following `R2 = R1 * R3` waits if that load has not returned. Outstanding means issued but not yet completed.
- L1TEX in the Long Scoreboard definition names the global/local/texture/surface operation class, not proof that data was served exclusively from L1. It can involve L1 hits, L2, or DRAM.
- **SRAM** is a storage technology, not a stall category. L1 and shared memory share physical SRAM resources but ordinary shared-load dependencies are Short Scoreboard, while global-load dependencies are Long Scoreboard.
- **MIO = memory input/output**, an internal grouping/path for operations including shared-memory work and some special operations; it is not another cache level or a register storage type.
- The register may be the destination in every case: global load → register (Long); shared load → register (Short); fixed-latency arithmetic → register (Wait). Classification follows the producing operation, not the fact that the result resides in a register.
- Ready register values still require operand collection; register-file contention can cause Dispatch Stall. Do not conflate reading an existing value with waiting for a new value to be produced.

### Scoreboard versus profiler counter; L1TEX versus MIO

- **Operation:** a load starts fetching data. **Scoreboard:** execution bookkeeping tracks its outstanding dependency and blocks unsafe consumers. **Profiler stall counter:** records cycles/samples in the blocked state; it does not count completed loads or unique dependencies. Exact units depend on the metric.
- Conceptual register example: `R1 = load_global(X); R2 = R1 * R3;`. The second instruction waits for the new R1 value. Register storage existing does not make the pending value available.
- **L1TEX in Long Scoreboard** refers to the global/local/surface/texture operation class, not only L1 hits. The request may be served by L1, L2, or farther down the memory hierarchy.
- **Shared-memory naming caveat:** shared storage belongs to the unified L1TEX physical subsystem, but its dependent-consumer waits are generally classified as Short Scoreboard. Physical grouping and scoreboard category are different distinctions.
- **MIO = memory input/output:** a hardware instruction-handling path/group with a queue; not another cache or memory level. Includes shared-memory and some special-math/control operations.
- **Register destination does not determine the stall:** global load → register → consumer can give Long Scoreboard; shared load → register → consumer can give Short Scoreboard; fixed-latency arithmetic → register → consumer can give Wait.
- Already-produced operands can separately suffer register-access contention during collection (documented Dispatch Stall mechanism); this differs from waiting for the producer to finish. Registers being fast does not remove producer dependencies.

## Q&A: does every operation experience a dependency stall?

- **No.** A dependency means an instruction needs an earlier result. A dependency **stall** occurs only if that result is still unavailable when needed.
- Execution latency is not automatically stall time. Inputs may already be ready, or independent instructions may provide enough time for the producer to finish.
- An addition reading ready registers needs no memory load merely because it uses registers. Register storage and the operation producing a new register value are different things.

```text
R3 = R1 + R2       // Ready inputs; starts producing R3
R8 = R6 * R7       // Independent work
R9 = R10 + R11     // Independent work
R4 = R3 * R5       // No dependency stall if R3 is ready by now
```

- Likewise, `R1 = load_global(X)` followed by independent work and then `R2 = R1 + R3` incurs no Long Scoreboard wait for this dependency if the load has already completed.
- **Classification follows the unfinished producer:** fixed-latency arithmetic → Wait; global/local/texture/surface load path → Long Scoreboard; shared load/MIO path → Short Scoreboard. The result being destined for a register does not determine the category.
- These are dependency-stall categories, not a classification of every operation or every stall. Barriers, instruction fetch, pipeline/queue availability, and other conditions can also block issue.
- **Two distinct effects:** independent work within a warp can avoid reaching a consumer too early; another eligible warp can keep the scheduler issuing while the first warp remains stalled.

### Pending review question

A hypothetical kernel has **3% DRAM throughput**, frequent **Long Scoreboard** stalls, and many cycles with **no eligible warp**. Is it DRAM-bandwidth-limited, memory-latency-limited, or is the evidence insufficient? Explain how low bandwidth usage and substantial memory waiting can coexist. Learner answer pending.

## Not Selected and finding stall states in Nsight

- **Not Selected:** the warp is eligible, but another eligible warp is chosen to issue. This is a scheduler selection state, not a dependency/resource stall. It does not by itself mean an issue slot was wasted.
- Example: three eligible warps, one issue slot; one is selected and the others are not selected.

1. **Details → Warp State Statistics:** expand the section for the kernel-wide warp-state breakdown. Read the displayed units; the normalized chart is not a raw sample-count table.
2. **Source → SASS:** select **Warp Stall Sampling (All Samples)** or **Warp Stall Sampling (Not-issued Samples)** in the metric/navigation selector when collected. Inspect the breakdown at instruction locations. UI labels/visibility vary by version; source code correlation requires debug line information, but SASS can still be inspected.
3. **Raw:** search `smsp__pcsamp_warps_issue_stalled_` for collected sampling counters. Reason suffixes include `wait`, `short_scoreboard`, `long_scoreboard`, `not_selected`, `barrier`, and `dispatch_stall`; `_not_issued` restricts observations to cycles where the scheduler issued nothing.

- **Not Selected versus Not Issued:** the former describes an eligible warp not chosen; the latter filters scheduler-idle cycles. Do not confuse the two.
- **Missing data:** switching tabs cannot recover counters that were not collected. Use an expanded report with the relevant sections/counters; a missing counter is not zero.
- **Sampling caution:** All and Not-Issued samples can come from separate passes; do not subtract them as exact cycle counts. Dependency samples normally identify the waiting consumer instruction; trace its producer in SASS/source.
- References: [NVIDIA Source view](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page), [NVIDIA warp stall reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).

## No Instructions versus IMC Miss

| State | What is missing or pending? | Example |
| --- | --- | --- |
| **No Instructions** | Selection for instruction fetch, or completion of an instruction-cache miss | Execution jumps to code whose machine instructions are not available in the instruction cache yet |
| **IMC Miss** | A constant operand missed in the **Immediate Constant Cache** | An instruction needs a value addressed as `c[bank][offset]` in SASS and waits for that cache miss to complete |

- **Code versus operand:** No Instructions concerns fetching the machine instruction; IMC Miss concerns fetching a constant value used by an instruction.
- **No Instructions does not mean the warp has finished.** It also does not prove an instruction-cache miss: waiting for fetch selection is another documented cause.
- A literal encoded directly in an instruction does not need a separate constant-cache fetch. Source-level constants are not all implemented the same way.
- See [constants and instruction reissue](15-constants-and-instruction-reissue.md) for literal/kernel-argument/constant-memory distinctions and the documented IMC reissue example.
- In sampling counters, search reason suffixes **`no_instructions`** and **`imc_miss`**; use the lookup procedure above. Neither state alone establishes scheduler idleness or runtime impact.
- Sources: [NVIDIA stall definitions](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html), [NVIDIA constant/instruction-cache guidance](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

## What is an L1TEX operation?

- **L1TEX** names the L1 data-cache/texture hardware subsystem. **TEX means texture**, not Tensor Core.
- In the **Long Scoreboard definition**, an L1TEX operation means a **global, local, surface, or texture memory operation** handled by that path. A normal tensor load such as `r = X[i]` is a global-memory example; it does not require a texture API.
- Simplified global-load path: SM load instruction → L1TEX path → data returned from an applicable cache level or DRAM → destination register ready. Cache policy determines which caches participate; this is not a promise of an L1 lookup/hit for every load.
- Example: `r = X[i]; y = r + 1;`. If `r` is still pending when the add needs it, that consumer can report **Long Scoreboard**. The label does not tell us whether the load hit L1, hit L2, or accessed DRAM.
- **Local memory** here is CUDA thread-private memory, such as register spills; it is not shared memory or necessarily on-chip storage.
- **Physical-unit caveat:** unified L1/shared SRAM does not make every shared operation a Long Scoreboard producer. Ordinary shared-load dependencies are generally **Short Scoreboard** via the MIO operation class.
- Reference: [NVIDIA hardware model and stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Dependent loads: the next address needs the previous result

- **Dependent loads:** an earlier load returns information needed to calculate a later load’s address. The later load must wait for that information.
- **Embedding example** (conceptual pseudocode):

```text
token_id = token_ids[position]       // Load 1
value = embeddings[token_id, 0]      // Load 2 needs token_id to locate its row
```

- **Independent loads:** if both addresses are known, a later load can potentially issue while an earlier load is pending:

```text
a = X[i]
b = Y[i]
```

| Question | Dependent loads | Independent loads |
| --- | --- | --- |
| Is the next address already known? | No; needs the earlier result | Yes |
| Can these loads overlap? | Later load must wait for address input | Potentially, subject to issue/resource availability |
| Can another warp supply overlapping work? | Yes, if eligible | Yes, if eligible |

- **Low bandwidth despite stalls:** request a small amount → wait → discover next address → request again. Too few independent requests in flight can leave bandwidth unused. Other warps may hide the latency; this pattern does not guarantee low device bandwidth.
- **Different from load-to-arithmetic dependence:** `r = X[i]; y = r + 1` blocks arithmetic; the embedding example blocks launching the next load itself.
- **Source/SASS evidence:** trace the first load’s destination register into the address calculation and then the next load. This establishes a dependency, not an actual stall duration. Long Scoreboard samples at the waiting consumer add evidence of waiting; inspect surrounding independent work too.
- **Scope caution:** aggregate/source samples do not automatically prove every warp followed the same path or waited on every execution.
- **Review context:** this mechanism was introduced during Q15 and then taught explicitly; not previously established independent knowledge.
- Reference: [NVIDIA consumer attribution and latency guidance](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
