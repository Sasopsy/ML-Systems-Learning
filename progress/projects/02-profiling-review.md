# Project 2 — Comprehensive profiling review

- Scope: notes 13–18. Approved 2026-10-18; open-note TUTOR, one question at a time.
- **Progress: Paused at user request after 20/24 reviewed; resume with unanswered Q21.** Estimate 6–10 targeted follow-ups, not a mandatory quota.
- Each core question has at most two related subquestions. Feedback: correct / partly supported / needs revision. Evidence level: independent with notes / assisted / not yet demonstrated.
- Teach gaps directly, then use one changed follow-up; revisit persistent gaps later. Summarize progress every four core questions. Do not repeat already-demonstrated arithmetic as basic drills.
- Completion: every round has evidence and unfamiliar final scenario succeeds without leading hints; essential gaps receive changed-example checks. Numerical acceptance and full Project 2 remain separate.
- Prior guided answers are retained in session records; do not relabel them independent transfer. No new GPU runs planned.

- Revision answers and exact resume prompt: [Q&A notes](../../project_02_performance_laboratory/notes/05-review/19-profiling-review-questions-and-answers.md). Q15 dependent-load transfer, Q20 latency distinction, and throttle classification remain potential targeted checks; no mastery inferred.

## Memory work and traffic

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 1 | More copy instructions versus similar DRAM bytes; explanations and discriminating counters | Assisted; revisit measurement selection in Q4 | Variable bytes independently identified; cache reuse taught; total requests corrected; 66.7% hit rate correct |
| 2 | L1/L2 traffic calculation with separate hit-rate denominators | Correct | Independent answer with notes allowed: 400 L2 requests, 300 L2 hits, 100 DRAM sectors |
| 3 | Instruction count versus transferred bytes versus FLOPs | Assisted; follow-up answered | Correct time decrease; concurrency as cause remains hypothesis. Category/opcode selection taught |
| 4 | Find instruction/traffic counters and recognize missing data | Correct for read traffic; total-traffic scope clarified | Independently named integer thread counter and DRAM read counter; recognized uncollected metric is not zero |

## Reading utilization

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 5 | Normalize activity with units and peak rate | Correct | Independent with notes allowed: 60% throughput, 75% active cycles; terminology refined |
| 6 | Maximum-of-constituents versus average/sum | Correct with refinement | 80% maximum and not useful FLOP/s; low utilization is not necessarily good performance |
| 7 | Mem Busy versus Mem Pipes Busy versus Max Bandwidth | Correct with aggregation refinement | Matched all three; rejected DRAM saturation; maximum constituent explained |
| 8 | Active/elapsed cycles and hardware/clock scopes | Arithmetic correct; clock-domain explanation assisted | 75%/30% independently; correctly chose DRAM cycles after worked clock-domain example |

## Residency and scheduling

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 9 | Resource-limited theoretical residency | Correct | 3 blocks limited by shared memory; 24 warps, 37.5% independently |
| 10 | Theoretical/achieved occupancy versus GPU coverage | Assisted arithmetic correction | Used earlier 4-warps/block value; corrected to 12.5%; follow-up 2 blocks = 25% correct; idle coverage distinction correct |
| 11 | Eligibility time distribution versus average and issue rate | Assisted; changed example correct | Initially equated eligibility and issue rate; follow-up 1.2 eligible and 0.6 instructions/cycle correct |
| 12 | All-samples versus Not-Issued stall counts | Correct | Scheduler no-issue filter and Dispatch Stall selected correctly; explanation supplied |

## Dependencies and latency hiding

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 13 | Producer/consumer dependency versus actual stall | Correct partial answer | No stall if ready; Long Scoreboard if pending; other-warp issue explanation supplied |
| 14 | Classify dependency, dispatch, throttle, barrier, and selection states | Correct for five presented states | Wait, Short Scoreboard, Not Selected, Barrier, Dispatch Stall all matched; throttle not yet tested |
| 15 | Low DRAM plus Long Scoreboard with alternative explanations | Assisted | Leading latency diagnosis and per-tier checks taught; dependent-load prerequisite introduced then explained; transfer unassessed |
| 16 | Prefetching, requests in flight, register pressure, and runtime | Correct with refinement | Explained overlap/register tradeoff; chose runtime; no-guarantee-of-zero-stalls refinement |

## SRAM, caches, wavefronts

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 17 | Per-block allocation versus SRAM carveout/L1 | Correct numbers; distinction assisted | 64 KiB allocated/unused; no automatic L1 expansion correct; allocation versus partition clarified |
| 18 | Same-SM/cross-SM reuse and L1/L2/TEX | Correct with ownership explanation | L1 same SM/L2 cross SM; ordinary per-block shared ownership explained |
| 19 | New bank mapping, wavefront count, broadcast | Assisted; changed example correct | Stride 4 and broadcast corrected; stride 8 = 4 banks/8 steps correct |
| 20 | Service rate versus latency and per-unit bytes/cycle | Throughput correct; latency assisted | 1 completion/cycle and 128 GB/s correct; doubled service not necessarily doubled latency taught |

## Complete diagnoses

| Q | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| 21 | Off/on evidence versus coverage hypothesis | Asked; unanswered at pause | Resume here after user returns |
| 22 | 128/512-block instruction work and runtime | Not asked | Not yet demonstrated |
| 23 | Optimization acceptance: correctness, timing, and activity | Not asked | Not yet demonstrated |
| 24 | Unfamiliar report: leading explanation, alternative, next measurement | Not asked | Not yet demonstrated |
