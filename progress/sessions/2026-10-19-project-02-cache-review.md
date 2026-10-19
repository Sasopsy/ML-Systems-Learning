# 2026-10-19 — Cache review

## Context

- Project 2 comprehensive profiling review; TUTOR, open notes.
- Target: distinguish instruction work, cache requests, hits, and DRAM traffic. Local documentation only; starting commit and hardware versions not rechecked.

## Work and evidence

- Read tracker, current progress, note 17 cache section, and session template. Updated tracker and progress records with Python.
- Evidence: learner answers in conversation; no new counters collected.

## Experiment

- None; simplified teaching examples only.

## Learning and handoff

- Q1: learner needed cache-reuse explanation; initially counted only two hits as L2 requests, excluding first miss. Corrected to 384 bytes requested, 128 from DRAM; learner then correctly computed 66.7% hits. Assisted understanding; counter selection still to revisit in Q4.
- Q2: independently answered changed two-cache example with notes allowed: 400 L2 requests, 300 L2 hits, 100 sectors fetched from DRAM.
- Next: Q3, explain whether more total executed instructions establishes more useful floating-point work.
- Documentation local; no commit, push, GPU execution, or remote synchronization performed.

## Follow-up: instruction categories

- Q3: learner independently rejected equating total instruction increase with useful floating-point work. Proposed throughput summaries to locate the increase; category/opcode counts and missing-data limitations explained.
- On request, added integer counter, thread-versus-warp units, SASS address-use inspection, and version-dependent opcode counters to note 18. Prior lookup found no matching integer counter in on035 CSV/log search; no new profiling performed.
- Next: answer pending unchanged-bytes/higher-throughput timing question before Q4. Documentation edits only, no commit or push.

## Round 1 recap

- Q3 follow-up: learner correctly identified decreased time for fixed bytes/higher throughput, but concurrency explanation remains unverified.
- Q4: correctly selected integer predicated-on thread counter and DRAM read-byte counter; correctly rejected interpreting absent counter as zero. Clarified all DRAM traffic includes writes too.
- 4/24 reviewed: Q1/Q3 assisted, Q2/Q4 correct with notes allowed (Q4 traffic scope refinement). Next Q5 toy activity normalization. No new GPU measurements, commit, or push.

## Follow-up: normalization and memory pipelines

- Q5 independently correct: 480/(200*4)=60% throughput, 150/200=75% active cycles. Q6 correctly chose maximum 80% and rejected useful-FLOP/s interpretation; refined lower utilization versus good performance.
- Q7 paused for pipeline explanation. On request, added Mem Pipes Busy plain-language definition, LSU processing-path example, scheduler issue versus pipeline handling, and acceptance versus completion distinction to note 17.
- Next: answer Q7 matching question and DRAM-saturation implication. Documentation only; no new measurement or remote synchronization.

## Round 2 complete: clock domains

- Q7 all mappings correct; refined Mem Busy as maximum normalized constituent rather than unspecified all-hardware aggregation.
- Q8 75% active/30% elapsed issue independently correct. Clock-domain question required explanation; learner then correctly chose DRAM elapsed cycles for DRAM bytes/cycle normalization. Assisted clock-domain understanding, not independent transfer.
- Added requested different-clock-rates explanation, toy 1/2 GHz example, active-versus-elapsed distinction, and normalization units to note 17. Examples are not measured device frequencies.
- Next: Q9 resource-limited theoretical residency. Documentation local; no GPU run, commit, push, or remote sync.
