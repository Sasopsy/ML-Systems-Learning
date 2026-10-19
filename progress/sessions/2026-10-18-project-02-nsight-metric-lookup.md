# 2026-10-18 — Nsight metric lookup

## Context

- Project 2; TUTOR/open-note review. Local workspace, no GPU execution. Starting commit not rechecked.
- Target: locate rate, work, and traffic metrics for before/after interpretation.

## Work and evidence

- Checked official Nsight sections/memory tables documentation and parsed saved on035 raw CSV with Python.
- Verified scheduler-active issue 65.112002%, SM elapsed issue 57.580204%, and existing instruction/traffic lookup paths. Added compact lookup table to note 18.
- UI navigation is documented, not a live inspection. Missing opcode data cannot be recovered simply by switching tabs.

## Experiment

- None. Existing capture only.

## Learning and handoff

- Learner correctly calculated 1.28/12.8 GB/s concurrency examples and described cross-warp/within-warp overlap and register/residency tradeoff. Guidance refined issue opportunities versus ready work and intermediate activity improvement versus final runtime benefit.
- Next: use metrics to compare instruction/traffic work and runtime; independent transfer still unassessed.
- Documentation local; no commit, push, or remote synchronization.

## Comprehensive review — Q1 follow-up

- Approved 24-question tracker started; Q1 attempted. Learner independently proposed varying bytes per copy instruction; did not identify cache reuse and proposed issued instructions as discriminating evidence.
- Explained cache-served repeated requests, copy width/participating lanes, and per-level traffic/hit/miss counts versus issued instruction counts. These remain hypotheses for the measured variants.
- User requests short-note locations with future answers. Relevant note 18: checked prediction and metric lookup.
- Next: changed-example check with doubled L2 read sectors, unchanged DRAM bytes, unchanged instruction width/lane participation. No new GPU run; updates local.
