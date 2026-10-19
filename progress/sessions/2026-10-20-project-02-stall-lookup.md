# 2026-10-20 — Stall-state lookup

## Context

- Project 2; TUTOR open-note review. Local documentation work; starting commit not rechecked.

## Work and evidence

- Read note 16 and review tracker. Checked NVIDIA Profiling Guide, Source view documentation, and Compute Triage search results.
- All five Q14 matches correct. Found existing Not Selected definition at note 16 line 102; added dedicated definition and stall lookup procedure.
- Reconciled tracker Q9–Q14 with conversation, distinguishing assisted and independent responses. No GPU run or live GUI verification.

## Learning and handoff

- Learner correctly matched Wait, Short Scoreboard, Not Selected, Barrier, Dispatch Stall. Throttle distinction not tested by this question.
- Next: Q15 low DRAM throughput plus Long Scoreboard, preserving alternative explanations.
- Documentation local; no commit, push, or remote synchronization.

## Follow-up: No Instructions and IMC Miss

- Paused Q15 to explain instruction-fetch selection/cache misses versus Immediate Constant Cache operand misses, using NVIDIA definitions checked in the preceding turn.
- On request checked notes: note 15 already covered constants and IMC reissue; note 16 briefly listed instruction-fetch waiting. Added dedicated comparison and cross-link, avoiding duplicate reissue explanation.
- Next remains Q15; no new GPU run, live UI inspection, commit, or push.

## Q15 and residency clarification

- Learner response to Q15 remained vague; explained insufficient latency hiding as leading hypothesis and L1/L2 saturation as alternatives. Dependent-load follow-up not answered.
- On request added admitted-to-SM state examples, eligibility transitions, independent-next-instruction caveat, and L1TEX operation explanation to note 16. Checked NVIDIA hardware/stall definitions; no experiment.
- Next remains dependent-load follow-up for Q15; understanding assisted, not independently demonstrated.

## Dependent loads prerequisite

- Learner correctly recalled elapsed-window normalization but used operation count; clarified DRAM bytes and traced load→address→load dependencies with consumer-attributed sampling.
- Learner noted dependent loads had not been explicitly taught; acknowledged prerequisite gap and supplied embedding example versus independent X/Y loads. Understanding reported, not independently tested.
- Added requested concise section to note 16. Next: changed Q15 application before Q16. No new GPU run or synchronization.

## Service work versus total latency example

- Q20 learner correctly answered one completion/cycle and 128 GB/s, but assumed doubled service steps necessarily double total latency. Explained other stages and supplied separate 6+1 versus 6+2 cycle illustration.
- On request checked note 17: general caveat existed; added missing worked table and overlap/queueing assumptions.
- Q21 off/on coverage comparison has been asked and remains unanswered. No new experiment, commit, or push.

## User-requested pause and Q&A handoff

- User explicitly stopped the review and requested a separate questions-and-answers file. Created note 19 with Q1–Q20 corrected revision answers and discussed follow-ups, references to topic notes, and unanswered Q21 prompt. Updated notes index, tracker, current status, and project progress.
- Reconciled Q15–Q20 evidence: Q15 assisted with newly taught prerequisite; Q16 correct with refinement; Q17 distinction assisted; Q18 correct with ownership explanation; Q19 corrected then transfer arithmetic correct; Q20 throughput correct but latency explanation assisted.
- Resume at Q21. Q22–Q24 not yet asked; optional later transfer checks do not justify restarting all questions.
- No new experiment, commit, push, or remote synchronization. Existing local work preserved.
