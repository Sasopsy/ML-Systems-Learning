# 2026-10-25 — Online softmax recurrence and pair merging

## Context

- Project 3, TUTOR, local macOS. Learning target: rescaling shifted sums and combining online normalizer pairs safely.
- Existing CUDA baseline/warp harnesses await the user-deferred GPU batch.

## Work and evidence

- Read learning instructions, current status/project note, latest session and note 06.
- Learner correctly supplied exp(-2) for maximum change 3→5; factor 1 and exp(-1) contribution when 4 follows; rescaled merge of [2,3] and [5,4]; NaN for empty-empty maximum subtraction; output exp(x_i-m)/s.
- Assistant explained general update/merge formulas, stable exponent evaluation, preserving old maxima, and empty-pair guards after both shuffle calls.
- Expanded note 06 and progress records. No kernel implementation, compilation, numerical experiments or GPU execution during this follow-up.

## Learning and handoff

- Correct guided conceptual responses; general formulas and empty-group handling were supplied. Not evidence of independent CUDA implementation or performance mastery.
- Next task: learner implements a small device pair-merge helper, including empty-group cases, before building the online kernel.
- Existing baseline/warp GPU correctness and performance remain untested. Changes local/uncommitted; no remote action, commit or push.

## Delegated helper and kernel skeletons

- User requested scaffolding. Inspected current softmax source and added `OnlineState`, `merge_online` with empty/rescale TODOs and a deliberately invalid NaN result placeholder, plus `softmax_online` with shared pair storage, identifiers, initial empty state and six staged TODOs.
- Comments require both partner fields to be shuffled by every lane before a conditional valid-partner merge; cover empty padding, block publication and final output traversal.
- Existing kernel bodies and the two-variant host harness remain unchanged. New kernel is incomplete and not launched.
- Next: learner implements helper M1–M2, then online TODOs 1–2 for review. Ran `git diff --check`; no whitespace errors. No CUDA compilation/device execution or performance claim.
