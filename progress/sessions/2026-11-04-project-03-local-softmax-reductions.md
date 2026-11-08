# 2026-11-04 — Explicit local reductions in warp softmax

## Context

- Project 3 softmax blog notes; TUTOR explanation and authorized documentation edit.
- Learning target: distinguish per-thread reductions from cross-lane reductions.
- Starting commit: `b664fcb`; local Mac. Existing local changes preserved.

## Work and evidence

- Updated `project_03_gpu_operator_library/notes/05-softmax-comparison/11-pytorch-warp-softmax.md`.
- Made `lmax = -infinity` and the local maximum loop explicit, followed by the
  warp maximum. Made `lsum = 0`, exponentiation and local accumulation explicit,
  followed by the warp sum. Added the two local variables to the legend.
- Checked sequential step numbers 1–35, LaTeX brace balance and local links with
  Python; passed. Ran `git diff --check`; passed. No GPU run or visual render.

## Learning and handoff

- User questioned where the local maximum occurred, then acknowledged the
  local-versus-warp explanation and requested it be explicit in the notes.
- Explanation assisted; independent transfer not assessed.
- Next: continue the warp-softmax algorithm walkthrough before other families.
- Changes remain local/uncommitted; no commit, push or kernel changes.

## Follow-up — explanatory examples

- Added three short explanations to note11: capacity/log2 with width900 padding;
  interleaved per-thread values versus adjacent addresses across lanes, illustrated
  by a lane/slot table; and the two-stage local/warp maximum and sum at width1024.
- Explained register-local summaries before cross-lane exchanges and the retained
  logits-to-exponentials lifetime. Algorithm and measured results unchanged.
- Reviewed the examples and local links; `git diff --check` passed. No GPU work or
  new mastery assessment. Next remains the warp-softmax walkthrough.

- Follow-up: user requested concision; reduced the three explanatory subsections
  to three short bullets with examples. Algorithm unchanged; whitespace check passed.
