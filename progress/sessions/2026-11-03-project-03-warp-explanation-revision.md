# 2026-11-03 — Integrate warp softmax algorithm and template explanation

## Context

- Project3, TUTOR explanation revision plus authorized note maintenance.
- User found the prior presentation messy: template names appeared before their
  purpose. Requested one integrated explanation of algorithm, method and choices.
- Starting commit: `b664fcb`; local macOS, existing uncommitted work preserved.
- Learning target: follow one row through the kernel and connect each compile-time
  choice to the concrete work it controls.

## Work and evidence

- Read learning instructions, current status/project, prior session and note11;
  rechecked the bundled kernel's load, exponential/sum and output branches.
- Rewrote note11 around input[37,1024]: select compiled capacity, assign rows/lanes,
  retain private data, reduce maximum, retain exponentials/reduce sum, normalize.
  Explain each template argument at its point of use, then recap the call.
- Kept later variations for width900, width128, other dtypes/modes and source
  references. Updated index and learning instructions with the teaching preference.
- Checked local Markdown link targets and `git diff --check`; no implementation,
  CUDA compilation, GPU measurements or new algorithmic performance conclusions.

## Learning and handoff

- User feedback concerns explanation structure; no independent mastery assessed.
- Next: discuss the integrated walkthrough before other kernel families/profiles.
- Documentation changes remain local/uncommitted. No commit, push or remote run.

## Follow-up — assume basic CUDA, reduce exposition

- User clarified that basic CUDA explanations were unnecessary. Condensed note11
  to mapping, private storage, five algorithm steps and the relevant template
  choices, with a width table and brief comparison to retained Online.
- Updated learning instructions: assume basic CUDA knowledge, introduce only
  kernel-specific terminology as needed, and keep explanations concise.
- Retained source references and measured-scope limits. This preference is not
  evidence of independent mastery. No code/GPU changes or new performance claims.
- Checked note links and `git diff --check`; changes remain local/uncommitted.

## Follow-up — restore kernel-level reasoning

- User found the condensed version too stripped down. Reworked note11 into a
  substantive explanation connecting mapping to retained storage, capacity
  specialization, scalar reductions, in-place exponentials, mode flags and output.
- Retained the width900 example, per-width mapping table, source pointers and
  comparison tradeoffs. Removed elementary CUDA exposition while keeping the
  reasons behind implementation choices. Updated the teaching preference to
  distinguish assumed prerequisites from desired technical depth.
- Rechecked the source helper/dispatch, note link targets and `git diff --check`.
  Documentation only; no new GPU measurements or mastery assessment.

## Follow-up — paper-style algorithm and explicit retained state

- User requested computational pseudocode instead of prose-heavy explanation.
  Rewrote note11 with contract, compile-time/runtime parameters, host launch,
  per-thread private state, four device stages and XOR all-reduction helper.
- Explained retained storage lifetime explicitly: logits survive the maximum
  reduction, then the same slots hold exponentials through the sum and store.
  Included width128 batching, width900 padding and width1024 instantiation.
- Ran a transient Python CPU model against an independent stable-softmax
  reference for (M,N)=(1,17),(1,128),(37,128),(37,512),(37,900),(37,1024),
  (37,2048); all passed, with each valid output written exactly once.
- Checked local Markdown targets and `git diff --check`; passed. This is a
  source/pseudocode sanity check, not CUDA execution or performance evidence.
- Updated index and teaching preference. No independent mastery assessed.
  Next: discuss this algorithm before Reg/General families and profile attribution.
  No commit/push or kernel implementation changes; existing work preserved.

## Follow-up — one continuous algorithm box

- User clarified the exact format: one long algorithm box like FlashAttention
  papers, with explanations above/below. Consolidated host selection, parameters,
  private state, computation and both inline reductions into one numbered box.
- Preserved algorithm semantics and source references; recorded the single-box
  preference in learning instructions. No new CUDA work or measurements.
- Checked local note link targets and `git diff --check`; passed. Next remains
  discussion of this algorithm before later kernel families/profile attribution.

## Follow-up — LaTeX with descriptive names

- User requested LaTeX and rejected single-letter variable abbreviations.
  Replaced the text box with one LaTeX boxed array, named dimensions, retained
  storage and reductions descriptively, and mapped names back to PyTorch source
  in a separate table. Updated teaching preference and current progress.
- Algorithm semantics unchanged; no GPU execution or new performance claim.
  Checked local links, LaTeX environment/brace balance and `git diff --check`.
  No visual LaTeX renderer was run. Changes remain local/uncommitted.

## Follow-up — compact names and legend

- User found the full descriptive identifiers visually cluttered. Replaced them
  with short mnemonic names and an explicit legend. Kept one LaTeX box, using
  per-thread vector notation and one inline XOR helper to reduce repetition.
- Clarified that independent row work is grouped for presentation; the source
  helper interleaves rows at each shuffle offset. No algorithmic change or GPU run.
- Checked LaTeX structure, local links and `git diff --check`; no visual renderer
  was run. Recorded the revised teaching preference. Next remains discussion of
  warp softmax before other families. Changes local/uncommitted.
