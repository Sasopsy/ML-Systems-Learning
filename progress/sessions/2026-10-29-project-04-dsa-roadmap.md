# 2026-10-29 — Project 4 DSA curriculum and numbering migration

## Context

- Project and milestone: add Project 4 DSA after Project 3; renumber later projects.
- Mode: BUILD for explicitly delegated repository documentation; future lessons
  remain TUTOR, with learner-owned algorithm implementations.
- Learning target: future beginner DSA in C++17, including language prerequisites.
- Starting commit: `b664fcb`.
- Environment: local macOS workspace; no compiler/framework/GPU version probe
  needed for this documentation change.

## Work and evidence

- Added the approved nine-stage curriculum, capstone contract, validation and exit
  criteria to `roadmap.md`; updated headings, dependencies and recommended order.
- Created `progress/projects/04-dsa-for-ml-systems.md`, marked planned/not assessed.
- Recorded beginner DSA background and C++ preference in the learning instructions.
- Renamed the runtime record to `progress/projects/10-11-inference-runtime.md`;
  repaired its status link and annotated its historical entry and original session.
- Added a migration explanation to `progress/README.md` and current status.
- Preserved existing Project 3 GEMM notes and uncommitted handoff documentation.
- No DSA source, exercises, project folder or lesson notes created: these begin
  when teaching starts. No algorithm implementation or runtime experiment.
- Validation: `git diff --check` passed. A Python documentation check compared
  all 20 project headings with the original `git show HEAD:roadmap.md` mapping;
  all matched. All 20 checked local links/anchors resolved (edited documents and
  added status links); the original runtime session remains an unchanged prefix
  followed by a migration annotation. Confirmed the old runtime record is renamed
  and the future DSA exercise folder has not been created.

## Numbering migration

| Previous ID | Current ID |
| --- | --- |
| 1–3 | 1–3 (unchanged) |
| New DSA project | 4 |
| 4A / 4B / 4C | 5A / 5B / 5C |
| 5 / 6 / 7 / 8 | 6 / 7 / 8 / 9 |
| 9 / 10 (runtime track) | 10 / 11 |
| 11 / 12 / 13 | 12 / 13 / 14 |
| 14 / 15 / 16 / 17 | 15 / 16 / 17 / 18 |

- Current roadmap and project records use the new IDs. Dated historical entries
  retain their original IDs and claims; use this table to interpret them.
- Phase labels group topics and are separate from project IDs; they are unchanged.
- Historical mentions of `progress/projects/09-10-inference-runtime.md` describe
  the original path; current links resolve to the renamed record.

## Learning and handoff

- Learner preference: C++ from the start; no prior DSA background.
- Understanding not assessed. Planning and documentation are not lesson completion.
- Active next task: Project 3 learner derives grid/edge guard and flat A/B/C
  offsets for row-major GEMM with M37/K130/N65 and 16×16-thread blocks.
- DSA starts later with a traced linear scan and a separate learner exercise.
- Local changes only; no commit, push, or remote synchronization performed.
