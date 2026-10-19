# Working in this repository

At the start of each session, read:
1. `ai_prompts_instructions/ML_SYSTEMS_AI_INSTRUCTIONS.md`
2. `progress/STATUS.md`
3. The active project's note in `progress/projects/` and the latest relevant session note.

Default to the TUTOR workflow described in the learning instructions. The user
implements learning-target code; explain, give progressive hints, and review.
Only implement learning-target code when explicitly requested. Repository support
work can be delegated separately. Do not treat finished code as proof of mastery.

## Quick revision notes

The user has authorized maintaining a `notes/` folder inside each project. Create
it when a project begins and keep a short `README.md` index. After a lesson, add or
update a concise topic note: the core idea, a small example when helpful, and key
pitfalls. Capture covered material, not the entire future syllabus. Use relative
links and keep notes easy to revise; experiment logs and mastery evidence belong
in `progress/`. Notes summarize lessons, not proof of independent understanding.

Organize topic notes into descriptive section folders under `notes/` (for example,
`timing-and-setup/`, `gpu-hardware-and-scheduling/`, and `review/`) rather than a
large flat list. Keep `notes/README.md` as the top-level map and a short README
inside each section. Reuse a suitable section for new lessons; create a new one
only for a distinct topic. Keep review Q&A separate from lesson explanations.
When moving notes, preserve their content and existing lesson numbers, update
relative links and repository references, and check that local link targets exist.
Keep progress/mastery evidence in `progress/`, not in place of the notes index.

Use short bullet points, descriptive headings, and bold key terms for quick
scanning. Use numbered steps for procedures, small tables for comparisons or
formulas, and brief code examples where helpful. Avoid dense paragraphs; keep
examples and pitfalls easy to find. Apply this style to future lesson notes.

## Durable progress records

The user has authorized maintaining `progress/` so sessions can continue across
machines. After meaningful work, update `progress/STATUS.md`, the relevant project
note, and add or extend a dated note in `progress/sessions/`. Use the template in
`progress/templates/session.md`. Record tested facts, commands, environment,
evidence locations, limitations, and one concrete next task. Distinguish observed
results from plans and assisted success from independent understanding.

Keep existing session history; append a new note or label follow-up entries. Use a
descriptive suffix if multiple sessions occur on the same date. Use repository-relative
paths in committed documents. Do not record credentials or machine-specific absolute
paths. Do not claim tests, experiments, or remote synchronization happened without
checking them. Never mark the complete roadmap project finished from one milestone.

Before machine handoff, update the progress notes alongside relevant code. Commit
and push when authorized; otherwise state which changes remain local. Do not force
push, discard local work, or silently resolve a conflict involving learning progress.
