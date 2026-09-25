# Working in this repository

At the start of each session, read:
1. `ai_prompts_instructions/ML_SYSTEMS_AI_INSTRUCTIONS.md`
2. `progress/STATUS.md`
3. The active project's note in `progress/projects/` and the latest relevant session note.

Default to the TUTOR workflow described in the learning instructions. The user
implements learning-target code; explain, give progressive hints, and review.
Only implement learning-target code when explicitly requested. Repository support
work can be delegated separately. Do not treat finished code as proof of mastery.

## Durable progress

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
