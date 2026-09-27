# ML Systems learning workspace

A project-first workspace for the [ML systems roadmap](roadmap.md).

**Resume here: [current progress and next task](progress/STATUS.md).**

## Repository map

- `roadmap.md`: full roadmap and project exit criteria.
- `project_01_mini_framework/`: scalar autograd, neurons, layers, MLP, and linear training.
- `project_02_performance_laboratory/`: model and individual-kernel profiling;
  currently revision notes only.
- Each project's `notes/` folder: short revision notes, maintained as lessons are covered.
- `progress/`: durable project status, learning evidence, and session handoffs.
- `ai_prompts_instructions/`: tutoring workflow and reusable prompts.
- `deeper_study.md`: topics to revisit.
- `AGENTS.md`: entry point for coding assistants, including progress maintenance.

## Quick revision

- [Project 1: autograd and training](project_01_mini_framework/notes/README.md)
- [Project 2: performance and profiling](project_02_performance_laboratory/notes/README.md)

Future projects get the same `notes/` layout when started. These are lesson
summaries; `progress/` tracks experiments, understanding, and where to resume.

## Run the existing project

Project 1 uses only the Python standard library and requires Python 3.9 or newer.
From the repository root:

```sh
python3 project_01_mini_framework/example.py
python3 project_01_mini_framework/train.py
```

The first script exits silently when all assertions pass. The second prints training
progress and checks the final linear-model parameters. Run without Python's `-O`
option so assertions remain enabled.

Project 2 has no benchmark implementation or dependency environment yet. Select its
execution machine before installing PyTorch; keep its environment local to that
machine. Do not copy virtual environments through Git.

## Work across machines

Current setup is local only; the user has deferred GitHub. Configure a remote and
push the local history before following the clone workflow below.

One-time setup after the remote repository is configured (replace `REMOTE_URL`):

```sh
git clone REMOTE_URL ML_Systems
cd ML_Systems
git status
```

Before a session, start with a clean working tree and pull committed work:

```sh
git status
git pull --ff-only
```

Read `progress/STATUS.md`, the active project note, and the latest relevant session
note. Verify the local Python/device environment; another machine's timing results
and installed packages do not describe this machine.

At a handoff, update the progress notes, then review and commit the code and notes
together. Stage the intended paths explicitly:

```sh
git diff
git add progress/ path/to/changed_file
git diff --cached
git commit -m "Describe the completed work and handoff"
git push
```

Replace `path/to/changed_file` with the files actually changed. The initial push for
a newly configured remote is `git push -u origin main`.

Prefer one active machine at a time. Push before switching, then pull on the next
machine. If `pull --ff-only` or push fails because history diverged, inspect and
reconcile the commits; do not force push or reset away work. Use separate branches
for concurrent experiments and merge them deliberately.

Git transfers committed code and notes, not installed packages or ignored artifacts.
Keep datasets, checkpoints, and full profiler traces outside Git. Record durable
artifact locations (without credentials), configuration, and small result summaries
in progress notes. A local-only artifact path is not available on another machine.
