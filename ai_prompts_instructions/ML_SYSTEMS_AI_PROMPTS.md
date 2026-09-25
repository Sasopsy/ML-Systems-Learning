# ML Systems AI Starter Prompts

First load ML_SYSTEMS_AI_INSTRUCTIONS.md into the assistant's context. Explicitly
ask it to read the file; do not assume an arbitrary filename is loaded automatically.
These are session templates, not a requirement to use every prompt every day.

## 1. Start the profiling-first project

Read ML_SYSTEMS_AI_INSTRUCTIONS.md. Use TUTOR mode.

I am starting my profiling-first ML systems roadmap. My first workload is a small
existing CNN training step; a decoder comes after I can measure and explain the CNN.
The goal is to understand measurement and execution, not implement model layers.

Use available context. Inspect the environment read-only when tools permit; otherwise
ask for only the missing hardware/software details needed to proceed.

Propose one small initial milestone. Separate the code I should write from support
code you may implement. Define the workload, measurement boundary, and acceptance
criteria. Start with the minimum concepts needed for this milestone.

You may propose a simple baseline using synthetic inputs, but label exactly what it
excludes. Do not build the entire harness or add optimizations. Ask for my prediction
about what the first measurement includes, then wait for my attempt.

## 2. Learn an unfamiliar mechanism

Use TUTOR mode. The concept is: [concept].
My current understanding is: [my explanation, or "I do not have a model yet"].
I encountered it while: [specific project task].

Correct the first consequential misconception. Give one concrete example and connect
it to what I should observe in code or a trace. Use official documentation for
version-specific behavior. Then give me a different prediction or implementation
task. Do not show its answer before I attempt it. Do not require me to rediscover
facts I have not been taught.

## 3. Review measurements without taking over the diagnosis

Use REVIEW mode. Do not edit code.

Workload and environment: [details]
Measurement boundary and method: [details]
Raw results and trace excerpts: [data or accessible artifact paths]
My explanation: [hypothesis]
My predicted next result: [prediction]

First audit whether these measurements support any conclusion. Then separate observed
facts from my interpretation. Identify the strongest competing explanation and one
small experiment to distinguish them. Identify any confounders that would make that
experiment inconclusive. Ask me to predict its outcomes before discussing a fix.
Do not infer unavailable trace details or invent a diagnosis from missing data.

## 4. Delegate peripheral implementation

Use BUILD mode for this scope only: [specific supporting component].
My learning target is [mechanism]; do not implement or modify that part.

Allowed files: [paths]
Required interface and behavior: [contract]
Checks and acceptance criteria: [requirements]
Execution/compute limits: [limits]

Preserve baseline definitions, reference results, and learning-target TODOs. Make a
small reviewable change, not a framework. Report the diff, tests actually executed,
and anything unverified. Do not change dependencies or expand scope silently.

## 5. Turn a paper or documentation page into an exercise

Use TUTOR mode. Read this source: [paper or official documentation].
The question I need to answer is: [specific systems question].
My current understanding is: [attempt].

Identify the relevant section and explain its assumptions, mechanism, and limits.
Separate what the source establishes from your own interpretation. Derive one small
experiment that would help me understand the mechanism in my project. Let me predict
and implement it before showing a complete solution. Do not replace reading the
important source section with an enormous summary.

## 6. Create a debugging exercise

Use EXAM mode. Create a self-contained toy exercise involving one profiling or
benchmarking mistake that fits what I have already studied. Do not edit my real
baseline or introduce unrelated faults.

Give me the code/workload and symptoms without naming the mistake. Ask me to choose
my first measurement and explain why. Do not reveal the answer until I attempt it
or explicitly request a hint. Afterward, assess the diagnostic reasoning, not just
whether I guessed the fix.

## 7. Finish a session and test transfer

Use REVIEW mode first, then EXAM mode.

My conclusion is: [my explanation]
Evidence and reproduction commands: [details]
What I still cannot explain: [gaps]

Check whether the conclusion follows from the evidence. Mark independent understanding
separately from assisted success. Ask one fresh explanation, modification, or diagnosis
question that changes a meaningful assumption. Do not include its answer yet.
After the attempt, prepare a short session note and a different task for later review.
Do not declare mastery merely because the benchmark or implementation works.

## Minimal experiment note

Learning target:
Question and workload:
Measurement boundary:
My prediction before assistance:
AI's strongest alternative:
Experiment and controls:
Commands, versions, and raw evidence paths:
Observed result:
My explanation:
What is still uncertain:
What I completed independently:
Next transfer task:
