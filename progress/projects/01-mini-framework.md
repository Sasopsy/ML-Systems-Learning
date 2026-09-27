# Project 1: Mini deep-learning framework

Implementation: **paused, incomplete**. Active learning focus moved to Project 2.
Revision notes: [index](../../project_01_mini_framework/notes/README.md), summarized
from the existing scalar implementation and learning record on 2026-09-25.
At the user's request, revision notes now use bullet points, short sections,
tables, and numbered procedures for quick review; no new learning evidence.

## Implemented and reviewed

- `project_01_mini_framework/value.py`: scalar values, parent links, addition,
  multiplication, composed negation/subtraction, reflected addition, ReLU,
  reverse topological traversal, and gradient accumulation.
- `project_01_mini_framework/nn.py`: `Neuron`, `Layer`, and `MLP`, with shared
  parameter objects, ReLU hidden layers, and a linear output layer.
- `project_01_mini_framework/example.py`: assertions covering forward values,
  aliasing, shared graph branches, finite differences, ReLU, neuron/layer/MLP
  behavior, parameter identity, and input gradients.
- `project_01_mini_framework/train.py`: mean squared error over five examples
  following `y = 2x + 1`, with manual SGD and per-step gradient reset.

## Reproduction and observed results

Run from the repository root:

```sh
python3 project_01_mini_framework/example.py
python3 project_01_mini_framework/train.py
```

Both passed on 2026-09-25 using Python 3.9.6 on macOS arm64. After 100 training
steps: `w = 2.0`, `b = 0.9999999997962964`; both final assertions passed.

## Understanding and diagnosis evidence

- Learner correctly derived the chain rule for `(a*b + c)^2` in conversation.
- Implementations were learner-written with assistant-provided skeletons, hints,
  and tests: classify as assisted, not independently mastered.
- A previous assistant check changed multiplication accumulation from `+=` to `=`
  in memory: the test gradient changed from 24 to 20 and was rejected. This is
  assistant-observed evidence, not a learner-authored diagnostic demonstration.
- A training failure was traced to 20 iterations being insufficient for the chosen
  tolerance; the learner changed to 100. The gradient rules were correct.
- Topological graphs/sorting remain on the learner's `deeper_study.md` list.

## Remaining scope and limitations

- XOR training was proposed, and an assistant-only in-memory reference run worked.
  No `train_xor.py` is present, so learner completion is unverified.
- Current usage rebuilds graphs each training step and resets parameter gradients.
  Repeated backward on the same graph is not defined/tested as a supported contract;
  intermediate gradients can retain prior contributions.
- Forward values must not be mutated before backward; saved-value/mutation semantics
  are not implemented as a robust contract.
- Tensor storage, shapes/strides, broadcasting, views, tensor autograd, dedicated
  optimizer abstraction, CPU tensor backend, CUDA backend, CNN, and recurrent model
  remain unimplemented.
- The roadmap's full exit criterion has not been independently assessed.

If returning to this project, resume with the XOR exercise or explicitly select the
next tensor milestone; do not infer completion from the scalar MLP.
