# CCVR direction gate v1

This repository implements a frozen, staged falsification protocol. The
dataset gate must pass before feature archives may be downloaded. OpenCLIP
must pass before EVA-CLIP may be downloaded. No human annotation, Omni calls,
media downloads, threshold changes, or GPU 4-7 use are permitted.

The task is video-level constraint satisfaction. It does not claim that
multiple conditions occur in the same temporal window.

## Operator-aware near misses

For conjunctions, a near miss satisfies at least one literal but not all
literals. For disjunctions, a negative necessarily satisfies no branch, so the
closest valid negative is a base-relevant video that satisfies none. The
manifest retains the preregistered field name `partial_negatives`; its meaning
is operator-aware and is recorded in every row as `negative_definition`.

## Formal state transitions

`direction_selection_pending` -> `public_dataset_gate_failed` or
`data_gate_passed` -> `problem_gate_failed` / `solved_by_simple_logic` /
`problem_confirmed` -> `problem_confirmed_method_foothold_missing` /
`direction_confirmed`.

