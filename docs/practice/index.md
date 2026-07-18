# Practice

Theory becomes useful when experiments are reproducible, diagnosable, and fairly compared.

## Experiment design

- Environment version and wrappers
- Random seeds and number of runs
- Evaluation protocol, hyperparameters, and compute budget
- Learning curves with uncertainty

## Implementation notes

- Handle terminated and truncated episodes correctly
- Bootstrap correctly at episode boundaries
- Normalize observations and advantages when appropriate
- Evaluate with exploration disabled
