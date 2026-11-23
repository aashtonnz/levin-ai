# Idea 4: do parts emerge from learnable coupling? (GPU run, 8 seeds, 3000 training steps, coupling cost 0.05)

Hidden structure: 3x3 tissues of 5x5 cells, 4 options each. The model is not told where tissues are;
its goal panel is a 5x5 grid of learned codes that does not line up with the tissues.

## Independent rule (talking across tissue boundaries is only a nuisance) - main test

| | Learned coupling | Fixed coupling (control) |
|---|---|---|
| Coupling on hidden tissue boundaries | 0.23 | 0.93 (fixed) |
| Coupling inside tissues | 0.46 | 0.93 (fixed) |
| Coupled groups vs true tissues (adjusted Rand index; 1 perfect, 0 chance) | 0.34 (per seed 0.75, 0.13, 0.46, 0.20, 0.39, 0.42, 0.36, 0.00) | - |
| Groups found per seed (true answer: 9) | 54, 162, 105, 160, 117, 110, 56, 10 | - |
| New patterns correct: encoder | 46% | 86% |
| New patterns correct: local search on panel | 39% | 46% |
| New patterns correct: gradient-tuned codes | 5% | 21% |

## Boundary rule (tissues must compare neighbours across boundaries) - comparison

| | Learned coupling | Fixed coupling (control) |
|---|---|---|
| Coupling on hidden tissue boundaries | 0.37 | 0.93 (fixed) |
| Coupling inside tissues | 0.42 | 0.93 (fixed) |
| Coupled groups vs true tissues (adjusted Rand index) | 0.12 | - |
| New patterns correct: encoder | 58% | 52% |
| New patterns correct: local search on panel | 57% | 39% |
| New patterns correct: gradient-tuned codes | 2% | 4% |
