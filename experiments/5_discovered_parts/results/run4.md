# Discovered parts as the goal panel (8 seeds, T4 GPU) - corrected run

Fixes vs earlier runs: every panel gets a position cue within its own part; encoder replaced by a 7x7 window
around each part's centre + small MLP (the old pooled encoder merged horizontal/vertical-bar goals onto one code).

## Phase 1: parts from annealed coupling
- Coupling on hidden tissue boundaries: 0.34; inside tissues: 0.55
- Match to true tissues (adjusted Rand index): raw 0.48, after merging fragments 0.66
- Parts per seed after merging: [9, 9, 6, 3, 5, 8, 7, 8] (true answer: 9)
- Phase-1 model's own accuracy on new patterns: 100%

## Phase 2: steering to new patterns (% fully correct; after damage in brackets)

| Goal panel | Parts | Encoder | Search (own-part errors) | Gradient-tuned codes | Search cost |
|---|---|---|---|---|---|
| discovered parts | 7 | 27% (27%) | 27% (27%) | 25% (24%) | 128 batched trial runs, 13s |
| true tissues (best case) | 9 | 75% (74%) | 75% (75%) | 75% (75%) | 64 batched trial runs, 7s |
| fixed 5x5 grid (misaligned) | 25 | 90% (86%) | 1% (1%) | 4% (4%) | 64 batched trial runs, 7s |

Total time 1779s
