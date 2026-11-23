# Plan

The question: can a network built from competent parts, with a small goal panel per part, be retargeted by searching the panel instead of retraining, and can it find its own parts? Toy grids first; a real task only if the method survives scaling.

## T-3: Scale-up test
Change one thing at a time from the current setup (3×3 tissues of 5×5 cells, 4 options, independent rule):
- 5×5 grid of tissues (25 parts).
- Irregular tissues: random shapes of varied size (for example a Voronoi partition), so spectral clustering cannot rely on a regular layout.
- 8 options per part.
- Shift rule (each tissue shows a neighbour's setting), so search must handle interactions.
Measure for each: parts found and match to true tissues, steering success through discovered and true parts, retraining and prompt-tuning on the same examples, and trial runs used by search. Pass criteria to fix before running: steering through discovered parts within 10 points of true parts, and at least 30 points above the best baseline.

## T-4: Gate after scale-up
If discovered-part steering holds up, move to the part-quality check and the real-task pilot. If it does not, find which change broke it (size, shape, options or interaction) and rethink part-finding or search for that case before going further.

## T-2: Part-quality check
After finding parts, run search on the training examples. If a part's own error stays high, its boundaries are likely wrong: split it, or rerun clustering with a different number of parts, and keep the version whose search fits the examples best. Uses only the training examples, no true labels.

## T-5: Literature check and real-task pilot choice
Compare with neural cellular automata, modular and mixture-of-experts networks, steering vectors and controllable generation. Choose one real task where parts are plausible, such as image generation with controllable regions or a simulated body with controllable limbs.

## T-6, T-7: Review and write-up
A fresh-eyes review of code and results before the write-up, then a short report with methods, results, failures and comparisons.
