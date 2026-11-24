#!/bin/bash
# T-18 gentler version: randomise exactly k squares' codes per training example (small squares 6x6, world 0),
# 4 seeds each (2 seeds proved too noisy), plus the best earlier setting (60% of squares) at 4 seeds for comparison.
# Rule unchanged: pass if best search >= 80% (mean over seeds) and encoder >= 90%. A passing variant then gets the
# full test: 8 seeds on worlds 0 and 1 with the T-17 criteria.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --irregular --seeds 4 --hier 6 --out $name "$@" > logs/$name.log 2>&1; touch logs/$name.done ) & }
run loc3_k1   --code-noise-k 1
run loc3_k2   --code-noise-k 2
run loc3_k4   --code-noise-k 4
run loc3_p6   --code-noise 0.6
echo "started: loc3_k1 loc3_k2 loc3_k4 loc3_p6"
