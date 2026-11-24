#!/bin/bash
# T-18 screen: make each square's code mean something locally. During training each part's code is replaced by a
# random one with probability p and the error on that part's cells is ignored, so the other parts must still draw
# correctly whatever their neighbours' codes are. World 0 (irregular), 4 options, 2 seeds per variant.
# Rule fixed before running: a variant passes the screen if best search >= 80% (mean of 2 seeds) and the one-shot
# encoder stays >= 90%. The best passing variant then gets the full test with the T-17 criteria unchanged
# (8 seeds; world 0 >= 89%; world 1 within 10 points of true parts, which scored 100%).
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --irregular --seeds 2 --out $name "$@" > logs/$name.log 2>&1; touch logs/$name.done ) & }
run loc_h36_p3 --hier 3,6 --code-noise 0.3
run loc_h36_p6 --hier 3,6 --code-noise 0.6
run loc_f6_p3  --hier 6 --code-noise 0.3
run loc_f6_p6  --hier 6 --code-noise 0.6
echo "started: loc_h36_p3 loc_h36_p6 loc_f6_p3 loc_f6_p6"
