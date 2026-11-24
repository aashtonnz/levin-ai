#!/bin/bash
# T-17: nested squares ("pixel resolution" parts) instead of discovered parts. 8 seeds each, 4 options.
# Fixed before running:
#   H1 (development world, irregular seed 0): nested 3,6 best search >= 89% (true parts got 99%; within 10 points).
#   H2 (held-out world, irregular seed 1): nested 3,6 within 10 points of true parts on the same world.
#   Controls (reported, not criteria): a single coarse grid (3) and a single fine grid (6), to see whether nesting matters.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --irregular --out $name "$@" > logs/$name.log 2>&1; touch logs/$name.done ) & }
run hier36_irr0  --hier 3,6
run coarse3_irr0 --hier 3
run fine6_irr0   --hier 6
run hier36_irr1  --hier 3,6 --world-seed 1
run true_irr1    --world-seed 1
echo "started: hier36_irr0 coarse3_irr0 fine6_irr0 hier36_irr1 true_irr1"
