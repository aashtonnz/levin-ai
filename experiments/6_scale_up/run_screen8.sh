#!/bin/bash
# T-16 screen: why does training fail with 8 options? True parts only, 2 seeds per variant, all side by side.
# Decision rule (fixed before running): a variant passes the screen if best search >= 80% averaged over its 2 seeds
# and the control (A) stays low; the best passing variant then gets a full 8-seed run before any claim.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --seeds 2 --out s8_$name "$@" > logs/s8_$name.log 2>&1; touch logs/s8_$name.done ) & }
run A_control   --options 8
run B_longer    --options 8 --iters2 6000
run C_bigger    --options 8 --hidden 128
run D_curric    --options 8 --curriculum
run E_4options  --options 4
echo "started screen: A_control B_longer C_bigger D_curric E_4options"
