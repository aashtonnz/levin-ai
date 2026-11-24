#!/bin/bash
# T-18 follow-up screen: small squares only (6x6), more code noise and/or longer training. 2 seeds, world 0.
# Same rule as run_local.sh: pass if best search >= 80% (mean of 2 seeds) and encoder >= 90%.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --irregular --seeds 2 --hier 6 --out $name "$@" > logs/$name.log 2>&1; touch logs/$name.done ) & }
run loc2_p6_long  --code-noise 0.6 --iters2 5000
run loc2_p8       --code-noise 0.8
run loc2_p8_long  --code-noise 0.8 --iters2 5000
echo "started: loc2_p6_long loc2_p8 loc2_p8_long"
