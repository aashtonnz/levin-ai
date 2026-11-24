#!/bin/bash
# T-16 scaling check: does 8-option training keep improving with much longer training?
# 4 seeds, true parts only, 20000 steps, learning rate kept high until 80% (drops at 80% and 95%),
# encoder accuracy logged every 2000 steps. Reading fixed before running: if accuracy is still rising over the last
# two checkpoints before the first drop (by 5+ points on average), longer training is a live fix; if it has levelled
# off well below 80%, it is not.
cd "$(dirname "$0")"; mkdir -p logs
( python levin_scale.py --true-only --seeds 4 --options 8 --iters2 20000 --lr-drops 0.8,0.95 --eval-every 2000 \
    --out long8 > logs/long8.log 2>&1; touch logs/long8.done ) &
echo "started long8"
