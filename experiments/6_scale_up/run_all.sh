#!/bin/bash
# Run the scale-up conditions in three lanes side by side on one GPU (one run at a time per lane, so memory fits
# on a 15 GB T4). One log per condition in logs/; each lane writes logs/lane<N>.done when finished.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lane() { n=$1; shift; ( for job in "$@"; do set -- $job; name=$1; shift; python "$@" > logs/$name.log 2>&1; done; touch logs/lane$n.done ) & }
lane 1 "big levin_scale.py --skip-grid --side 25 --tissues 25 --out big" \
       "all levin_scale.py --skip-grid --side 25 --tissues 25 --irregular --options 8 --rule shift --out all"
lane 2 "irregular levin_scale.py --skip-grid --irregular --out irregular" \
       "options8 levin_scale.py --skip-grid --options 8 --out options8"
lane 3 "shift levin_scale.py --skip-grid --rule shift --out shift" \
       "retrain_big ../3_grid_2d/levin2d.py --grid 5 --rule shift --out retrain_big"
echo "started 3 lanes: [big, all] [irregular, options8] [shift, retrain_big]"
