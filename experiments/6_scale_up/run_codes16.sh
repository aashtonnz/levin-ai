#!/bin/bash
# T-16: rerun the two 8-option conditions with 16 codes (spare codes), two lanes side by side.
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( python levin_scale.py --skip-grid --options 8 --codes 16 --out options8_c16 > logs/options8_c16.log 2>&1; touch logs/options8_c16.done ) &
( python levin_scale.py --skip-grid --side 25 --tissues 25 --irregular --options 8 --rule shift --codes 16 --out all_c16 > logs/all_c16.log 2>&1; touch logs/all_c16.done ) &
echo "started: options8_c16, all_c16"
