#!/bin/bash
# Start every scale-up condition at once on one GPU; one log per condition in logs/.
cd "$(dirname "$0")"; mkdir -p logs
run() { name=$1; shift; nohup python "$@" > logs/$name.log 2>&1 & }
run big       levin_scale.py --skip-grid --side 25 --tissues 25 --out big
run irregular levin_scale.py --skip-grid --irregular --out irregular
run options8  levin_scale.py --skip-grid --options 8 --out options8
run shift     levin_scale.py --skip-grid --rule shift --out shift
run all       levin_scale.py --skip-grid --side 25 --tissues 25 --irregular --options 8 --rule shift --out all
run retrain_big ../3_grid_2d/levin2d.py --grid 5 --rule shift --out retrain_big
echo "started: big irregular options8 shift all retrain_big"
