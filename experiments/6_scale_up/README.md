# Scale-up test

Does steering through discovered parts survive larger, less tidy worlds? Each condition changes one thing from the
base setup (15×15 cells, 9 square tissues, 4 options, independent rule), plus one with every change at once.

| Condition | Command |
|---|---|
| Bigger world (25 tissues) | `python levin_scale.py --skip-grid --side 25 --tissues 25 --out big` |
| Irregular tissue shapes | `python levin_scale.py --skip-grid --irregular --out irregular` |
| 8 options per part | `python levin_scale.py --skip-grid --options 8 --out options8` |
| Interacting parts (shift rule) | `python levin_scale.py --skip-grid --rule shift --out shift` |
| All of the above | `python levin_scale.py --skip-grid --side 25 --tissues 25 --irregular --options 8 --rule shift --out all` |
| Retraining comparison, 25 known parts | `python ../3_grid_2d/levin2d.py --grid 5 --rule shift --out retrain_big` |

Pass criteria, fixed before running: valid if steering through true parts reaches at least 80%; passes if steering
through discovered parts is within 10 points of true parts.

`run_all.sh` runs the conditions in three lanes side by side on one GPU (one run at a time per lane, so memory fits on a T4) and writes a log per condition to `logs/`.

## Search in the loop (T-19)

`run_search.sh` trains with a short square-by-square search inside training (`--search-train`, `--search-start`,
`--search-sweeps`, `--search-distill`), so that codes found by search are the ones the model learns to use. Its
header fixes the pass criteria: `bash run_search.sh screen` first, then `bash run_search.sh full VARIANT` with the
variant the screen rule picks. With these options off, the script behaves exactly as before.
