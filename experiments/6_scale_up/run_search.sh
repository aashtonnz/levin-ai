#!/bin/bash
# T-19: train with search in the loop (irregular worlds, 4 options, small squares 6x6 or nested squares 3,6).
#
# Method. For the first 30% of training, codes come from the encoder (as in T-17). After that, for half of each
# batch (--search-train 0.5) the codes come from a short square-by-square search with the current weights: start
# from code 0 everywhere; for each colour group of squares in turn, try every code and keep, for each square, the
# one with the lowest error over that square's own cells; 2 sweeps. The search has no gradient. The model is then
# trained so that those searched codes draw the target, and the encoder is trained to predict the searched codes.
# The best version of each seed is judged by held-out error under searched codes, counted only after search starts.
# The test procedures are unchanged from T-17 (search from code 0, 4 sweeps; and probe-influence search).
#
# Fixed before running (2026-11-24):
#   Stage 1, screen (4 seeds, irregular world 0), three variants:
#     s6_f50  small squares 6x6, half of each batch searched
#     s36_f50 nested squares 3,6, half of each batch searched
#     s6_f100 small squares 6x6, every example searched (encoder learns only from the searched codes)
#   Screen rule (as T-18): a variant passes the screen if best search (higher of the two test searches, mean over
#   seeds) >= 80%. Among passing variants, the one with the highest best search goes to stage 2. If none passes,
#   T-19 fails and there is no stage 2.
#   Stage 2, full test (8 seeds), the chosen variant, with the T-17 criteria:
#     H1 (development world, irregular seed 0): best search >= 89% (true parts got 99%; within 10 points).
#     H2 (held-out world, irregular seed 1): best search within 10 points of true parts on world 1 (T-17 run
#         true_irr1: 100%, same code path), so >= 90%.
#   Reported, not criteria: one-shot encoder accuracy, gradient-tuned codes, held-out error, run time.
#   Note: the variant is chosen on world 0, so H1 is not a clean test; H2 (world 1, unseen) is.
#
# Usage: bash run_search.sh screen          # stage 1, three runs side by side on one GPU
#        bash run_search.sh full s6_f50     # stage 2, with the variant chosen by the screen rule
cd "$(dirname "$0")"; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; ( python levin_scale.py --true-only --irregular --out $name "$@" > logs/$name.log 2>&1; touch logs/$name.done ) & }
flags() { case $1 in
  s6_f50)  echo "--hier 6 --search-train 0.5" ;;
  s36_f50) echo "--hier 3,6 --search-train 0.5" ;;
  s6_f100) echo "--hier 6 --search-train 1.0" ;;
  *) echo "unknown variant $1" >&2; exit 1 ;; esac; }
case $1 in
  screen)
    for v in s6_f50 s36_f50 s6_f100; do run srch_$v --seeds 4 $(flags $v); done
    echo "started: srch_s6_f50 srch_s36_f50 srch_s6_f100" ;;
  full)
    f=$(flags $2) || exit 1
    run full_$2_irr0 $f
    run full_$2_irr1 $f --world-seed 1
    echo "started: full_$2_irr0 full_$2_irr1" ;;
  *) echo "usage: bash run_search.sh screen | full VARIANT"; exit 1 ;;
esac
