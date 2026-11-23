# Decisions

Newest first. Each line: date, ID and name, what was decided and why.

- 2026-11-24. D-11: Run scale-up conditions side by side. The conditions run as separate processes on one GPU at the same time, since each alone leaves most of the GPU idle; results are unaffected.
- 2026-11-24. D-10: Scale-up pass criteria, fixed before running. A condition is valid if steering through true parts reaches at least 80%; it passes if steering through discovered parts is within 10 points of true parts. Gradient-tuned codes are reported but not a criterion, since in this setting they act on the same good panel; the comparison with retraining is rerun separately on the larger grid with known parts.
- 2026-11-24. D-9: Restart seeds that fail to learn. After training, a seed whose held-out error is more than 10× the median seed is retrained from a new start and the better version kept, the same rule for every panel; occasional seeds never learn, and this is detectable without labels.
- 2026-11-24. D-8: Keep each seed's best version. Each seed is checked on held-out examples every 100 steps and its best version kept (early stopping); some seeds learned well and then broke late in training.
- 2026-11-23. D-7: Spectral clustering finds the parts. Replaced cutting weak links and merging fragments; it judges whole regions rather than single links, which raised the match to true tissues from about 0.6 to 0.9.
- 2026-11-23. D-6: Window encoder. Each part's code is chosen from a 7×7 window of the target around the part's centre, the same for every panel; the earlier pooled encoder gave two different goals the same code.
- 2026-11-23. D-5: Same position cue for every panel. Each cell gets its offset from its part's centre and whether it is on the part's edge, so that part size alone does not decide steering results.
- 2026-11-23. D-4: Gradually introduced coupling cost. The cost on communication is off for the first 30% of training and reaches full strength at 70%; at full strength from the start, no parts formed.
- 2026-11-23. D-3: Pass criteria fixed before the 2D test. Valid only if correct settings by hand reach at least 80%; search must be within 2 points of that and at least 30 points above the best baseline.
- 2026-11-23. D-2: Learning-rate schedule. The learning rate drops at 40% and 70% of training, after a late blow-up spoiled the first 2D shift-rule run.
- 2026-11-22. D-1: Toy grids first, GPU in Colab. Fast, cheap tests on small grids before anything larger; full runs on the owner's Colab GPU.
