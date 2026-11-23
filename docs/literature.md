# Related work (first pass, 2026-11-24)

A quick scan to see what already exists near this project. Not exhaustive; to be extended before any write-up.

## Neural cellular automata (NCA) with goals or conditions
- **Growing neural cellular automata** (Mordvintsev, Randazzo, Niklasson, Levin, Distill 2020): the base method; a shared local rule grows and regrows a target image.
- **Goal-guided NCA** (Sudhakaran et al., 2022, arXiv:2205.06806): goal encodings change cell behaviour during growth; robust when only some cells receive the goal. Goals are trained, not searched, and cover the whole pattern.
- **Conditional NCA** (Sakour, 2025, arXiv:2512.08360): a class vector broadcast to every cell grows different MNIST digits with one rule.
- **Reusable regulatory handles** (Hartl, Montero, Barylli, Risi, Levin, arXiv:2609.29755): low-rank weight directions (LoRA) steer a frozen NCA's growth, such as scaling, and can be combined. The handles are global, not tied to parts, and isolating feature-specific directions proved hard. Closest in spirit to this project.
- Levin lab overviews: Hartl, Levin, Pio-Lopez, *Neural cellular automata: applications to biology and beyond classical AI* (arXiv:2509.11131); *BraiNCA* (arXiv:2604.01932).

## Modularity from a cost on connections
- Clune, Mouret, Lipson (2013, PMC3574393): a cost on connections makes evolved networks modular.
- Brain-inspired modular training, BIMT (Liu, Khona, Tegmark, arXiv:2305.08746) and *Growing Brains* (Liu, Khona, Fiete, Tegmark, arXiv:2310.07711): a connection-length cost gives visibly modular, more interpretable networks.
- Qian, Liang, Fiete (PRX Life 2026): noise-motivated regularisation yields modular connectivity that improves robustness and compositional generalisation.

## Finding modules in trained networks
- Filan et al., *Clusterability in neural networks* (arXiv:2003.04881 → 2103.03386): spectral clustering of weights; pruning and dropout make networks more modular.
- Golechha et al. (arXiv:2502.02470): standard models form enmeshed clusters; a clusterability loss makes separable modules, though they did not specialise more than in ordinary models.

## Steering frozen models without retraining
- Steer2Adapt (arXiv:2602.07276): adapts a language model to a new task by searching a combination of existing steering vectors from a handful of examples.
- Compositional steering tokens (arXiv:2601.05062) and vision–language steering of frozen robot policies (arXiv:2602.03973).

## What looks new here, tentatively
Each ingredient exists on its own: goal-conditioned NCAs, connection-cost modularity, clustering to find modules, and steering frozen models by search over a few controls. Not yet seen together:
1. one goal code **per part**, rather than a global goal or global weight direction;
2. parts **discovered** from learned, costly coupling and used directly as the control panel;
3. **search** over those per-part codes, with influence probing when parts interact, compared head-to-head with retraining and prompt tuning.

The novelty claim rests on that combination and on whether it holds beyond toy worlds. Comparisons a write-up would need: goal-guided NCA (trained global goals), LoRA-style regulatory handles, and search over steering-vector combinations.
