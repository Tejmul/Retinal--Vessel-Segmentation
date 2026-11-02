## Ablation: SGL pseudo-labels x clDice loss (last checkpoint)

Identical network, split, schedule, optimiser, augmentation and seed (123) across all four rows; the loss is the only change.

| # | configuration | SGL | clDice | Acc | AUC | Se | Sp | Dice/F1 | clDice | IoU-v | beta0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R | published weights in our code | yes (K=8) | no | 0.9705 | 0.9886 | 0.8380 | 0.9834 | 0.8316 | n/a | 0.7120 | n/a |
| B | human labels, BCE | no | no | 0.9695 | 0.9872 | 0.8012 | 0.9860 | 0.8200 | 0.8118 | 0.6956 | 32.59 |
| C | clDice only | no | yes | 0.9599 | 0.9818 | 0.8677 | 0.9690 | 0.7904 | 0.8595 | 0.6537 | 8.15 |
| A | SGL baseline (paper loss) | yes | no | 0.9630 | 0.9849 | 0.8713 | 0.9720 | 0.8038 | 0.8319 | 0.6723 | 29.64 |
| D | SGL + clDice (proposed) | yes | yes | 0.9422 | 0.9764 | 0.8833 | 0.9480 | 0.7274 | 0.8565 | 0.5719 | 9.55 |

### Paired per-image differences (20 test images)

| comparison | metric | mean delta | std | images improved | Wilcoxon p |
|---|---|---|---|---|---|
| D - A  (clDice on top of SGL) | Acc | -0.0208 | 0.0049 | 0/20 | 0.000 |
| D - A  (clDice on top of SGL) | AUC | -0.0085 | 0.0042 | 0/20 | 0.000 |
| D - A  (clDice on top of SGL) | Se | +0.0120 | 0.0280 | 12/20 | 0.097 |
| D - A  (clDice on top of SGL) | Sp | -0.0240 | 0.0073 | 0/20 | 0.000 |
| D - A  (clDice on top of SGL) | Dice/F1 | -0.0764 | 0.0141 | 0/20 | 0.000 |
| D - A  (clDice on top of SGL) | clDice | +0.0247 | 0.0109 | 20/20 | 0.000 |
| D - A  (clDice on top of SGL) | IoU-v | -0.1003 | 0.0184 | 0/20 | 0.000 |
| D - A  (clDice on top of SGL) | beta0 | -20.0912 | 16.2484 | 0/20 | 0.000 |
| C - B  (clDice without SGL) | Acc | -0.0096 | 0.0032 | 0/20 | 0.000 |
| C - B  (clDice without SGL) | AUC | -0.0054 | 0.0036 | 1/20 | 0.000 |
| C - B  (clDice without SGL) | Se | +0.0665 | 0.0314 | 20/20 | 0.000 |
| C - B  (clDice without SGL) | Sp | -0.0170 | 0.0033 | 0/20 | 0.000 |
| C - B  (clDice without SGL) | Dice/F1 | -0.0296 | 0.0188 | 1/20 | 0.000 |
| C - B  (clDice without SGL) | clDice | +0.0477 | 0.0162 | 20/20 | 0.000 |
| C - B  (clDice without SGL) | IoU-v | -0.0419 | 0.0266 | 1/20 | 0.000 |
| C - B  (clDice without SGL) | beta0 | -24.4479 | 16.8813 | 0/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | Acc | -0.0066 | 0.0051 | 1/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | AUC | -0.0023 | 0.0019 | 0/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | Se | +0.0701 | 0.0226 | 20/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | Sp | -0.0140 | 0.0044 | 0/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | Dice/F1 | -0.0162 | 0.0268 | 5/20 | 0.009 |
| A - B  (SGL pseudo-labels alone) | clDice | +0.0201 | 0.0142 | 18/20 | 0.000 |
| A - B  (SGL pseudo-labels alone) | IoU-v | -0.0233 | 0.0376 | 5/20 | 0.007 |
| A - B  (SGL pseudo-labels alone) | beta0 | -2.9508 | 6.0610 | 7/20 | 0.059 |

### Verdict on the stated hypothesis

| prediction | observed | holds? |
|---|---|---|
| Se rises by 2-5 pp | +1.20 pp | no |
| F1 rises by 0.5-1.5 pp | -7.64 pp | no |
| Sp falls by at most 1 pp | -2.40 pp | no |

clDice metric changed by +2.47 pp. The review set a 0.01 (1 pp) F1 noise floor on this 20-image test set, so an F1 move of 7.64 pp is above that floor and should be read as real without repeating it across seeds.
