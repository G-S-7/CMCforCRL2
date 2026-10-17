# Continual learning: SAC baseline vs SAC + CMC

Tasks: T1 push-v2 → T2 push-wall-v2 → T3 shelf-place-v2  
Paired seeds: [1, 2, 3] (values are mean ± std over seeds)

Evaluation protocol identical in both arms (same tasks, budget, episode count and evaluation episodes).

## ⚠ Tasks not reliably learned

A task that was never reliably learned cannot meaningfully be forgotten; read its numbers with care.

- SAC baseline: T1 reached only 0.72 success right after training (< 0.8).
- SAC baseline: T2 reached only 0.70 success right after training (< 0.8).
- SAC baseline: T3 reached only 0.57 success right after training (< 0.8).
- SAC + CMC: T1 reached only 0.72 success right after training (< 0.8).
- SAC + CMC: T3 reached only 0.26 success right after training (< 0.8).

## Success rate (primary)

### SAC baseline

| after stage | T1 push-v2 | T2 push-wall-v2 | T3 shelf-place-v2 |
|---|---|---|---|
| after T1 | **0.72 ± 0.10** | – | – |
| after T2 | 0.32 ± 0.16 | **0.70 ± 0.35** | – |
| after T3 | 0.02 ± 0.02 | 0.00 ± 0.00 | **0.57 ± 0.10** |

### SAC + CMC

| after stage | T1 push-v2 | T2 push-wall-v2 | T3 shelf-place-v2 |
|---|---|---|---|
| after T1 | **0.72 ± 0.10** | – | – |
| after T2 | 0.36 ± 0.09 | **0.88 ± 0.06** | – |
| after T3 | 0.14 ± 0.06 | 0.54 ± 0.14 | **0.26 ± 0.19** |

### Continual-learning metrics (success)

| group | metric | SAC baseline | SAC + CMC | CMC − baseline |
|---|---|---|---|---|
| New-task learning | T1 after T1 | 0.72 ± 0.10 | 0.72 ± 0.10 | 0.00 ± 0.00 |
| New-task learning | T2 after T2 | 0.70 ± 0.35 | 0.88 ± 0.06 | 0.18 ± 0.39 |
| New-task learning | T3 after T3 | 0.57 ± 0.10 | 0.26 ± 0.19 | -0.31 ± 0.25 |
| Retained value | T1 after T2 | 0.32 ± 0.16 | 0.36 ± 0.09 | 0.03 ± 0.22 |
| Retained value | T1 after T3 | 0.02 ± 0.02 | 0.14 ± 0.06 | 0.12 ± 0.07 |
| Retained value | T2 after T3 | 0.00 ± 0.00 | 0.54 ± 0.14 | 0.54 ± 0.14 |
| Forgetting | T1: after T1 − after T2 | 0.40 ± 0.10 | 0.37 ± 0.18 | -0.03 ± 0.22 |
| Forgetting | T1: after T1 − after T3 | 0.70 ± 0.11 | 0.58 ± 0.04 | -0.12 ± 0.07 |
| Forgetting | T2: after T2 − after T3 | 0.70 ± 0.35 | 0.33 ± 0.14 | -0.37 ± 0.30 |
| Summary | Average final (all tasks after T3) | 0.20 ± 0.03 | 0.31 ± 0.10 | 0.12 ± 0.12 |
| Summary | Average forgetting (T1, T2 after T3) | 0.70 ± 0.22 | 0.46 ± 0.07 | -0.24 ± 0.19 |

Forgetting > 0 means success dropped after later training; CMC − baseline < 0 for forgetting means CMC forgot less.

## Return (secondary)

### SAC baseline

| after stage | T1 push-v2 | T2 push-wall-v2 | T3 shelf-place-v2 |
|---|---|---|---|
| after T1 | **2931 ± 272** | – | – |
| after T2 | 1306 ± 502 | **2940 ± 1334** | – |
| after T3 | 169 ± 126 | 43 ± 15 | **1716 ± 338** |

### SAC + CMC

| after stage | T1 push-v2 | T2 push-wall-v2 | T3 shelf-place-v2 |
|---|---|---|---|
| after T1 | **2931 ± 272** | – | – |
| after T2 | 1232 ± 312 | **3811 ± 412** | – |
| after T3 | 421 ± 189 | 2087 ± 515 | **615 ± 508** |

### Continual-learning metrics (return)

| group | metric | SAC baseline | SAC + CMC | CMC − baseline |
|---|---|---|---|---|
| New-task learning | T1 after T1 | 2931 ± 272 | 2931 ± 272 | 0 ± 0 |
| New-task learning | T2 after T2 | 2940 ± 1334 | 3811 ± 412 | 870 ± 1647 |
| New-task learning | T3 after T3 | 1716 ± 338 | 615 ± 508 | -1101 ± 545 |
| Retained value | T1 after T2 | 1306 ± 502 | 1232 ± 312 | -73 ± 813 |
| Retained value | T1 after T3 | 169 ± 126 | 421 ± 189 | 252 ± 137 |
| Retained value | T2 after T3 | 43 ± 15 | 2087 ± 515 | 2044 ± 508 |
| Forgetting | T1: after T1 − after T2 | 1626 ± 357 | 1699 ± 554 | 73 ± 813 |
| Forgetting | T1: after T1 − after T3 | 2762 ± 335 | 2511 ± 249 | -252 ± 137 |
| Forgetting | T2: after T2 − after T3 | 2897 ± 1325 | 1723 ± 755 | -1174 ± 1562 |
| Summary | Average final (all tasks after T3) | 643 ± 128 | 1041 ± 290 | 398 ± 341 |
| Summary | Average forgetting (T1, T2 after T3) | 2830 ± 772 | 2117 ± 256 | -713 ± 847 |

## Figures

![matrix](matrix_success.png)

![per-task success across stages](stage_success.png)

![forgetting](forgetting.png)

![learning curves](learning_curves.png)
