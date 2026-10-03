# UFM validation audit

Run: `ufm_full_v1`. Validation only; test untouched.

| Regime | History | Cases | UFM NDCG@10 | TF-IDF NDCG@10 | Difference |
|---|---|---:|---:|---:|---:|
| zero_shot | all | 10000 | 0.188403 | 0.186491 | +0.001912 |
| zero_shot | known_history | 5129 | 0.367329 | 0.363602 | +0.003727 |
| zero_shot | empty_history | 4871 | 0.000000 | 0.000000 | +0.000000 |
| extreme_cold | all | 10000 | 0.113300 | 0.191555 | -0.078255 |
| extreme_cold | known_history | 5247 | 0.215933 | 0.365075 | -0.149142 |
| extreme_cold | empty_history | 4753 | 0.000000 | 0.000000 | +0.000000 |
| cold | all | 10000 | 0.189511 | 0.195763 | -0.006252 |
| cold | known_history | 5300 | 0.339796 | 0.351592 | -0.011796 |
| cold | empty_history | 4700 | 0.020040 | 0.020040 | +0.000000 |
| warm | all | 10000 | 0.651231 | 0.467661 | +0.183570 |
| warm | known_history | 4925 | 0.678716 | 0.305984 | +0.372732 |
| warm | empty_history | 5075 | 0.624559 | 0.624559 | +0.000000 |

Cold macro difference: -0.027532; exploratory user-bootstrap 95% CI [-0.030221187790143055, -0.025081582306626247].

Temperature: 1.789353; fit=20000, audit=20000.

| Audit metric | Before | After |
|---|---:|---:|
| ece | 0.164477 | 0.041497 |
| nll | 4.309607 | 3.906583 |
| brier_multiclass | 0.963379 | 0.927849 |

Validation was already used for checkpoint selection; calibration and CI are exploratory.
Confidence is conditional on sampled candidates, not purchase probability.
One training seed; user bootstrap does not measure training-seed variability.
