# Bugfix Requirements Document

## Introduction

UFM-Rec model significantly underperforms compared to TF-IDF baseline on cold-start items (items with 1-5 training interactions). The Extreme Cold NDCG@10 is 0.113 for UFM versus 0.192 for TF-IDF (gap of -0.078). This is the core research problem - the model should outperform or at least match TF-IDF on cold-start items, not underperform significantly.

The root cause is that the fusion gate doesn't effectively suppress the collaborative filtering branch for low-count items where CF embeddings are poorly trained due to limited gradient updates on random initialization.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN an item has 1-5 training interactions (extreme_cold) AND the user has a known history THEN the model produces NDCG@10 of 0.216 versus TF-IDF's 0.365 (gap of -0.149)

1.2 WHEN computing the fusion gate weights for an item with low train count THEN the gate relies on learned uncertainty from MLP which may not correlate with item popularity due to limited training data

1.3 WHEN the count_aware_fusion variant is used THEN the count penalty `u_cf = u_cf + count_penalty` is applied but the penalty strength may be insufficient to suppress CF for extreme cold items

1.4 WHEN items have 1-5 interactions THEN their collaborative embeddings are poorly trained (random initialization + few gradient updates produce noisy vectors that harm predictions)

1.5 WHEN computing cold macro NDCG@10 (average of zero_shot, extreme_cold, cold regimes) THEN UFM achieves 0.164 versus TF-IDF's 0.191 (gap of -0.028)

### Expected Behavior (Correct)

2.1 WHEN an item has 1-5 training interactions (extreme_cold) AND the user has a known history THEN the system SHALL produce NDCG@10 >= 0.19 (matching or beating TF-IDF baseline)

2.2 WHEN computing fusion gate weights for an item with low train count THEN the system SHALL suppress the collaborative filtering branch in favor of the semantic branch

2.3 WHEN the cold-start optimized variant is used THEN the system SHALL apply a count-aware constraint that forces CF weight toward zero for items below a threshold

2.4 WHEN items have 1-5 interactions THEN the system SHALL rely primarily on semantic embeddings (foundation model features) which are well-formed regardless of interaction count

2.5 WHEN computing cold macro NDCG@10 THEN the system SHALL achieve >= 0.19 (matching or beating TF-IDF baseline)

### Unchanged Behavior (Regression Prevention)

3.1 WHEN an item has >5 training interactions (warm regime) AND the user has a known history THEN the system SHALL CONTINUE TO produce NDCG@10 >= 0.65 (no degradation on warm items)

3.2 WHEN an item has zero interactions (zero_shot regime) THEN the system SHALL CONTINUE TO rely solely on semantic embeddings (CF branch already unavailable)

3.3 WHEN an item has >5 training interactions THEN the system SHALL CONTINUE TO use the collaborative filtering branch with learned uncertainty-based fusion

3.4 WHEN the user has empty history THEN the system SHALL CONTINUE TO produce NDCG@10 of 0.000 (no change to empty history behavior)

3.5 WHEN computing warm regime NDCG@10 THEN the system SHALL CONTINUE TO achieve >= 0.65 with the existing fusion mechanism
