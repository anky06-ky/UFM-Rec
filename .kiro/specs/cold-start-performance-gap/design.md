# Cold-Start Performance Gap Bugfix Design

## Overview

This design addresses the significant underperformance of UFM-Rec on cold-start items compared to the TF-IDF baseline. The Extreme Cold NDCG@10 is 0.113 for UFM versus 0.192 for TF-IDF (gap of -0.078). The root cause is that the fusion gate doesn't effectively suppress the collaborative filtering branch for low-count items where CF embeddings are poorly trained.

We implement three new variants: `count_aware_hard` (hard constraint), `hybrid_confidence_gate` (learned + count-based), and `cold_adapter` (separate adapter network). Each variant forces the model to rely primarily on semantic embeddings for cold items while preserving CF for warm items.

## Glossary

- **Bug_Condition (C)**: The condition where an item has 1-5 training interactions (extreme_cold regime) and the model underperforms compared to TF-IDF baseline
- **Property (P)**: The desired behavior where the model achieves NDCG@10 >= 0.19 for extreme_cold items by suppressing CF for cold items
- **Preservation**: Existing warm-item behavior (NDCG@10 >= 0.65) and non-CF functionality must remain unchanged
- **isBugCondition**: Function that determines if an item is in the bug-affected regime (train_count <= threshold)
- **train_count**: Number of training interactions for an item, used to determine cold-start regime
- **fusion_gate**: Mechanism that combines CF and semantic branch outputs with weights based on uncertainty
- **threshold**: Configurable parameter (default: 5) below which items are considered "cold"
- **cold_adapter**: Lightweight neural network specifically for processing cold items using only semantic features

## Bug Details

### Bug Condition

The bug manifests when an item has 1-5 training interactions (extreme_cold regime). The fusion gate mechanism (`reliability_weights` function) relies on learned uncertainty from MLP which doesn't adequately correlate with item popularity due to limited training data. CF embeddings for cold items are poorly trained (random initialization + few gradient updates) producing noisy vectors that harm predictions.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ItemContext(item_id, train_count, user_history)
  OUTPUT: boolean

  RETURN input.train_count >= 1
         AND input.train_count <= 5
         AND input.user_history IS NOT empty
         AND model_ndcg(input) < tfidf_ndcg(input)
END FUNCTION
```

### Examples

- **Extreme Cold Item (train_count=2)**: UFM produces NDCG@10=0.216 vs TF-IDF's 0.365 (gap=-0.149). CF embedding is noisy due to limited training.
- **Zero Shot Item (train_count=0)**: System correctly falls back to semantic-only (CF unavailable). No bug - demonstrates semantic branch capability.
- **Warm Item (train_count=50)**: UFM achieves NDCG@10=0.65+ with CF branch contributing significantly. No bug - CF branch is well-trained.
- **Boundary Case (train_count=5)**: Edge of extreme_cold regime. CF may be marginally useful but semantic should dominate.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Warm items (train_count > 5) must continue to achieve NDCG@10 >= 0.65
- Zero-shot items (train_count = 0) must continue to use semantic-only (no CF available)
- Empty user history must continue to produce NDCG@10 of 0.000
- Existing variants (full, count_aware_fusion, etc.) must continue to work unchanged
- All command-line arguments for existing variants must remain compatible

**Scope:**
All inputs that do NOT involve items with 1-5 training interactions should be completely unaffected by this fix. This includes:
- Items with train_count = 0 (zero-shot regime)
- Items with train_count > 5 (warm regime)
- Empty user histories
- Non-extreme_cold evaluation metrics

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Inadequate Count Penalty in count_aware_fusion**: The current penalty `u_cf = u_cf + count_penalty` uses `count_penalty = exp(-train_counts[candidates].float() / 5.0)` which may be insufficient for extreme cold items. For train_count=1, this adds only ~0.82 to uncertainty, not enough to fully suppress CF.

2. **Learned Uncertainty Not Correlated with Popularity**: The `cf_uncertainty` MLP learns from training residuals but cold items have few training examples, so the MLP receives limited gradient signal for these items. The uncertainty values may not reflect true unreliability.

3. **CF Embeddings Are Noisy for Cold Items**: Random initialization plus 1-5 gradient updates produces embeddings that contain more noise than signal. These embeddings actively harm predictions when given significant weight in fusion.

4. **No Hard Fallback Mechanism**: The current fusion always allows some CF contribution (weights never forced to exactly 0 based on count alone). A hard threshold would guarantee semantic-only for extreme cold.

## Correctness Properties

Property 1: Bug Condition - Cold Item Semantic Dominance

_For any_ item where train_count is between 1 and threshold (inclusive) AND the user has a non-empty history, the fixed model SHALL suppress the collaborative filtering branch contribution such that the semantic branch weight >= 0.9, resulting in NDCG@10 >= 0.19 for extreme_cold items.

**Validates: Requirements 2.1, 2.2, 2.4**

Property 2: Preservation - Warm Item CF Utilization

_For any_ item where train_count > threshold AND the user has a non-empty history, the fixed model SHALL produce the same CF/semantic weight distribution as the original model within tolerance of 0.05, preserving NDCG@10 >= 0.65 for warm items.

**Validates: Requirements 3.1, 3.3, 3.5**

Property 3: Preservation - Zero-Shot and Empty History

_For any_ item where train_count = 0 OR the user has empty history, the fixed model SHALL produce exactly the same behavior as the original model (semantic-only or NDCG=0.000 respectively), preserving all zero-shot and empty-history functionality.

**Validates: Requirements 3.2, 3.4**

Property 4: Variant Backward Compatibility

_For any_ existing variant (full, no_uncertainty, fixed_fusion, etc.) that does NOT match the three new cold-start variants, the fixed model SHALL produce identical outputs to the original implementation for all inputs.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

**File**: `src/ufm_model.py`

**Changes**:

1. **UFMConfig Dataclass - Add New Parameters**:
   - Add `cold_threshold: int = 5` - threshold below which items are "cold"
   - Add `cold_alpha: float = 1.0` - strength of count-based prior for hybrid_confidence_gate
   - Add `cold_adapter_dim: int = 64` - hidden dimension for cold_adapter variant
   - Update variant validation to include `count_aware_hard`, `hybrid_confidence_gate`, `cold_adapter`

2. **UFMRec.__init__ - Add Cold Adapter Network**:
   - Add `self.cold_adapter = mlp(config.feature_dim * 2 + 2, config.cold_adapter_dim, config.dim)` for cold_adapter variant
   - Initialize cold adapter parameters with same pattern as main adapter

3. **UFMRec.forward - Add Variant-Specific Logic**:
   - **count_aware_hard**: After computing `cf_ok`, add:
     ```python
     if self.config.variant == "count_aware_hard":
         cold_candidates = train_counts[candidates] <= self.config.cold_threshold
         cf_ok = cf_ok & ~cold_candidates
     ```
   - **hybrid_confidence_gate**: Modify uncertainty computation:
     ```python
     if self.config.variant == "hybrid_confidence_gate":
         count_prior = self.config.cold_alpha * torch.exp(-train_counts[candidates].float() / self.config.cold_threshold)
         u_cf = u_cf + count_prior
     ```
   - **cold_adapter**: Add cold-specific processing:
     ```python
     if self.config.variant == "cold_adapter":
         cold_mask = train_counts[candidates] <= self.config.cold_threshold
         # Use cold_adapter for cold items, main adapter for warm
         cold_sem = self.cold_adapter(torch.cat([sem_user, cs, modality_flags.float()], -1))
         sem = torch.where(cold_mask.unsqueeze(-1), cold_sem, sem)
     ```

4. **Update variant validation in __post_init__**:
   - Add new variants to the validation set

**File**: `src/train_ufm_recommender.py`

**Changes**:

1. **Add Command-Line Arguments**:
   - `--cold-threshold` (int, default=5)
   - `--cold-alpha` (float, default=1.0)
   - `--cold-adapter-dim` (int, default=64)

2. **Update parse_args()**:
   - Add new arguments to argument parser
   - Add validation: cold_threshold >= 1, cold_alpha >= 0, cold_adapter_dim >= 1

3. **Pass new parameters to UFMConfig**:
   - Update model config creation to include new parameters

**File**: `scripts/run_cold_start_experiments.py` (NEW)

**Purpose**: Script to run comparative experiments across all cold-start variants

**Implementation**:
- Accept variant name and parameters as arguments
- Load training data and features
- Train model with specified variant
- Evaluate on cold-start regimes
- Output metrics to JSON

**File**: `configs/cold_start_variants.json` (NEW)

**Purpose**: Configuration file for all cold-start variant experiments

**Structure**:
```json
{
  "variants": {
    "count_aware_hard": {
      "threshold_values": [3, 5, 10],
      "expected_behavior": "Hard suppression of CF for items below threshold"
    },
    "hybrid_confidence_gate": {
      "threshold_values": [3, 5, 10],
      "alpha_values": [0.5, 1.0, 2.0],
      "expected_behavior": "Soft suppression via increased uncertainty prior"
    },
    "cold_adapter": {
      "threshold_values": [3, 5, 10],
      "adapter_dim_values": [32, 64, 128],
      "expected_behavior": "Separate adapter for cold items with semantic-only processing"
    }
  },
  "success_criteria": {
    "extreme_cold_ndcg@10_min": 0.19,
    "cold_macro_ndcg@10_min": 0.19,
    "warm_ndcg@10_min": 0.65
  }
}
```

### Implementation Details

**Variant 1: count_aware_hard**
- Simpler implementation - just mask out CF for cold items
- No additional learnable parameters
- Predictable behavior - always semantic-only below threshold
- Risk: May lose useful CF signal for borderline items

**Variant 2: hybrid_confidence_gate**
- Builds on existing count_aware_fusion pattern
- Configurable strength via alpha parameter
- Softer transition - allows some CF for near-threshold items
- Expected to be most robust across datasets

**Variant 3: cold_adapter**
- Most complex - adds new neural network
- Learns cold-specific representations
- Potentially better adaptation to cold regime
- Risk: May overfit to cold items if not regularized

## Testing Strategy

### Validation Approach

The testing strategy follows a three-phase approach: first, surface counterexamples demonstrating the bug on unfixed code; second, verify the fix works correctly for cold items; third, verify preservation of warm-item behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples demonstrating the bug BEFORE implementing the fix. Confirm root cause analysis.

**Test Plan**: Evaluate current model on cold-start items and compare CF weight distribution between cold and warm items.

**Test Cases**:
1. **Extreme Cold CF Weight Test**: Evaluate model on items with train_count=1-5, check that CF weight is > 0.3 (indicating insufficient suppression) - will fail on unfixed code
2. **Warm CF Weight Test**: Evaluate model on items with train_count>50, check that CF weight is > 0.5 (indicating CF contribution) - should pass on unfixed code
3. **Zero-Shot Semantic Test**: Evaluate model on items with train_count=0, verify semantic-only processing - should pass on unfixed code
4. **Count Penalty Analysis**: For count_aware_fusion variant, compute actual penalty values for train_count=1,2,3,4,5 and verify they don't suppress CF enough

**Expected Counterexamples**:
- CF weight remains > 0.3 even for train_count=1 items
- count_aware_fusion penalty of exp(-1/5)=0.82 insufficient to suppress CF
- NDCG@10 for extreme_cold significantly below TF-IDF baseline

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  output := UFMRec_fixed(input)
  ASSERT output.weights[..., 0] <= 0.1  -- CF weight <= 10%
  ASSERT output.ndcg@10 >= 0.19
END FOR
```

**Test Cases**:
1. **count_aware_hard Threshold Test**: For items with train_count <= threshold, verify CF weight = 0
2. **hybrid_confidence_gate Prior Test**: For items with train_count=1, verify u_cf increased by ~alpha*exp(-1/threshold)
3. **cold_adapter Routing Test**: For items with train_count <= threshold, verify cold_adapter network is used
4. **NDCG Improvement Test**: Verify extreme_cold NDCG@10 >= 0.19 for all three variants

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  output_original := UFMRec_original(input)
  output_fixed := UFMRec_fixed(input)
  ASSERT abs(output_original.scores - output_fixed.scores) < epsilon
END FOR
```

**Test Cases**:
1. **Warm Item Preservation**: For items with train_count > threshold, verify scores match original within tolerance
2. **Zero-Shot Preservation**: For items with train_count=0, verify exact same behavior (no CF available anyway)
3. **Empty History Preservation**: For users with empty history, verify same NDCG=0.000 behavior
4. **Variant Isolation Test**: Verify other variants (full, semantic_only, etc.) produce identical outputs

### Unit Tests

- `test_count_aware_hard_suppresses_cold()`: Verify CF weight = 0 for train_count <= threshold
- `test_hybrid_confidence_gate_increases_uncertainty()`: Verify u_cf increased by count prior
- `test_cold_adapter_routing()`: Verify correct adapter used based on train_count
- `test_warm_item_preservation()`: Verify warm items produce same scores as original
- `test_config_validation()`: Verify new config parameters validated correctly
- `test_variant_backward_compatibility()`: Verify existing variants unchanged

### Property-Based Tests

- Generate random train_counts and verify CF suppression for count <= threshold
- Generate random user histories and verify preservation for warm items
- Generate random item features and verify semantic branch processing unchanged
- Test threshold boundary conditions (count = threshold-1, threshold, threshold+1)

### Integration Tests

- Full training run with count_aware_hard variant, verify cold macro NDCG >= 0.19
- Full training run with hybrid_confidence_gate variant, verify cold macro NDCG >= 0.19
- Full training run with cold_adapter variant, verify cold macro NDCG >= 0.19
- Verify warm NDCG >= 0.65 for all three variants (no regression)
- Compare all three variants against TF-IDF baseline on cold-start metrics
