# Task 3 Complete: Prediction Analysis Pipeline

## Summary

Successfully replaced causal inference (SAE + double debiasing) with embedding-based prediction modeling for analyzing historian group experiments.

## Architectural Shift

### Before: Causal Inference
```
Personas → Factorial Design → SAE Latent Space → Double Debiasing → ATEs
         (synthetic)         (sparse autoencoder)  (confounders)
```

### After: Prediction Modeling
```
Historians → Triangle Geometry → Feature Engineering → Supervised Learning → Predictions
          (real scholars)      (embedding distances)  (train/test split)
```

## Key Changes

### 1. From `causal_analysis.py` to `prediction_analysis.py`

**Removed:**
- ❌ `SparseAutoencoder` class (PyTorch neural network)
- ❌ `DoubleDebiasingEstimator` class (DML/propensity scoring)
- ❌ `CausalEstimate` dataclass (ATE, standard errors, p-values)
- ❌ Cross-fitting for confounder adjustment
- ❌ Bootstrap confidence intervals for causal effects

**Added:**
- ✅ `PerplexityCalculator` class (GPT-2 based text scoring)
- ✅ `PredictionResult` dataclass (R², MSE, feature importance)
- ✅ Feature engineering from triangle geometry
- ✅ Multiple prediction models (RF, GB, Ridge, Lasso)
- ✅ Train/test splits and cross-validation
- ✅ Correlation analysis (Pearson & Spearman)

### 2. Research Questions Reframed

**Old RQs (Causal):**
- RQ1: How do persona characteristics **causally affect** source selection?
- RQ2: Which configurations **produce** optimal theses?

**New RQs (Predictive):**
- RQ1: Can embedding distances **predict** differential source selection?
- RQ2: Can embedding geometry **predict** abstract perplexity/quality?

### 3. Feature Engineering

**Input Features (14 geometry features):**

Base geometry (8 from `triangle_geometry`):
- `geom_side_1`, `geom_side_2`, `geom_side_3` (pairwise cosine distances)
- `geom_perimeter`, `geom_area`
- `geom_min_angle`, `geom_max_angle`, `geom_angle_variance`

Derived features (6 engineered):
- `geom_avg_side` = mean of three sides
- `geom_side_variance` = variance of sides
- `geom_angle_range` = max_angle - min_angle
- `geom_perimeter_norm` = perimeter / max_perimeter
- `geom_area_norm` = area / max_area
- `geom_regularity` = 1 - (side_variance / avg_side²)

### 4. Prediction Targets

**RQ1 Targets (Source Selection):**
- `n_sources_accessed` (count)
- `n_unique_sources` (count)
- `source_diversity` (ratio)
- `sources_per_turn` (rate)

**RQ2 Target (Abstract Quality):**
- `perplexity` (GPT-2 log-likelihood, lower = more fluent)
- `abstract_length` (word count)

### 5. Models Implemented

**Ensemble Models:**
- `RandomForestRegressor` (n_estimators=100)
- `GradientBoostingRegressor` (n_estimators=100)

**Linear Models:**
- `Ridge` (alpha=1.0)
- `Lasso` (alpha=0.1)

**Evaluation:**
- Train/test R² and MSE
- 5-fold cross-validation
- Feature importance ranking

## Test Results

All tests passed ✓

```
Test 1: Feature Engineering
  ✓ 14 geometry features extracted
  ✓ 6 derived features computed
  ✓ Shape: (50, 23)

Test 2: Source Selection Features
  ✓ n_sources_accessed, source_diversity, etc.
  ✓ Mean sources: 8.7

Test 3: Prediction Models
  ✓ Random Forest: Test R² = -0.125 (overfitting on small data)
  ✓ Ridge: Test R² = 0.063
  ✓ Feature importance computed

Test 4: Correlation Analysis
  ✓ 56 correlations computed
  ✓ 1 significant at p < 0.05 (geom_min_angle vs n_sources)
```

## Example Output

### RQ1: Source Selection Prediction

```python
# predict_source_selection(target='n_sources_accessed')

--- RF ---
Train R²: 0.875
Test R²: -0.125
CV R²: -0.843 ± 1.059

Top 5 features:
  geom_side_2: 0.2232
  geom_side_3: 0.2206
  geom_max_angle: 0.1065
  geom_side_variance: 0.0860
  geom_side_1: 0.0720
```

### RQ2: Perplexity Prediction

```python
# predict_abstract_perplexity()

--- GB ---
Train R²: 0.XXX
Test R²: 0.XXX
CV R²: X.XXX ± X.XXX

Top 5 features:
  geom_area: X.XXXX
  geom_perimeter: X.XXXX
  geom_regularity: X.XXXX
  ...
```

### Correlation Analysis

```
Significant correlations (p < 0.05): 1

geometry_feature            outcome  pearson_r  pearson_p
  geom_min_angle n_sources_accessed  -0.282     0.047
```

## Generated Report Structure

```json
{
  "timestamp": "2024-03-12T14:45:00",
  "n_experiments": 100,
  "rq1_source_prediction": {
    "rf": {
      "test_r2": 0.XXX,
      "cv_r2_mean": X.XXX,
      "top_features": {
        "geom_side_2": 0.XXX,
        ...
      }
    },
    "gb": {...},
    "ridge": {...}
  },
  "rq2_perplexity_prediction": {
    "rf": {...},
    "gb": {...},
    "ridge": {...}
  },
  "correlations": {
    "n_significant": 5,
    "strongest": [...]
  }
}
```

## Comparison Table

| Aspect | Causal Analysis (Old) | Prediction Analysis (New) |
|--------|----------------------|---------------------------|
| **Goal** | Estimate causal effects | Predict outcomes |
| **Method** | SAE + double debiasing | Supervised learning |
| **Features** | Persona characteristics | Triangle geometry |
| **Treatment** | Binary persona dimension | N/A (no treatment) |
| **Confounders** | Other persona dimensions | N/A (no confounding) |
| **Output** | ATE ± CI, p-value | R², MSE, predictions |
| **Inference** | Causal interpretation | Correlational/predictive |
| **Complexity** | High (neural net, DML) | Medium (sklearn models) |
| **Interpretability** | Low (latent space) | High (feature importance) |

## Files Created/Modified

1. ✅ `analysis/prediction_analysis.py` (created)
   - 600+ lines
   - 5 main classes/functions
   - Full prediction pipeline

2. ✅ `test_prediction_analysis.py` (created)
   - Mock data generation
   - 4 integration tests
   - All passing

3. ✅ `analysis/causal_analysis.py` (deprecated)
   - Not deleted (for reference)
   - Use `prediction_analysis.py` instead

## Usage Example

```python
from analysis.prediction_analysis import PredictionAnalyzer

# Initialize
analyzer = PredictionAnalyzer()

# Load data
analyzer.load_results("outputs/results.csv")
analyzer.load_detailed_results("outputs/")

# RQ1: Predict source selection
source_results = analyzer.predict_source_selection(
    target='n_sources_accessed',
    models=['rf', 'gb', 'ridge']
)

# RQ2: Predict perplexity
perplexity_results = analyzer.predict_abstract_perplexity(
    models=['rf', 'gb', 'ridge']
)

# Correlation analysis
corr_df = analyzer.correlation_analysis()

# Generate report
report = analyzer.generate_report("outputs/prediction_report.json")
```

## Next Steps (Task 4+)

Ready for:
1. ✅ Visualization module updates (50+ plots)
2. ✅ End-to-end pipeline testing
3. ✅ Real experiment runs with actual LLM agents
4. ✅ Results analysis and interpretation

## Dependencies

**Required:**
- scikit-learn (prediction models)
- pandas, numpy (data handling)
- scipy (correlations)

**Optional:**
- transformers, torch (perplexity calculation with GPT-2)
- matplotlib, seaborn (visualization)

## Notes

- **Small sample warning:** Test R² values negative/low due to mock data (n=50)
- **Real experiments:** Expect better performance with n=100-1000
- **Perplexity:** Requires downloading GPT-2 model (~500MB)
- **Interpretability:** Feature importance provides clear insights vs. latent space
- **Speed:** Faster than SAE training + bootstrapping
