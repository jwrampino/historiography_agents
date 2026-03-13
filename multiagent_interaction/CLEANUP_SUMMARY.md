# Pipeline Cleanup Summary

## What Was Done

### ✅ Removed Pipeline 2 (Causal Inference)

**Deleted files:**
1. `personas/persona_manager.py` - Synthetic persona grid
2. `analysis/causal_analysis.py` - SAE + double debiasing
3. `run_experiment.py` - Old entry point
4. `example_usage.py` - Old examples
5. `visualize.py` - Old standalone visualization

### ✅ Updated Pipeline 1 (Prediction-Based)

**Fixed imports:**
- `personas/__init__.py` → now exports `HistorianManager`
- `analysis/__init__.py` → now exports `PredictionAnalyzer`
- `experiments/experiment_runner.py` → fixed path imports
- `agents/multi_agent_system.py` → fixed path imports
- `analysis/prediction_analysis.py` → fixed path imports
- `visualization/generate_all.py` → fixed path imports

**Connected to real data:**
- `sources/source_library.py` → added `load_from_historian_pipeline()`
- `config/config.yaml` → points to `../data/index`
- Lazy loading from `../data/raw/*/metadata/*.json`

### ✅ Clean File Structure

```
multiagent_interaction/
├── README.md                       # Overview + architecture
├── QUICKSTART.md                   # Your exact commands
├── personas/
│   ├── historian_manager.py        # ✓ Pipeline 1
│   └── historian_personas.json
├── sources/
│   └── source_library.py           # ✓ Loads from ../data
├── agents/
│   └── multi_agent_system.py       # ✓ LangGraph dialogue
├── experiments/
│   └── experiment_runner.py        # ✓ Main orchestrator
├── analysis/
│   └── prediction_analysis.py      # ✓ RF, GB, Ridge models
├── visualization/
│   ├── generate_all.py             # ✓ Master generator
│   ├── geometry_viz.py
│   ├── prediction_viz.py
│   └── experiment_viz.py
├── test_*.py                       # ✓ All tests
└── config/
    └── config.yaml                 # ✓ Points to ../data
```

## Pipeline 1: End-to-End

```
1. personas/historian_manager.py
   Load 25 real historians from OpenAlex
   ↓
2. sources/source_library.py
   Connect to ../data/index/corpus.faiss
   ↓
3. experiments/experiment_runner.py
   Run N experiments with 3 historians each
   ↓
4. analysis/prediction_analysis.py
   Predict outcomes using regression models
   ↓
5. visualization/generate_all.py
   Generate 50+ visualizations
```

## Research Questions

**RQ1:** Can embedding geometry predict source selection?
- Features: Triangle geometry (sides, angles, area)
- Target: Number of sources used
- Models: RandomForest, GradientBoosting, Ridge

**RQ2:** Can embedding geometry predict perplexity?
- Features: Same geometry
- Target: Abstract perplexity score
- Models: Same

## Data Integration

✅ Uses real FAISS index: `../data/index/corpus.faiss`
✅ Loads metadata on-demand: `../data/raw/*/metadata/*.json`
✅ Three data sources: Chronicling America, Internet Archive, NARA
✅ Lazy loading for efficiency

## No More Pipeline 2

❌ No synthetic personas
❌ No causal inference
❌ No SAE (Sparse Autoencoder)
❌ No double debiasing
❌ No duplicate implementations

## Result

**Clean, single-purpose prediction pipeline.**

Run your experiments:
```bash
python experiments/experiment_runner.py \
  --strategy filtered \
  --n-samples 10 \
  --max-experiments 10
```
