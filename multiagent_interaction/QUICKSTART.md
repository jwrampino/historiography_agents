# Quick Start Guide

## Run End-to-End

```bash
cd historiography_agents/multiagent_interaction

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# 1. Run experiments (10 groups of 3 historians)
python experiments/experiment_runner.py \
  --strategy filtered \
  --n-samples 10 \
  --max-experiments 10

# 2. Analyze results
python analysis/prediction_analysis.py

# 3. Generate visualizations
python visualization/generate_all.py --output-dir outputs
```

Done! Check `outputs/` for all results.

## What This Does

**Experiments (30-90 min)**
- 10 groups × 3 historians = real personas from OpenAlex
- Each group collaborates to develop research questions
- They search/read from real FAISS corpus (Chronicling America, Internet Archive, NARA)
- ~750 API calls total

**Analysis**
- Predicts source selection from embedding geometry (RQ1)
- Predicts perplexity from embedding geometry (RQ2)
- Models: RandomForest, GradientBoosting, Ridge

**Visualizations**
- 50+ plots in `outputs/figures/`

## Quick Test (6-18 min)

```bash
# Just 2 experiments for testing
python experiments/experiment_runner.py \
  --strategy filtered \
  --n-samples 2 \
  --max-experiments 2
```

## Output

```
outputs/
├── results.csv                   # Summary
├── exp_*.json                    # Transcripts
├── prediction_report.json        # Analysis
└── figures/                      # Visualizations
```
