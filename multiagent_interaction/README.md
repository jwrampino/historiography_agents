# Multi-Agent Historian System - Prediction-Based Analysis

A system for simulating collaborative historical research through multi-agent dialogue, with **prediction models** to analyze how historian configurations affect outcomes.

## Overview

Groups of AI historian agents with different theoretical orientations, methodological approaches, and domain expertise collaboratively develop research questions by:

1. **Exploring primary sources** through a FAISS-based document library (real data from historian_pipeline)
2. **Engaging in dialogue** to share insights and critique proposals
3. **Iteratively refining** research questions and abstracts until consensus

The system then uses **supervised prediction models** (RandomForest, GradientBoosting, Ridge) to answer:

- **RQ1**: Can embedding geometry predict source selection patterns?
- **RQ2**: Can embedding geometry predict abstract quality (perplexity)?

## Quick Start

```bash
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

## Pipeline Structure

```
1. personas/historian_manager.py
   ↓ (Load real historians from OpenAlex)

2. sources/source_library.py
   ↓ (Connect to ../data/index/corpus.faiss)

3. experiments/experiment_runner.py
   ↓ (Run multi-agent dialogues)

4. analysis/prediction_analysis.py
   ↓ (Predict outcomes using regression)

5. visualization/generate_all.py
   ↓ (Create 50+ plots)
```

## Research Questions

### RQ1: Source Selection Prediction
Can triangle geometry (embedding distances between historians) predict how many sources they'll use?

- **Features**: sides, angles, area, perimeter of historian embedding triangle
- **Target**: n_sources_used per experiment
- **Models**: RandomForest, GradientBoosting, Ridge

### RQ2: Perplexity Prediction
Can the same geometry predict abstract quality/novelty (measured via perplexity)?

- **Features**: same geometry features
- **Target**: abstract perplexity score
- **Models**: same regression models

## File Structure

```
multiagent_interaction/
├── personas/
│   ├── historian_manager.py        # Real historian personas (OpenAlex)
│   └── historian_personas.json     # 25 real historians with embeddings
├── sources/
│   └── source_library.py           # FAISS + lazy metadata loading
├── agents/
│   └── multi_agent_system.py       # LangGraph dialogue system
├── experiments/
│   └── experiment_runner.py        # Main experiment orchestrator
├── analysis/
│   └── prediction_analysis.py      # Prediction models (Pipeline 1)
├── visualization/
│   ├── generate_all.py             # Master visualization generator
│   ├── geometry_viz.py             # Triangle geometry plots
│   ├── prediction_viz.py           # Prediction results
│   └── experiment_viz.py           # Experiment dynamics
├── test_integration.py             # Tests
├── test_prediction_analysis.py
├── test_visualizations.py
└── config/
    └── config.yaml                 # Configuration
```

## Configuration

`config/config.yaml`:
```yaml
sources:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  index_path: "../data/index"       # Points to historian_pipeline data
  metadata_path: "../data/raw"

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  temperature: 0.7

experiment:
  n_agents_per_group: 3
  max_dialogue_turns: 50
```

## Data Integration

Uses real corpus from `historiography_agents/data/`:
- FAISS index: `../data/index/corpus.faiss`
- Metadata: `../data/raw/*/metadata/*.json`
- Sources: Chronicling America, Internet Archive, NARA

## Dependencies

```bash
pip install anthropic langchain-anthropic langgraph faiss-cpu \
  sentence-transformers pandas numpy scikit-learn matplotlib \
  seaborn tqdm pyyaml
```

## Output

After running experiments:
```
outputs/
├── results.csv                  # Summary of all experiments
├── exp_00000.json              # Detailed transcript for each
├── exp_00001.json
├── ...
├── prediction_report.json      # Analysis results
├── correlations.csv            # Feature correlations
└── figures/                    # 50+ visualizations
    ├── geometry_*.png
    ├── prediction_*.png
    └── experiment_*.png
```

## Testing

```bash
# Test with mock data
python test_integration.py
python test_prediction_analysis.py
python test_visualizations.py
```

## Time Estimates

| Experiments | Duration | API Calls |
|-------------|----------|-----------|
| 10          | 30-90 min | ~750 |
| 50          | 2.5-7.5 hrs | ~3,750 |
| 100         | 5-15 hrs | ~7,500 |

## Clean Architecture

✅ Single pipeline (prediction-based)
✅ Real data from historian_pipeline
✅ No causal inference complexity
✅ Clear research questions
✅ Reproducible results
