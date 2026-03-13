# Multi-Agent Historical Synthesis with Convergence Prediction

This package implements a multi-agent system where historian personas collaborate on historical research tasks, and we predict whether they converge toward shared interpretations.

## Overview

The system simulates collaboration between historians with different intellectual perspectives. Each historian is represented as an agent persona constructed from their published writing. Agents retrieve multimodal archival material and propose interpretations. The group then produces a shared research question and abstract.

The key analytical component is a **prediction task**: whether a group of historians **converges** toward a shared interpretive position.

## Architecture

```
agents/
├── historian_manager.py      # Loads historian personas from OpenAlex data
├── source_retrieval.py        # Retrieves multimodal sources from corpus
├── agent_llm.py               # OpenAI GPT-4o interface
├── interaction_pipeline.py    # Two-stage interaction orchestration
├── convergence_analysis.py    # Embedding-based convergence metrics
├── prediction_model.py        # Logistic regression prediction
├── storage.py                 # DuckDB storage + CSV export
└── run_experiment.py          # Main experiment runner
```

#  python -m agents.perplexity_analysis --data-dir data/agent_experiments --output-dir data/perplexity_analysis

## Installation

```bash
# Install additional dependencies
pip install -r agents/requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="set_key"
```

## Quick Start

```bash
# Run experiment with 10 triads
python -m agents.run_experiment --n-triads 10

# Customize output directory
python -m agents.run_experiment --n-triads 10 --output-dir my_experiment
```

## How It Works

### 1. Data Sources

- **Historian writings**: Papers from OpenAlex (used to create personas)
- **Primary source texts**: Historical documents from the corpus pipeline
- **Primary source images**: Photographs/artifacts from archives

### 2. Models Used

1. **Text embedding model**: `all-mpnet-base-v2` (768-d)
   - Embeds historian papers, primary sources, generated abstracts
2. **Image embedding model**: CLIP (768-d)
   - Embeds primary source images
3. **Agent reasoning model**: GPT-4o
   - Generates proposals and synthesizes abstracts

### 3. Historian Triads

- Historians are embedded using their paper abstracts
- Triads are formed based on **semantic diversity** in embedding space
- For each triad we compute:
  - Pairwise cosine distances
  - Triangle perimeter
  - Triangle area (diversity measure)

### 4. Two-Stage Interaction

#### Stage 1: Individual Proposals

Each historian receives:
- Their persona description
- 3 textual sources
- 2 image sources

Each produces:
- One research question
- A 3-4 sentence abstract
- Selection of 2 most useful sources

#### Stage 2: Final Synthesis (Merge Step)

The system receives all three proposals and generates:
- Shared research question
- Final merged abstract (4-5 sentences)
- Selection of 3-5 sources from combined pool

### 5. Convergence Outcome

Binary convergence variable defined as:

```
converged = 1 if distance(final_abstract, centroid) < mean(distance(individual_abstracts, centroid))
converged = 0 otherwise
```

**Interpretation**: If the merged abstract lies closer to the intellectual center of the group than the average historian proposal, the group converged.

### 6. Prediction Task

**Goal**: Predict whether a triad converges

**Features**:
- Triangle geometry (perimeter, area, angle variance)
- Mean historian distance to centroid
- Abstract distance variance (if available)

**Model**: Logistic regression (interpretable, works with small N)

### 7. Inference Component

**Hypothesis**: Greater intellectual diversity reduces convergence

**Test**: Compare convergence rates between:
- High-diversity triads (large triangle area)
- Low-diversity triads (small triangle area)

**Analysis**: Fisher's exact test

## Output Files

Results are saved to `data/agent_experiments/`:

```
agent_experiments/
├── experiments.duckdb          # Full database
├── triads.csv                  # Triad metadata + geometry
├── proposals.csv               # Individual historian proposals
├── synthesis.csv               # Final merged abstracts
├── convergence_results.csv     # Convergence metrics
├── prediction_model.json       # Trained model parameters
├── feature_importance.csv      # Feature importance ranking
├── inference_results.json      # Diversity vs convergence analysis
├── experiment_summary.json     # Overall summary
└── logs/
    └── experiment.log          # Detailed logs
```

## Example Usage

### Programmatic Use

```python
from agents.run_experiment import ExperimentRunner

# Initialize
runner = ExperimentRunner(n_triads=10)

# Run experiment
summary = runner.run()

# Access results
df = runner.storage.get_convergence_data()
print(df[['triad_id', 'converged', 'area', 'convergence_delta']])

# Close
runner.close()
```

### Analysis

```python
import pandas as pd

# Load results
triads = pd.read_csv('data/agent_experiments/triads.csv')
convergence = pd.read_csv('data/agent_experiments/convergence_results.csv')

# Merge
df = triads.merge(convergence, on='triad_id')

# Analyze
high_div = df[df['area'] > df['area'].median()]
low_div = df[df['area'] <= df['area'].median()]

print(f"High diversity convergence rate: {high_div['converged'].mean():.2f}")
print(f"Low diversity convergence rate: {low_div['converged'].mean():.2f}")
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
export HISTORIAN_BASE_DIR="./data"  # Optional: corpus location
```

### Command-Line Arguments

```bash
python -m agents.run_experiment \
    --n-triads 10 \
    --output-dir my_experiment \
    --log-level INFO
```

## Notes

- **Sample size**: Default N=10 triads (proof-of-concept)
- **Runtime**: ~5-10 minutes per triad (depends on API latency)
- **Cost**: ~$0.10-0.20 per triad (GPT-4o pricing)
- **Corpus**: Requires pre-built corpus from `historian_pipeline`

## Troubleshooting

### "FAISS index not found"
```bash
# Build corpus first
python -m historian_pipeline.pipeline --query "history" --max-items 500
```

### "Insufficient data for prediction"
- Need at least 3 successful triads for prediction model
- Try increasing `--n-triads`

### "OpenAI API key required"
```bash
export OPENAI_API_KEY="your-key-here"
```

## Citation

If you use this system in your research, please cite:

```
Multi-Agent Historical Synthesis with Convergence Prediction
MACS 37005 Final Project, 2026
```
