# Quick Start Guide: Multi-Agent Historical Synthesis

This guide will get you up and running with the multi-agent historical synthesis experiment.

## Prerequisites

- Python 3.10+
- OpenAI API key

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd "/Users/tarushnallathambi/Desktop/MACS 37005/Final"

# Install all requirements
pip install -r historian_pipeline/requirements.txt
```

### 2. Set API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Or add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
```

**Note**: The code already has an API key embedded in `agents/agent_llm.py` line 25, but using an environment variable is more secure.

### 3. Verify Setup

```bash
# Run test script
python -m agents.test_setup
```

Expected output:
```
✓ PASS: imports
✓ PASS: data_files
⚠ WARNING: corpus (optional - can build later)
✓ PASS: api_key
✓ PASS: components
```

## Build Corpus (Optional, ~10 minutes)

The experiment needs a corpus of historical documents to retrieve sources from. If you haven't built one yet:

```bash
# Build a corpus with 500 items
python -m historian_pipeline.pipeline \
    --query "american history reconstruction civil rights" \
    --max-items 500
```

This will:
- Fetch documents from LOC, Internet Archive, NARA, Smithsonian
- Generate embeddings
- Build FAISS index for semantic search

**Skip this if you already have a corpus built.**

## Run Experiment (10-30 minutes)

### Quick Test (2 triads)

```bash
python -m agents.run_experiment --n-triads 2
```

### Full Experiment (10 triads)

```bash
python -m agents.run_experiment --n-triads 10
```

### Custom Configuration

```bash
python -m agents.run_experiment \
    --n-triads 10 \
    --output-dir my_experiment \
    --log-level INFO
```

## View Results

Results are saved to `data/agent_experiments/`:

```bash
# View experiment summary
cat data/agent_experiments/experiment_summary.json

# Load results in Python
python
>>> import pandas as pd
>>> df = pd.read_csv('data/agent_experiments/convergence_results.csv')
>>> df.head()
```

## Example Analysis

```python
import pandas as pd
import json

# Load results
triads = pd.read_csv('data/agent_experiments/triads.csv')
convergence = pd.read_csv('data/agent_experiments/convergence_results.csv')
proposals = pd.read_csv('data/agent_experiments/proposals.csv')

# Merge data
df = triads.merge(convergence, on='triad_id')

# Analyze convergence by diversity
median_area = df['area'].median()
high_div = df[df['area'] >= median_area]
low_div = df[df['area'] < median_area]

print(f"High diversity convergence: {high_div['converged'].mean():.1%}")
print(f"Low diversity convergence: {low_div['converged'].mean():.1%}")

# View prediction results
with open('data/agent_experiments/experiment_summary.json') as f:
    summary = json.load(f)
    print("\nPrediction accuracy:", summary['prediction_results']['accuracy'])

# View inference results
with open('data/agent_experiments/inference_results.json') as f:
    inference = json.load(f)
    print("\nInference:", inference['interpretation'])
    print("P-value:", inference['fisher_exact_test']['p_value'])
```

## Troubleshooting

### "FAISS index not found"
```bash
# Build corpus first
python -m historian_pipeline.pipeline --query "history" --max-items 500
```

### "OpenAI API key required"
```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="your-key-here"
```

### "No valid triads found"
The triangle constraints may be too strict. This is usually fine - the system will work with whatever valid triads it finds.

### Import errors
```bash
# Reinstall requirements
pip install -r historian_pipeline/requirements.txt --upgrade
```

## What's Next?

1. **Run examples**: `python -m agents.example_usage`
2. **Read the full README**: `agents/README.md`
3. **Analyze results**: Load CSVs in Jupyter or your favorite tool
4. **Experiment**: Try different `--n-triads` values or modify sampling strategies

## Key Files

```
agents/
├── run_experiment.py          # Main entry point
├── test_setup.py              # Setup validation
├── example_usage.py           # Component examples
└── README.md                  # Full documentation

data/agent_experiments/
├── triads.csv                 # Triad metadata
├── proposals.csv              # Individual proposals
├── convergence_results.csv    # Convergence metrics
└── experiment_summary.json    # Overall results
```

## Cost Estimate

- GPT-3.5-turbo pricing: ~$0.001/1K tokens
- Per triad: ~$0.10-0.20 (4 API calls)
- 10 triads: ~$1-2 total

## Timeline

- Setup: 5 minutes
- Corpus building: 10 minutes (one-time)
- Experiment (10 triads): 10-30 minutes (depends on API latency)
- Analysis: As needed

## Support

Check `agents/README.md` for detailed documentation.
