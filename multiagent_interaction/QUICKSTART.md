# Quick Start Guide

Get up and running with the Multi-Agent Historian system in 5 minutes.

## Prerequisites

- Python 3.8+
- API key from Anthropic (Claude) or OpenAI

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Edit .env and add your API key
```

## Run Your First Experiment

```bash
# Run 5 small experiments (takes ~10-15 minutes)
python run_experiment.py --mode full --n-samples 5
```

This will:
1. Create 1,225 historian personas (7×5×5×7 factorial)
2. Build a library of 200 primary sources
3. Run 5 multi-agent experiments (3 agents each)
4. Perform causal analysis

## Check Results

```bash
# View system status
python utils.py status

# See experiment summary
python utils.py summary

# Export a chat history
python utils.py export exp_00000
```

Results are in `outputs/`:
- `results.csv` - Experiment metrics
- `exp_*.json` - Individual experiment details
- `analysis_report.json` - Causal analysis results

## Understanding the Output

Each experiment produces:

**Experiment File** (`exp_00000.json`):
```json
{
  "experiment_id": "exp_00000",
  "group_composition": [...],    // The 3 historian personas
  "chat_history": [...],         // Full dialogue with actions
  "sources_accessed": [...],     // Primary sources used
  "final_question": "...",       // Research question developed
  "final_abstract": "...",       // Abstract developed
  "turn_count": 42,
  "consensus_reached": true
}
```

**Analysis Report** (`analysis_report.json`):
- RQ1: Which persona characteristics affect source selection?
- RQ2: Which configurations produce the best theses?

## Common Commands

```bash
# Setup only (no experiments)
python run_experiment.py --mode setup-only

# Run more experiments
python run_experiment.py --mode run-only --n-samples 50

# Analyze existing results
python run_experiment.py --mode analyze-only

# Check status
python utils.py status

# Compare two experiments
python utils.py compare exp_00000 exp_00001
```

## Example Usage

Try the example script to understand the components:

```bash
python example_usage.py
```

This demonstrates:
- Creating personas
- Building source libraries
- Searching sources
- Adding custom documents

## Customization

**Change dialogue length:**

Edit `config/config.yaml`:
```yaml
dialogue:
  max_turns: 30  # Default is 50
```

**Use different LLM:**

Edit `config/config.yaml`:
```yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4"      # or "claude-sonnet-4-6"
```

**Add custom personas:**

Edit `config/config.yaml`:
```yaml
personas:
  fields:
    - "digital_history"  # Add your field
```

## Troubleshooting

**"No module named 'faiss'"**
```bash
pip install faiss-cpu
```

**"API key not found"**
- Check your `.env` file exists
- Verify the key is correct
- Make sure it's named `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

**"Index is empty"**
```bash
python run_experiment.py --mode setup-only
```

**Experiments taking too long**
- Reduce `--n-samples` (try 5-10 first)
- Decrease `max_turns` in config.yaml
- Use faster model (e.g., claude-haiku)

## Next Steps

1. **Scale up**: Run 50-100 experiments for robust statistics
2. **Analyze patterns**: Explore `analysis_report.json`
3. **Custom sources**: Add your own primary documents
4. **Fine-tune personas**: Adjust characteristics in config
5. **Export insights**: Use `utils.py` to extract specific results

## Getting Help

- Check `README.md` for full documentation
- Review `example_usage.py` for code examples
- See configuration options in `config/config.yaml`

## What's Happening Under the Hood?

1. **Factorial Design**: Creates all combinations of historian types
2. **Multi-Agent Dialogue**: Agents talk, search sources, and debate
3. **Induction + Deduction**: Thesis emerges from sources + dialogue
4. **Causal Analysis**: SAE + double debiasing finds what matters

**Research Questions Answered**:
- RQ1: How do historians select sources differently?
- RQ2: Which teams produce the best research?

## Performance Expectations

- **Setup**: ~2-3 minutes
- **Per experiment**: ~2-5 minutes (varies with turns and LLM speed)
- **5 experiments**: ~15-20 minutes total
- **50 experiments**: ~2-3 hours

## Resource Requirements

- **Memory**: ~2-4 GB RAM
- **Disk**: ~500 MB for outputs (grows with experiments)
- **API costs**: ~$0.10-0.50 per experiment (depends on LLM and turns)

---

**Ready to start?**

```bash
python run_experiment.py --mode full --n-samples 5
```

Then explore the results:

```bash
python utils.py summary
```
