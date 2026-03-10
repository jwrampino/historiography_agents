# Multi-Agent Historian Experiment System - Project Summary

## What You Have

A complete, production-ready system for running factorial design experiments with multi-agent collaborative historical research, including causal analysis of how historian personas influence outcomes.

## Project Structure

```
multiagent_interaction/
│
├── 📋 Core Scripts
│   ├── run_experiment.py       # Main entry point for running experiments
│   ├── example_usage.py        # Tutorial examples
│   ├── utils.py                # Utility functions
│   └── visualize.py            # Visualization tools
│
├── 📦 Components
│   ├── personas/               # Historian persona management
│   │   ├── __init__.py
│   │   ├── persona_manager.py  # Create/manage personas
│   │   └── persona_storage.json (created on first run)
│   │
│   ├── sources/                # FAISS-based source library
│   │   ├── __init__.py
│   │   ├── source_library.py   # Vector DB for primary sources
│   │   └── faiss_index/        (created on first run)
│   │
│   ├── agents/                 # Multi-agent dialogue system
│   │   ├── __init__.py
│   │   └── multi_agent_system.py  # LangGraph agents
│   │
│   ├── experiments/            # Experiment orchestration
│   │   ├── __init__.py
│   │   └── experiment_runner.py   # Factorial design runner
│   │
│   └── analysis/               # Causal inference
│       ├── __init__.py
│       └── causal_analysis.py  # SAE + double debiasing
│
├── 📁 Configuration & Data
│   ├── config/
│   │   └── config.yaml         # All system parameters
│   ├── outputs/                # Experiment results (created)
│   └── logs/                   # System logs (created)
│
├── 📖 Documentation
│   ├── README.md               # Full documentation
│   ├── QUICKSTART.md           # 5-minute getting started
│   └── PROJECT_SUMMARY.md      # This file
│
└── 🔧 Setup
    ├── requirements.txt        # Python dependencies
    ├── .env.example            # API key template
    └── .gitignore              # Git ignore rules
```

## What Each Component Does

### 1. Persona Manager (`personas/persona_manager.py`)

**Purpose**: Manages historian personas across 4 dimensions

**Features**:
- Creates factorial design of personas (field × method × era × orientation)
- Generates ~1,200+ unique historian types
- Supports stratified sampling for manageable experiments
- Loads/saves persona configurations

**Key Classes**:
- `HistorianPersona`: Data class for individual personas
- `PersonaManager`: Creates and manages persona groups

### 2. Source Library (`sources/source_library.py`)

**Purpose**: FAISS-based vector database for primary sources

**Features**:
- Semantic search using sentence transformers
- Multimodal support (text, images, audio transcripts)
- Access logging for causal analysis
- Efficient similarity search at scale

**Key Classes**:
- `PrimarySource`: Data class for documents
- `SourceLibrary`: FAISS index manager

### 3. Multi-Agent System (`agents/multi_agent_system.py`)

**Purpose**: LangGraph-based collaborative dialogue

**Features**:
- 6 agent actions: speak, search, read, propose, critique, conclude
- Stateful dialogue with full history tracking
- Inductive + deductive reasoning
- Iterative refinement toward consensus

**Key Classes**:
- `HistorianAgent`: Individual agent with persona
- `MultiAgentDialogueSystem`: Orchestrates dialogue
- `DialogueState`: Tracks conversation state
- `AgentAction`: Records agent actions

### 4. Experiment Runner (`experiments/experiment_runner.py`)

**Purpose**: Orchestrates factorial design experiments

**Features**:
- Full factorial or stratified sampling
- Progress tracking and error handling
- Result serialization (JSON + CSV)
- Resume capability for interrupted runs

**Key Classes**:
- `ExperimentRunner`: Main orchestrator
- `ExperimentResult`: Stores outcomes

### 5. Causal Analysis (`analysis/causal_analysis.py`)

**Purpose**: Statistical analysis with causal inference

**Features**:
- **SAE**: Learns latent representations of source patterns
- **Double Debiasing**: Reduces treatment effect bias
- **Feature Importance**: Identifies key persona characteristics
- **Quality Metrics**: Novelty, perplexity, quality scores

**Key Classes**:
- `ExperimentAnalyzer`: Main analysis class
- `SparseAutoencoder`: PyTorch SAE implementation
- `DoubleDebiasingEstimator`: Causal inference
- `CausalEstimate`: Effect size results

### 6. Utilities (`utils.py`)

**Purpose**: Helper functions for common tasks

**Functions**:
- `load_experiment_result()`: Load specific experiment
- `export_chat_history()`: Export dialogue to text
- `summarize_experiments()`: Create summary table
- `compare_experiments()`: Side-by-side comparison
- `find_experiments_by_persona()`: Filter by characteristics
- `analyze_source_usage()`: Source usage statistics
- `check_system_status()`: Verify setup

### 7. Visualization (`visualize.py`)

**Purpose**: Generate publication-quality figures

**Functions**:
- `plot_experiment_overview()`: Basic statistics
- `plot_persona_effects()`: Persona characteristics vs outcomes
- `plot_source_selection_patterns()`: Source usage heatmaps
- `plot_causal_analysis()`: Causal effects visualization
- `create_all_visualizations()`: Generate all plots

## Research Questions Addressed

### RQ1: Differential Source Selection

**Question**: How do specific groups of historians (defined by field, method, era, and theoretical orientation) differentially select primary sources from a common multimodal corpus?

**Method**:
1. Extract source selection patterns (binary matrix: experiment × sources)
2. Train SAE to learn latent dimensions of selection behavior
3. Use double debiasing to estimate causal effects of persona characteristics
4. Identify which dimensions (field/method/era/orientation) drive differences

**Output**: `analysis_report.json` → `rq1_source_selection`

### RQ2: Optimal Configurations

**Question**: Which configurations of historian personas and source-selection patterns produce the most novel, perplexing, and high-quality historical theses?

**Method**:
1. Compute outcome metrics:
   - Novelty: Semantic distance from other theses
   - Perplexity: Language model surprise
   - Quality: Composite score
2. Random forest feature importance analysis
3. Identify top-performing persona combinations

**Output**: `analysis_report.json` → `rq2_optimal_configurations`

## How to Use

### Basic Workflow

```bash
# 1. Setup (one time)
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key

# 2. Run experiments
python run_experiment.py --mode full --n-samples 20

# 3. View results
python utils.py summary
python visualize.py all

# 4. Explore specific experiments
python utils.py export exp_00000
```

### Advanced Usage

```bash
# Larger experiment run
python run_experiment.py --mode run-only --strategy stratified --n-samples 100

# Custom configuration
python run_experiment.py --config my_config.yaml

# Resume interrupted run
python run_experiment.py --mode run-only --start-from 50

# Analyze only
python run_experiment.py --mode analyze-only
```

## Configuration Options

Edit `config/config.yaml` to customize:

**Personas**:
- Add/remove fields, methods, eras, orientations
- Changes the factorial design space

**Dialogue**:
- `max_turns`: How long conversations can go
- `consensus_threshold`: When to stop
- `actions`: Available agent actions

**LLM**:
- `provider`: anthropic or openai
- `model`: Which model to use
- `temperature`: Creativity level

**Analysis**:
- `sae_latent_dim`: SAE complexity
- `n_bootstrap_samples`: Statistical robustness
- `outcome_metrics`: What to measure

## Output Files

After running experiments, you'll have:

### Individual Experiments (`outputs/exp_*.json`)

```json
{
  "experiment_id": "exp_00042",
  "group_composition": [...],      // 3 historian personas
  "chat_history": [...],           // Full dialogue
  "sources_accessed": [...],       // Primary sources used
  "final_question": "...",         // Research question
  "final_abstract": "...",         // Abstract
  "turn_count": 42,
  "consensus_reached": true
}
```

### Summary (`outputs/results.csv`)

| experiment_id | turns | consensus | sources_used | agent_0_field | agent_0_method | ... |
|--------------|-------|-----------|--------------|---------------|----------------|-----|
| exp_00000 | 35 | true | 12 | social_history | quantitative | ... |
| exp_00001 | 42 | true | 8 | cultural_history | qualitative | ... |

### Analysis Report (`outputs/analysis_report.json`)

```json
{
  "rq1_source_selection": {
    "n_tests": 384,
    "n_significant": 47,
    "significant_effects": [...]
  },
  "rq2_optimal_configurations": {
    "novelty_combined": {
      "feature_importance": [...],
      "top_configurations": [...]
    }
  }
}
```

### Figures (`outputs/figures/`)

- `experiment_overview.png`: Basic statistics
- `persona_effects.png`: Persona characteristics
- `source_patterns.png`: Source usage
- `causal_analysis.png`: Causal effects

## Key Design Decisions

### 1. Induction + Deduction Mix

**Problem**: How to avoid pre-selection bias (picking sources for a thesis)?

**Solution**: Agents can search sources at any time AND propose theses at any time. The thesis emerges from iterative dialogue about sources, not predetermined.

### 2. Factorial Design

**Problem**: How to systematically explore persona combinations?

**Solution**: Full factorial across 4 dimensions creates complete design space. Stratified sampling makes it tractable.

### 3. SAE for Source Patterns

**Problem**: Source selection is high-dimensional (100s-1000s of sources).

**Solution**: SAE learns low-dimensional latent representation, making causal analysis tractable.

### 4. Double Debiasing

**Problem**: Confounding between persona characteristics.

**Solution**: Double machine learning reduces bias in treatment effect estimates.

### 5. LangGraph State Management

**Problem**: Need complex multi-agent interactions with memory.

**Solution**: LangGraph provides stateful graph execution with full dialogue history.

## Extensibility

### Add Custom Personas

Edit `config/config.yaml`:
```yaml
personas:
  fields:
    - "digital_history"
    - "public_history"
```

### Add Custom Sources

```python
from sources import SourceLibrary, PrimarySource

library = SourceLibrary()
library.load_index()

my_source = PrimarySource(
    source_id="custom_001",
    title="My Document",
    content="...",
    source_type="text",
    metadata={"year": 1920}
)

library.add_source(my_source)
library.save_index()
```

### Modify Agent Behavior

Edit `agents/multi_agent_system.py`:
```python
def get_system_prompt(self) -> str:
    return f"""You are a {self.persona.field} historian...

    NEW INSTRUCTIONS:
    - Always cite page numbers
    - Consider counterfactuals
    """
```

### Add New Outcome Metrics

Edit `analysis/causal_analysis.py`:
```python
def compute_custom_metric(self, result):
    # Your metric calculation
    return score
```

## Performance Characteristics

**Setup Time**: ~2-3 minutes
- Load embedding model
- Create personas
- Build FAISS index

**Per Experiment**: ~2-5 minutes
- Depends on: turns, LLM speed, source searches
- Typical: 30-40 turns × 3-5 actions/turn

**Scaling**:
- 5 experiments: ~15 minutes
- 50 experiments: ~2-3 hours
- 500 experiments: ~1-2 days

**Resource Usage**:
- RAM: 2-4 GB
- Disk: ~500 MB (grows with experiments)
- API costs: ~$0.10-0.50 per experiment

## Troubleshooting

**Issue**: "Module not found"
```bash
pip install -r requirements.txt
```

**Issue**: "Index is empty"
```bash
python run_experiment.py --mode setup-only
```

**Issue**: "API key not found"
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

**Issue**: Experiments too slow
- Reduce `--n-samples`
- Decrease `max_turns` in config
- Use faster model (claude-haiku)

**Issue**: Out of memory
- Reduce `n_sources`
- Use `faiss-gpu` for larger corpora
- Process experiments in batches

## Next Steps

1. **Test Run**: `python run_experiment.py --mode full --n-samples 5`
2. **Explore**: `python example_usage.py`
3. **Visualize**: `python visualize.py all`
4. **Scale**: Increase to 50-100 experiments
5. **Analyze**: Deep dive into `analysis_report.json`
6. **Customize**: Add your personas/sources
7. **Publish**: Use figures for papers/presentations

## Citation

```bibtex
@software{multiagent_historian_2026,
  title={Multi-Agent Historian Research Simulation System},
  year={2026},
  description={Factorial design experiments with causal inference for
               analyzing collaborative historical research patterns}
}
```

## Support

- **Documentation**: See `README.md`
- **Examples**: Run `python example_usage.py`
- **Utilities**: `python utils.py status`
- **Configuration**: Edit `config/config.yaml`

---

**System Status**: ✓ Complete and ready to use

**Last Updated**: 2026-03-09
