# Multi-Agent Historian Research Simulation

A comprehensive system for simulating collaborative historical research through multi-agent dialogue, with causal analysis of how historian personas influence source selection and thesis quality.

## Overview

This system implements a factorial design experiment where groups of AI historian agents with different theoretical orientations, methodological approaches, and domain expertise collaboratively develop research questions by:

1. **Exploring primary sources** through a FAISS-based multimodal document library
2. **Engaging in dialogue** to share insights and critique proposals
3. **Mixing induction and deduction** to let theses emerge from source exploration
4. **Iteratively refining** research questions and abstracts

The system then uses **Sparse Autoencoders (SAE)** and **Double Debiasing** techniques to perform causal inference and answer:

- **RQ1**: How do specific groups of historians (defined by field, method, era, and theoretical orientation) differentially select primary sources from a common multimodal corpus?

- **RQ2**: Which configurations of historian personas and source-selection patterns produce the most novel, perplexing, and high-quality historical theses?

## Architecture

```
multiagent_interaction/
├── personas/              # Historian persona management
│   ├── persona_manager.py
│   └── persona_storage.json
│
├── sources/               # FAISS-based source library
│   ├── source_library.py
│   └── faiss_index/
│
├── agents/                # Multi-agent dialogue system (LangGraph)
│   └── multi_agent_system.py
│
├── experiments/           # Factorial design experiment runner
│   └── experiment_runner.py
│
├── analysis/              # Causal inference (SAE + double debiasing)
│   └── causal_analysis.py
│
├── config/                # Configuration files
│   └── config.yaml
│
├── outputs/               # Experiment results
│   ├── results.csv
│   ├── exp_*.json
│   └── analysis_report.json
│
├── requirements.txt
└── run_experiment.py      # Main entry point
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up API keys** (create `.env` file):
```bash
# For Anthropic Claude
ANTHROPIC_API_KEY=your_key_here

# OR for OpenAI
OPENAI_API_KEY=your_key_here
```

3. **Configure experiment** (edit `config/config.yaml`):
   - Adjust persona dimensions
   - Set dialogue parameters
   - Configure LLM settings
   - Set analysis parameters

## Quick Start

### Full Pipeline (Setup + Run + Analyze)

```bash
python run_experiment.py --mode full --n-samples 20
```

This will:
1. Create historian personas across 4 dimensions
2. Build a source library with 200 primary sources
3. Run 20 multi-agent experiments
4. Perform causal analysis

### Step-by-Step Execution

**Step 1: Setup**
```bash
python run_experiment.py --mode setup-only --n-sources 500
```

**Step 2: Run Experiments**
```bash
python run_experiment.py --mode run-only --strategy stratified --n-samples 50
```

**Step 3: Analyze Results**
```bash
python run_experiment.py --mode analyze-only
```

## Components

### 1. Persona Manager

Manages historian personas defined by:
- **Field**: social_history, political_history, cultural_history, etc.
- **Method**: quantitative, qualitative, comparative, microhistory, etc.
- **Era**: ancient, medieval, early_modern, modern, contemporary
- **Theoretical Orientation**: marxist, poststructuralist, feminist, postcolonial, etc.

```python
from personas import PersonaManager

manager = PersonaManager()
personas = manager.create_persona_grid()  # Full factorial
groups = manager.generate_stratified_groups(group_size=3, n_samples=100)
```

### 2. Source Library

FAISS-based vector database for primary sources with:
- Semantic search using sentence transformers
- Access logging for causal analysis
- Support for multimodal documents (text, image captions, transcripts)

```python
from sources import SourceLibrary, PrimarySource

library = SourceLibrary()
library.initialize_embedding_model()

# Search for relevant sources
results = library.search("labor movements", k=5, agent_id="historian_001")
```

### 3. Multi-Agent Dialogue System

LangGraph-based stateful dialogue where agents can:

**Actions**:
- `SPEAK`: Share thoughts and arguments
- `SEARCH`: Query the source library
- `READ`: Read a source in detail
- `PROPOSE`: Propose research question/thesis
- `CRITIQUE`: Critique another's proposal
- `CONCLUDE`: Signal consensus

```python
from agents import MultiAgentDialogueSystem

system = MultiAgentDialogueSystem()
final_state = system.run_experiment(personas, experiment_id="exp_001")

# Access results
print(final_state.final_question)
print(final_state.sources_accessed)
print(final_state.messages)  # Full chat history
```

### 4. Experiment Runner

Orchestrates factorial design experiments:

```python
from experiments import ExperimentRunner

runner = ExperimentRunner()
runner.setup_personas(strategy="stratified", n_samples=50)
runner.setup_source_library()
runner.run_all_experiments()

# Export results
df = runner.get_results_dataframe()
df.to_csv("results.csv")
```

### 5. Causal Analysis

Implements:
- **Sparse Autoencoder (SAE)**: Learn latent representations of source selection patterns
- **Double Debiasing**: Reduce bias in treatment effect estimation
- **Feature Importance**: Identify which persona characteristics matter most

```python
from analysis import ExperimentAnalyzer

analyzer = ExperimentAnalyzer()
analyzer.load_results("outputs/results.csv")
analyzer.load_detailed_results("outputs/")

# Analyze RQ1: Source selection patterns
rq1_results = analyzer.analyze_rq1_source_selection()

# Analyze RQ2: Optimal configurations
rq2_results = analyzer.analyze_rq2_optimal_configurations()

# Generate full report
report = analyzer.generate_report("outputs/analysis_report.json")
```

## Output Structure

Each experiment produces:

```json
{
  "experiment_id": "exp_00042",
  "group_composition": [
    {
      "persona_id": "historian_0123",
      "field": "social_history",
      "method": "quantitative",
      "era": "modern",
      "theoretical_orientation": "marxist"
    },
    // ... 2 more agents
  ],
  "chat_history": [
    {
      "agent_id": "historian_0123",
      "action_type": "search",
      "content": "labor strikes 1890s",
      "timestamp": "2026-03-09T10:30:45"
    },
    // ... all dialogue turns
  ],
  "sources_accessed": [
    {
      "source_id": "source_0045",
      "title": "Miners' Strike of 1894",
      "content": "..."
    }
  ],
  "final_question": "How did...",
  "final_abstract": "This research examines...",
  "turn_count": 42,
  "consensus_reached": true
}
```

## Analysis Report

The causal analysis produces:

```json
{
  "rq1_source_selection": {
    "n_tests": 384,
    "n_significant": 47,
    "significant_effects": [
      {
        "agent": 0,
        "dimension": "theoretical_orientation",
        "latent_dim": 3,
        "effect": 0.245,
        "p_value": 0.003
      }
    ]
  },
  "rq2_optimal_configurations": {
    "novelty_combined": {
      "feature_importance": [...],
      "r2_score": 0.67,
      "top_configurations": [...]
    }
  }
}
```

## Customization

### Custom Personas

Edit `config/config.yaml` to add new persona dimensions:

```yaml
personas:
  fields:
    - "digital_history"  # Add new field
    - "public_history"
```

### Custom Source Library

```python
from sources import SourceLibrary, PrimarySource

library = SourceLibrary()
library.initialize_embedding_model()

# Add your sources
sources = [
    PrimarySource(
        source_id="custom_001",
        title="Your Document",
        content="Document text...",
        source_type="text",
        metadata={"year": 1920, "author": "..."}
    )
]

library.add_sources_batch(sources)
library.save_index()
```

### Custom Agent Behavior

Modify the system prompt in `agents/multi_agent_system.py`:

```python
def get_system_prompt(self) -> str:
    return f"""You are a historian specialized in {self.persona.field}...

    Additional instructions:
    - Always cite sources
    - Consider counterfactuals
    ...
    """
```

## Advanced Usage

### Resume Interrupted Experiments

```bash
python run_experiment.py --mode run-only --start-from 25
```

### Use Custom Configuration

```bash
python run_experiment.py --config my_config.yaml
```

### Analyze Specific Outcomes

```python
analyzer = ExperimentAnalyzer()
analyzer.load_results("outputs/results.csv")

# Custom analysis
quality_metrics = analyzer.compute_thesis_quality_metrics()
print(quality_metrics.describe())

# Train SAE on custom features
analyzer.train_sae(latent_dim=64)
latent = analyzer.get_latent_representations()
```

## Research Questions Addressed

### RQ1: Differential Source Selection

The system tracks which sources each agent accesses during dialogue and uses:

1. **SAE** to learn latent representations of source selection patterns
2. **Double Debiasing** to estimate causal effects of persona characteristics on latent dimensions
3. **Feature importance** to rank which persona combinations drive different selection behaviors

**Key Insights**:
- Do marxist historians select different economic sources than poststructuralists?
- Do quantitative methods lead to broader or narrower source usage?
- How does field specialization interact with theoretical orientation?

### RQ2: Optimal Configurations for Quality

The system measures thesis outcomes via:

1. **Novelty**: Embedding distance from other theses (semantic uniqueness)
2. **Perplexity**: Language model surprise (complexity/unexpectedness)
3. **Quality**: Composite score including source diversity and argumentation

Then identifies which persona configurations maximize these outcomes.

**Key Insights**:
- Do diverse groups (mixed fields/methods) produce more novel theses?
- Which specific combinations yield highest quality?
- Is there a trade-off between novelty and consensus speed?

## Theory of Change

The system implements:

**Inductive Reasoning**: Agents search sources → observe patterns → form hypotheses

**Deductive Reasoning**: Agents propose theses → search for evidence → test predictions

**Mixed Approach**: Iterative dialogue allows both:
- Bottom-up pattern discovery from corpus
- Top-down hypothesis refinement through critique
- Emergent synthesis via collaborative reasoning

This prevents pre-selection bias (picking sources for a predetermined thesis) while enabling structured argumentation.

## Performance Notes

- Full factorial design with 7 fields × 5 methods × 5 eras × 7 orientations = **6,125 personas**
- Groups of 3 = **~38 billion possible combinations**
- Use `--strategy stratified` for manageable sampling
- Each experiment takes ~2-5 minutes depending on:
  - Number of dialogue turns (10-50)
  - Source searches per turn (0-3)
  - LLM latency

**Recommendations**:
- Start with 20-50 experiments for pilot testing
- Scale to 500+ for robust causal estimates
- Use GPU for FAISS if corpus > 10K sources

## Troubleshooting

**"Index is empty"**: Run setup first with `--mode setup-only`

**API rate limits**: Reduce `--n-samples` or add delays in `multi_agent_system.py`

**Memory issues**: Reduce `n_sources` or use `faiss-gpu` package

**Convergence issues**: Check `config.yaml` dialogue parameters:
```yaml
dialogue:
  min_turns: 10
  max_turns: 50  # Increase if agents don't reach consensus
```

## Citation

If you use this system in your research:

```bibtex
@software{multiagent_historian_2026,
  title={Multi-Agent Historian Research Simulation},
  author={Your Name},
  year={2026},
  description={Factorial design experiments with causal inference for
               analyzing collaborative historical research patterns}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open issues or pull requests for:
- Additional persona dimensions
- Alternative causal inference methods
- Improved dialogue strategies
- Enhanced quality metrics

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Claude](https://www.anthropic.com/) (Anthropic) or [GPT-4](https://openai.com/) (OpenAI)
