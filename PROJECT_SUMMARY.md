# Project Summary: Multi-Agent Historical Synthesis with Convergence Prediction

**MACS 37005 Final Project**
**Date**: March 2026

## Project Overview

This project implements a multi-agent system where historian personas—constructed from real scholarly publications—collaborate on historical research tasks. The system predicts whether groups of historians converge toward shared interpretations based on their intellectual diversity and the sources they examine.

## Complete System Architecture

```
Final/
├── persona.ipynb                    # Historian persona generation from OpenAlex
├── topic_papers.csv                 # 939 papers from historians
├── paper_author_edges.csv           # Paper-author relationships
├── ranked_historians.csv            # Top 25 historians by relevance
│
├── historian_pipeline/              # Phase 1: Corpus construction
│   ├── config/
│   │   └── settings.py              # Configuration and paths
│   ├── ingestors/                   # Archive API connectors
│   │   ├── base.py
│   │   ├── loc_ingestor.py
│   │   ├── internet_archive_ingestor.py
│   │   └── nara_smithsonian_ingestor.py
│   ├── embeddings/
│   │   ├── embedder.py              # Text + image embeddings (768-d)
│   │   └── faiss_index.py           # Vector search index
│   ├── storage/
│   │   ├── schema.py                # Data schema
│   │   └── corpus_store.py          # DuckDB storage
│   ├── utils/
│   │   └── text_utils.py            # Text processing
│   ├── pipeline.py                  # Main corpus pipeline
│   └── requirements.txt             # Dependencies
│
├── agents/                          # NEW: Multi-agent system
│   ├── __init__.py
│   ├── historian_manager.py         # Load & manage personas (from your code)
│   ├── source_retrieval.py          # Retrieve multimodal sources
│   ├── agent_llm.py                 # GPT-3.5-turbo interface
│   ├── interaction_pipeline.py      # Two-stage interaction
│   ├── convergence_analysis.py      # Embedding-based convergence
│   ├── prediction_model.py          # Logistic regression prediction
│   ├── storage.py                   # Experiment database
│   ├── run_experiment.py            # Main orchestrator
│   ├── test_setup.py                # Setup validation
│   ├── example_usage.py             # Usage examples
│   ├── requirements.txt             # Additional dependencies
│   └── README.md                    # Full documentation
│
├── data/
│   ├── raw/                         # Downloaded archive files
│   ├── embeddings/                  # .npy embedding vectors
│   ├── index/                       # FAISS index files
│   ├── db/
│   │   ├── corpus.duckdb            # Main corpus database
│   │   └── experiments.duckdb       # NEW: Experiment database
│   ├── corpus_export.csv            # Corpus CSV export
│   ├── downloaded_images/           # NEW: Downloaded source images
│   └── agent_experiments/           # NEW: Experiment outputs
│       ├── triads.csv
│       ├── proposals.csv
│       ├── synthesis.csv
│       ├── convergence_results.csv
│       ├── prediction_model.json
│       ├── feature_importance.csv
│       ├── inference_results.json
│       └── experiment_summary.json
│
├── QUICKSTART.md                    # NEW: Quick start guide
└── PROJECT_SUMMARY.md               # NEW: This file
```

## Key Components Added

### 1. **Historian Manager** (`historian_manager.py`)
- Uses your provided code
- Loads personas from OpenAlex data (topic_papers.csv, paper_author_edges.csv)
- Computes historian embeddings by averaging paper abstracts
- Generates triads with triangle geometry constraints
- Stratified sampling for diverse intellectual perspectives

### 2. **Source Retrieval** (`source_retrieval.py`)
- Derives retrieval queries from historians' research areas
- Retrieves 3 text sources + 2 image sources per historian
- Uses existing corpus FAISS index for semantic search
- Downloads and stores image files locally
- Formats sources for agent consumption

### 3. **Agent LLM** (`agent_llm.py`)
- OpenAI GPT-3.5-turbo interface
- Generates individual historian proposals (Stage 1)
- Synthesizes group abstracts (Stage 2)
- Structured prompt parsing
- Exponential backoff retry logic

### 4. **Interaction Pipeline** (`interaction_pipeline.py`)
- **Stage 1**: Each historian generates:
  - Research question
  - 3-4 sentence abstract
  - Selection of 2 most useful sources
- **Stage 2**: System synthesizes:
  - Shared research question
  - Final merged abstract (4-5 sentences)
  - 3-5 selected sources
- Single-shot synthesis (no long conversations)

### 5. **Convergence Analysis** (`convergence_analysis.py`)
- Embeds individual abstracts + final abstract
- Computes centroid of historian positions
- Measures distances to centroid
- **Binary convergence label**:
  ```
  converged = 1 if distance(final, centroid) < mean(distance(individuals, centroid))
  converged = 0 otherwise
  ```
- Additional statistics: variance, pairwise similarities, convergence delta

### 6. **Prediction Model** (`prediction_model.py`)
- **Features**: Triangle geometry (perimeter, area, angles), distances
- **Model**: Logistic regression (interpretable, works with small N)
- **Output**: In-sample accuracy, precision, recall, F1
- **Feature importance analysis**
- Saves model parameters to JSON

### 7. **Inference Analysis** (`prediction_model.py`)
- **Hypothesis**: Greater diversity reduces convergence
- Splits triads by median triangle area (diversity proxy)
- Compares convergence rates: high diversity vs low diversity
- **Statistical test**: Fisher's exact test
- Reports odds ratio and p-value

### 8. **Storage** (`storage.py`)
- DuckDB tables: `triads`, `proposals`, `synthesis`, `convergence_results`
- CSV export for all tables
- Joined views for analysis
- Supports both in-memory (testing) and persistent storage

### 9. **Main Orchestrator** (`run_experiment.py`)
- End-to-end pipeline:
  1. Load historian personas
  2. Sample N triads (stratified)
  3. Run interaction experiments
  4. Analyze convergence
  5. Train prediction model
  6. Run inference analysis
  7. Export all results
- Command-line interface
- Comprehensive logging
- Progress tracking

## Data Flow

```
OpenAlex Papers
    ↓
Historian Personas (embeddings from abstracts)
    ↓
Triad Formation (triangle geometry)
    ↓
Source Retrieval (from corpus, derived from historian papers)
    ↓
Stage 1: Individual Proposals (GPT-3.5)
    ↓
Stage 2: Synthesis (GPT-3.5)
    ↓
Convergence Analysis (embedding distances)
    ↓
Prediction Model (logistic regression)
    ↓
Inference Analysis (diversity vs convergence)
```

## Models Used

1. **Text Embedding**: `sentence-transformers/all-mpnet-base-v2` (768-d)
   - Historian papers → historian embeddings
   - Corpus text → searchable embeddings
   - Generated abstracts → convergence analysis

2. **Image Embedding**: CLIP ViT-L/14 (768-d)
   - Corpus images → searchable embeddings

3. **Agent Reasoning**: GPT-3.5-turbo
   - Individual proposals
   - Group synthesis

4. **Prediction**: Logistic regression (scikit-learn)
   - Binary convergence prediction

## Key Design Decisions

### Why GPT-3.5-turbo?
- Cost-effective ($0.001/1K tokens vs $0.03 for GPT-4)
- Sufficient for structured reasoning tasks
- Faster response times
- ~$1-2 for 10 triads

### Why Stratified Sampling?
- Ensures diversity across triads
- Mixes high/low diversity groups for inference testing
- Maximizes geometric coverage

### Why Binary Convergence?
- More stable than predicting continuous embeddings
- Interpretable outcome
- Works with small sample sizes (N=10)

### Why Single-Shot Synthesis?
- Original design had multi-round conversations (expensive, slow)
- Single structured prompt is:
  - Cheaper (~75% cost reduction)
  - Faster (~80% time reduction)
  - More predictable
  - Still captures group dynamics

### Why In-Sample Metrics?
- N=10 is proof-of-concept
- Too small for train/test split
- Cross-validation would be unstable
- Results labeled as exploratory, not definitive

## Usage

### Basic
```bash
# Install
pip install -r historian_pipeline/requirements.txt
export OPENAI_API_KEY="your-key"

# Run
python -m agents.run_experiment --n-triads 10
```

### Advanced
```bash
# Custom configuration
python -m agents.run_experiment \
    --n-triads 20 \
    --output-dir my_experiment \
    --log-level DEBUG

# Programmatic
python
>>> from agents.run_experiment import ExperimentRunner
>>> runner = ExperimentRunner(n_triads=10)
>>> summary = runner.run()
>>> runner.close()
```

## Outputs

All results saved to `data/agent_experiments/`:

1. **triads.csv**: Triad metadata + triangle geometry
2. **proposals.csv**: Individual historian proposals
3. **synthesis.csv**: Final merged abstracts
4. **convergence_results.csv**: Convergence metrics + labels
5. **prediction_model.json**: Trained model parameters
6. **feature_importance.csv**: Feature coefficients
7. **inference_results.json**: Diversity vs convergence analysis
8. **experiment_summary.json**: Overall summary

## Testing

```bash
# Validate setup
python -m agents.test_setup

# Run examples
python -m agents.example_usage

# Quick test (2 triads)
python -m agents.run_experiment --n-triads 2
```

## Performance

- **Setup**: 5 minutes
- **Corpus building**: 10 minutes (one-time)
- **Per triad**: 1-3 minutes (depends on API latency)
- **10 triads**: 10-30 minutes total
- **Cost**: ~$1-2 for 10 triads

## Limitations

1. **Sample size**: N=10 is small (proof-of-concept)
2. **API dependency**: Requires OpenAI API access
3. **Corpus dependency**: Needs pre-built corpus for retrieval
4. **English only**: All models are English-language
5. **Static personas**: Personas don't learn/adapt during interaction

## Future Extensions

1. **Larger N**: Run 50-100 triads for robust statistics
2. **Cross-validation**: With larger N, use proper train/test split
3. **Deeper interactions**: Multi-round conversations
4. **Dynamic personas**: Allow historians to update based on sources
5. **More modalities**: Add audio/video sources
6. **Different LLMs**: Test Claude, GPT-4, etc.
7. **Longitudinal studies**: Track convergence over time

## Dependencies

Core:
- Python 3.10+
- sentence-transformers (embeddings)
- openai (GPT-3.5)
- duckdb (storage)
- faiss-cpu (vector search)
- scikit-learn (prediction)
- scipy (statistics)

Full list: `historian_pipeline/requirements.txt`

## License

Academic research project - MACS 37005 Final

## Acknowledgments

- OpenAlex API for historian publication data
- Library of Congress, Internet Archive, NARA, Smithsonian for corpus sources
- OpenAI GPT-3.5-turbo for agent reasoning

---

**Status**: ✅ Complete and ready to run
**Last Updated**: March 12, 2026
