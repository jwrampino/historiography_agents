# Historiography Agents

A multi-agent system for studying how historian personas with different scholarly backgrounds converge on interpretations of historical events through collaborative deliberation.

## Overview

This project implements a computational framework for studying historiographical discourse using AI agents. It combines:
- Real historian personas derived from OpenAlex scholarly data
- A corpus of multimodal historical sources (Library of Congress, Internet Archive, NARA, Smithsonian)
- Multi-agent deliberation pipelines with geometric analysis of viewpoint diversity
- Predictive models for convergence patterns

## Project Structure

```
historiography_agents/
├── agents/                          # Multi-agent orchestration and interaction
│   ├── __init__.py
│   ├── agent_llm.py                # LLM wrapper for historian agents
│   ├── historian_manager.py        # Manages historian personas from OpenAlex data
│   ├── interaction_pipeline.py     # Two-stage deliberation pipeline (hypotheses → synthesis)
│   ├── run_factorial_experiment.py # Factorial experiment orchestrator
│   ├── source_retrieval.py         # Retrieves relevant historical sources
│   ├── storage.py                  # Experiment data persistence
│   ├── visualization.py            # Analysis and visualization utilities
│   └── recover_analysis.py         # Recovery/continuation of interrupted experiments
│
├── sources/                         # Phase 1: Corpus construction pipeline
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # Configuration: paths, API keys, constants
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py             # Text/image embedders (sentence-transformers, CLIP)
│   │   └── faiss_index.py          # FAISS IVF-PQ index for semantic search
│   ├── ingestors/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base ingestor with rate limiting
│   │   ├── loc_ingestor.py         # Library of Congress + Chronicling America
│   │   ├── internet_archive_ingestor.py
│   │   └── nara_smithsonian_ingestor.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── schema.py               # CorpusItem dataclass + DuckDB schema
│   │   └── corpus_store.py         # DuckDB CRUD operations
│   ├── utils/
│   │   ├── __init__.py
│   │   └── text_utils.py           # Text cleaning, era detection, normalization
│   ├── pipeline.py                 # Main corpus construction orchestrator
│   ├── example_config.json         # Example configuration for pipeline
│   └── README.md                   # Detailed Phase 1 pipeline documentation
│
├── data/                            # All data artifacts
│   ├── db/
│   │   └── corpus.duckdb           # Master historical corpus database
│   ├── embeddings/                 # Per-item .npy vectors (sharded by UUID prefix)
│   │   ├── 00/ ... ff/             # 256 subdirectories for sharded storage
│   ├── index/
│   │   ├── corpus.faiss            # FAISS semantic search index
│   │   └── id_map.json             # Integer ID ↔ source UUID mapping
│   ├── logs/                       # Pipeline and experiment logs
│   ├── factorial_n10/              # Small-scale factorial experiment (n=10)
│   │   ├── experiments.duckdb      # Experiment database
│   │   ├── triads.csv              # Triad configurations and geometry
│   │   ├── proposals.csv           # Individual historian hypotheses
│   │   ├── synthesis.csv           # Final synthesis outputs
│   │   ├── llm_interactions.csv    # All LLM API calls and responses
│   │   ├── convergence_results.csv # Convergence metrics
│   │   └── logs/                   # Experiment logs
│   ├── factorial_n207/             # Full-scale factorial experiment (n=207)
│   │   ├── experiments.duckdb      # Experiment database
│   │   ├── triads.csv              # All triad configurations
│   │   ├── proposals.csv           # Individual hypotheses
│   │   ├── synthesis.csv           # Synthesis outcomes
│   │   ├── llm_interactions.csv    # Complete LLM interaction log
│   │   ├── source_geometry.csv     # Source usage patterns and diversity
│   │   ├── convergence_results.csv # Convergence analysis
│   │   ├── perplexity_features.csv # Perplexity-based features
│   │   ├── prediction_model.json   # Trained convergence prediction model
│   │   ├── feature_importance.csv  # Model feature importance
│   │   ├── inference_results.json  # Inference performance metrics
│   │   ├── ablation_study.json     # Feature ablation results
│   │   └── figures/                # Generated visualizations (68 plots)
│   └── corpus_export.csv           # Full corpus export (9MB, ~9000 items)
│
├── ranked_historians.csv            # OpenAlex historians ranked by relevance
├── topic_papers.csv                 # Papers on American history topics (766KB)
├── paper_author_edges.csv           # Paper-author relationship graph (210KB)
├── persona.ipynb                    # Jupyter notebook: persona construction workflow
├── history_agents.yaml              # Conda environment specification
└── .gitattributes                   # Git LFS configuration for large files

```

## Key Components

### 1. Historian Personas (`agents/historian_manager.py`)
- Constructs personas from real historians' published work via OpenAlex API
- Each persona includes scholarly publications, abstracts, and 768-d embedding vectors
- Supports geometric diversity analysis for triad selection

### 2. Source Retrieval (`agents/source_retrieval.py`)
- Retrieves relevant historical documents from the corpus using semantic search
- Uses FAISS index for efficient similarity search
- Returns ranked source packets for each historian

### 3. Interaction Pipeline (`agents/interaction_pipeline.py`)
Two-stage deliberation process:
- **Stage 1**: Each historian independently formulates a hypothesis based on retrieved sources
- **Stage 2**: Historians deliberate and synthesize a final interpretation

### 4. Corpus Construction (`sources/`)
End-to-end pipeline for building the multimodal historical corpus:
- Ingests from 4 major archives (LoC, Internet Archive, NARA, Smithsonian)
- Generates cross-modal embeddings (text: sentence-transformers, images: CLIP)
- Builds FAISS IVF-PQ index for semantic search
- Stores metadata in DuckDB

### 5. Factorial Experiments (`agents/run_factorial_experiment.py`)
- Systematically tests all possible historian triads
- Tracks convergence metrics, source usage, and deliberation patterns
- Stores results in DuckDB with comprehensive logging

### 6. Analysis & Visualization (`agents/visualization.py`)
- Convergence analysis and trajectory visualization
- Source usage patterns and diversity metrics
- Predictive modeling of convergence outcomes

## Data Files

### Input Data
- `ranked_historians.csv`: 89KB, pre-ranked historians from OpenAlex
- `topic_papers.csv`: 766KB, papers on American history topics
- `paper_author_edges.csv`: 210KB, paper-author relationships

### Output Data
- `data/corpus_export.csv`: 9MB, complete corpus with metadata
- `data/factorial_n207/experiments.duckdb`: 12MB, full experiment database
- `data/factorial_n207/llm_interactions.csv`: 5MB, all LLM interactions
- `data/factorial_n207/figures/`: 68 visualization files

## Environment Setup

### Conda Environment
```bash
conda env create -f history_agents.yaml
conda activate history_agents
```

### Dependencies
- **Core**: Python 3.10, NumPy, Pandas, DuckDB
- **ML/Embeddings**: PyTorch, Transformers, Sentence-Transformers, FAISS
- **LLM/Agents**: LangChain, LangGraph, Anthropic, OpenAI, AutoGen
- **Archives**: internetarchive SDK

## Quick Start

### 1. Build Corpus (Phase 1)
```bash
# From the project root
python -m sources.pipeline --query "reconstruction era" --max-items 50
```

### 2. Run Factorial Experiment
```bash
# From the agents directory
python run_factorial_experiment.py
```

### 3. Analyze Results
```bash
python visualization.py
```

## Usage Examples

### Retrieve Historical Sources
```python
from agents.source_retrieval import SourceRetriever

retriever = SourceRetriever(corpus_db="data/db/corpus.duckdb")
sources = retriever.retrieve("reconstruction era freedmen", top_k=10)
```

### Load Historian Personas
```python
from agents.historian_manager import HistorianManager

manager = HistorianManager(
    papers_csv="topic_papers.csv",
    edges_csv="paper_author_edges.csv"
)
historians = manager.get_personas(n=3)
```

### Run Interaction Pipeline
```python
from agents.interaction_pipeline import InteractionPipeline

pipeline = InteractionPipeline(
    source_retriever=retriever,
    llm=agent_llm
)
result = pipeline.run_triad(historians, query="reconstruction era")
```

## Research Questions

This system addresses:
1. How does viewpoint diversity (measured geometrically in embedding space) affect convergence?
2. What source usage patterns emerge during deliberation?
3. Can convergence outcomes be predicted from initial historian configurations?
4. How do real scholarly backgrounds influence interpretation formation?

## Key Design Decisions

**Embedding Alignment**: All modalities projected to 1024-d space (text: sentence-transformers 768→1024, images: CLIP ViT-L/14 768→1024)

**Deduplication**: Items keyed by UUID with DuckDB uniqueness constraints

**Rate Limiting**: All archive ingestors respect `REQUEST_DELAY_SECONDS` (default: 0.5s)

**FAISS Fallback**: Corpora <9,984 vectors use flat IndexFlatIP; larger corpora use IVF-PQ

**Rights**: NARA items default to public domain (17 U.S.C. § 105)

## Experiment Outputs

### factorial_n207/ (Full Experiment)
- 207 triads tested
- ~4.97MB of LLM interactions logged
- 68 visualization figures generated
- Convergence prediction model trained
- Feature importance and ablation studies completed

### Metrics Tracked
- Convergence rates and trajectories
- Source diversity and overlap
- Perplexity-based linguistic features
- Geometric diversity of triads
- Prediction model performance

## File Formats

- **DuckDB**: Structured experiment data and corpus metadata
- **CSV**: Portable exports of triads, proposals, synthesis, interactions
- **NPY**: NumPy arrays for embeddings (sharded by UUID prefix)
- **JSON**: Configuration files and model outputs
- **IPYNB**: Jupyter notebooks for exploration and analysis

## License & Attribution

Uses data from:
- Library of Congress (public domain)
- Internet Archive (various licenses)
- NARA (U.S. federal records, public domain)
- Smithsonian Institution (various licenses)
- OpenAlex (scholarly metadata, CC0)

Always verify individual item rights before redistribution.

## Contact & Support

For questions about this implementation, refer to the detailed documentation in:
- `sources/README.md` - Corpus construction pipeline
- `agents/` - Multi-agent system architecture
- `data/factorial_n207/` - Full experimental results

---

**Project**: MACS 37005 Final Project - Historiography Agents
**Status**: Completed (experiments run, analysis generated, figures exported)
