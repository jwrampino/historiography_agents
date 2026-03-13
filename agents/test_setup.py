"""
Test script to validate the multi-agent system setup.
Run this to check if all components are properly installed.

Usage:
    python -m agents.test_setup
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")

    try:
        import numpy
        logger.info("  ✓ numpy")
    except ImportError:
        logger.error("  ✗ numpy missing: pip install numpy")
        return False

    try:
        import pandas
        logger.info("  ✓ pandas")
    except ImportError:
        logger.error("  ✗ pandas missing: pip install pandas")
        return False

    try:
        import duckdb
        logger.info("  ✓ duckdb")
    except ImportError:
        logger.error("  ✗ duckdb missing: pip install duckdb")
        return False

    try:
        import sentence_transformers
        logger.info("  ✓ sentence-transformers")
    except ImportError:
        logger.error("  ✗ sentence-transformers missing: pip install sentence-transformers")
        return False

    try:
        import sklearn
        logger.info("  ✓ scikit-learn")
    except ImportError:
        logger.error("  ✗ scikit-learn missing: pip install scikit-learn")
        return False

    try:
        import scipy
        logger.info("  ✓ scipy")
    except ImportError:
        logger.error("  ✗ scipy missing: pip install scipy")
        return False

    try:
        import openai
        logger.info("  ✓ openai")
    except ImportError:
        logger.error("  ✗ openai missing: pip install openai")
        return False

    try:
        from PIL import Image
        logger.info("  ✓ Pillow")
    except ImportError:
        logger.error("  ✗ Pillow missing: pip install Pillow")
        return False

    return True


def test_data_files():
    """Test if required data files exist."""
    logger.info("\nTesting data files...")
    from pathlib import Path

    base_dir = Path(__file__).parent.parent

    files_to_check = [
        'topic_papers.csv',
        'paper_author_edges.csv'
    ]

    all_exist = True
    for file in files_to_check:
        path = base_dir / file
        if path.exists():
            logger.info(f"  ✓ {file}")
        else:
            logger.error(f"  ✗ {file} not found at {path}")
            all_exist = False

    return all_exist


def test_corpus():
    """Test if corpus database exists."""
    logger.info("\nTesting corpus database...")
    from pathlib import Path

    corpus_db = Path("data/db/corpus.duckdb")

    if corpus_db.exists():
        logger.info(f"  ✓ Corpus database found at {corpus_db}")
        return True
    else:
        logger.warning(
            f"  ⚠ Corpus database not found at {corpus_db}\n"
            "  Run: python -m historian_pipeline.pipeline --query \"history\" --max-items 500"
        )
        return False


def test_api_key():
    """Test if OpenAI API key is set."""
    logger.info("\nTesting OpenAI API key...")
    import os

    if os.getenv("OPENAI_API_KEY"):
        logger.info("  ✓ OPENAI_API_KEY is set")
        return True
    else:
        logger.warning(
            "  ⚠ OPENAI_API_KEY not set\n"
            "  Set it with: export OPENAI_API_KEY='your-key-here'"
        )
        return False


def test_component_initialization():
    """Test if components can be initialized."""
    logger.info("\nTesting component initialization...")

    try:
        from agents.historian_manager import HistorianManager
        manager = HistorianManager(n_historians=5)
        logger.info("  ✓ HistorianManager")
    except Exception as e:
        logger.error(f"  ✗ HistorianManager failed: {e}")
        return False

    try:
        from agents.source_retrieval import SourceRetriever
        retriever = SourceRetriever()
        logger.info("  ✓ SourceRetriever")
    except Exception as e:
        logger.error(f"  ✗ SourceRetriever failed: {e}")
        return False

    try:
        from agents.convergence_analysis import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer()
        logger.info("  ✓ ConvergenceAnalyzer")
    except Exception as e:
        logger.error(f"  ✗ ConvergenceAnalyzer failed: {e}")
        return False

    try:
        from agents.storage import ExperimentStorage
        storage = ExperimentStorage(db_path=":memory:")
        storage.close()
        logger.info("  ✓ ExperimentStorage")
    except Exception as e:
        logger.error(f"  ✗ ExperimentStorage failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("MULTI-AGENT SYSTEM SETUP TEST")
    logger.info("="*60 + "\n")

    results = {
        'imports': test_imports(),
        'data_files': test_data_files(),
        'corpus': test_corpus(),
        'api_key': test_api_key(),
        'components': test_component_initialization()
    }

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n✓ All tests passed! System ready to run.")
        logger.info("\nRun experiment with:")
        logger.info("  python -m agents.run_experiment --n-triads 10")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
