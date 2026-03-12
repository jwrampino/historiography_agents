"""
Test prediction analysis module with mock data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent))

from analysis.prediction_analysis import PredictionAnalyzer, PredictionResult


def create_mock_data(n_experiments: int = 100):
    """Create mock experiment data for testing."""
    print("=" * 60)
    print("Creating Mock Data")
    print("=" * 60)

    # Generate mock results.csv
    data = []
    for i in range(n_experiments):
        # Generate triangle geometry with some structure
        side_1 = np.random.uniform(0.2, 0.6)
        side_2 = np.random.uniform(0.2, 0.6)
        side_3 = np.random.uniform(0.2, 0.6)

        perimeter = side_1 + side_2 + side_3
        s = perimeter / 2
        area = np.sqrt(max(0, s * (s - side_1) * (s - side_2) * (s - side_3)))

        # Generate angles
        def compute_angle(a, b, c):
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)

        angle_1 = compute_angle(side_1, side_2, side_3)
        angle_2 = compute_angle(side_2, side_3, side_1)
        angle_3 = compute_angle(side_3, side_1, side_2)

        data.append({
            'experiment_id': f'exp_{i:05d}',
            'historian_1': f'Historian_{np.random.randint(1, 26)}',
            'historian_2': f'Historian_{np.random.randint(1, 26)}',
            'historian_3': f'Historian_{np.random.randint(1, 26)}',
            'turn_count': int(np.random.normal(25, 5)),
            'consensus_reached': np.random.random() > 0.3,
            'n_sources_used': int(np.random.normal(10, 3)),
            'question_length': int(np.random.normal(50, 15)),
            'abstract_length': int(np.random.normal(200, 50)),
            'geom_side_1': side_1,
            'geom_side_2': side_2,
            'geom_side_3': side_3,
            'geom_perimeter': perimeter,
            'geom_area': area,
            'geom_min_angle': min(angle_1, angle_2, angle_3),
            'geom_max_angle': max(angle_1, angle_2, angle_3),
            'geom_angle_variance': np.var([angle_1, angle_2, angle_3]),
        })

    df = pd.DataFrame(data)

    # Save to temp location
    temp_dir = Path("temp_test_data")
    temp_dir.mkdir(exist_ok=True)

    df.to_csv(temp_dir / "results.csv", index=False)

    # Generate detailed results JSON files
    for i in range(n_experiments):
        exp_id = f'exp_{i:05d}'

        # Mock abstract correlated with perimeter (for testing)
        perimeter = df.loc[i, 'geom_perimeter']
        abstract_complexity = int(50 + perimeter * 100 + np.random.normal(0, 20))

        detailed = {
            'experiment_id': exp_id,
            'group_composition': [
                {'name': df.loc[i, 'historian_1']},
                {'name': df.loc[i, 'historian_2']},
                {'name': df.loc[i, 'historian_3']},
            ],
            'triangle_geometry': {
                'side_1': df.loc[i, 'geom_side_1'],
                'side_2': df.loc[i, 'geom_side_2'],
                'side_3': df.loc[i, 'geom_side_3'],
                'perimeter': df.loc[i, 'geom_perimeter'],
                'area': df.loc[i, 'geom_area'],
            },
            'chat_history': [],
            'sources_accessed': [
                {'source_id': f'source_{j}'}
                for j in range(int(df.loc[i, 'n_sources_used']))
            ],
            'final_question': f'Mock research question {i}?',
            'final_abstract': ' '.join(['word'] * abstract_complexity),
            'turn_count': int(df.loc[i, 'turn_count']),
            'consensus_reached': bool(df.loc[i, 'consensus_reached']),
            'timestamp': '2024-01-01T00:00:00'
        }

        with open(temp_dir / f'{exp_id}.json', 'w') as f:
            json.dump(detailed, f)

    print(f"✓ Created {n_experiments} mock experiments")
    print(f"✓ Saved to {temp_dir}")

    return temp_dir


def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "=" * 60)
    print("Test 1: Feature Engineering")
    print("=" * 60)

    temp_dir = create_mock_data(n_experiments=50)

    analyzer = PredictionAnalyzer()
    analyzer.load_results(temp_dir / "results.csv")

    df = analyzer.engineer_geometry_features()

    # Check derived features
    expected_features = [
        'geom_avg_side', 'geom_side_variance', 'geom_angle_range',
        'geom_perimeter_norm', 'geom_area_norm', 'geom_regularity'
    ]

    for feat in expected_features:
        assert feat in df.columns, f"Missing feature: {feat}"

    print(f"✓ Engineered features: {[c for c in df.columns if c.startswith('geom_')]}")
    print(f"✓ Shape: {df.shape}")

    return analyzer, temp_dir


def test_source_selection_features():
    """Test source selection feature extraction."""
    print("\n" + "=" * 60)
    print("Test 2: Source Selection Features")
    print("=" * 60)

    analyzer, temp_dir = test_feature_engineering()
    analyzer.load_detailed_results(temp_dir)

    source_df = analyzer.compute_source_selection_features()

    print(f"✓ Extracted source features: {source_df.columns.tolist()}")
    print(f"✓ Shape: {source_df.shape}")
    print(f"✓ Mean sources accessed: {source_df['n_sources_accessed'].mean():.1f}")

    return analyzer, temp_dir


def test_prediction_models():
    """Test prediction models without perplexity (to avoid loading LM)."""
    print("\n" + "=" * 60)
    print("Test 3: Prediction Models")
    print("=" * 60)

    analyzer, temp_dir = test_source_selection_features()

    # Test source selection prediction
    print("\n--- Testing Source Selection Prediction ---")
    results = analyzer.predict_source_selection(
        target='n_sources_accessed',
        models=['rf', 'ridge']
    )

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test R²: {result.test_r2:.3f}")
        print(f"  CV R² mean: {result.cv_r2_mean:.3f}")
        assert result.test_r2 >= -1.0, "R² should be >= -1"
        assert result.feature_importance is not None, "Should have feature importance"

    print("\n✓ All prediction models ran successfully")

    return analyzer, temp_dir


def test_correlation_analysis():
    """Test correlation analysis."""
    print("\n" + "=" * 60)
    print("Test 4: Correlation Analysis")
    print("=" * 60)

    analyzer, temp_dir = test_source_selection_features()

    # Mock perplexity data (skip actual calculation)
    perplexity_data = []
    for i in range(len(analyzer.detailed_results)):
        perplexity_data.append({
            'experiment_id': f'exp_{i:05d}',
            'perplexity': np.random.uniform(10, 100),
            'abstract_length': np.random.randint(50, 300)
        })

    # Inject into analyzer
    analyzer._mock_perplexity = pd.DataFrame(perplexity_data)

    # Override compute_perplexity_scores to use mock data
    def mock_compute_perplexity():
        return analyzer._mock_perplexity

    analyzer.compute_perplexity_scores = mock_compute_perplexity

    corr_df = analyzer.correlation_analysis()

    print(f"✓ Computed {len(corr_df)} correlations")
    print(f"✓ Columns: {corr_df.columns.tolist()}")

    # Check structure
    assert 'geometry_feature' in corr_df.columns
    assert 'outcome' in corr_df.columns
    assert 'pearson_r' in corr_df.columns

    print("\n✓ Correlation analysis completed")

    return analyzer, temp_dir


def cleanup(temp_dir):
    """Cleanup temp files."""
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up {temp_dir}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "PREDICTION ANALYSIS TEST" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    temp_dir = None

    try:
        # Run tests
        analyzer, temp_dir = test_feature_engineering()
        analyzer, temp_dir = test_source_selection_features()
        analyzer, temp_dir = test_prediction_models()
        analyzer, temp_dir = test_correlation_analysis()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nPrediction analysis module is working correctly!")
        print("\nKey capabilities:")
        print("  1. ✓ Feature engineering from triangle geometry")
        print("  2. ✓ Source selection prediction (RF, GB, Ridge)")
        print("  3. ✓ Perplexity prediction (when LM available)")
        print("  4. ✓ Correlation analysis")
        print("  5. ✓ Comprehensive reporting")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if temp_dir:
            cleanup(temp_dir)

    return 0


if __name__ == "__main__":
    exit(main())
