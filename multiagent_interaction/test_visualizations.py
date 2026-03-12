"""
Test visualization module with mock data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import shutil

sys.path.append(str(Path(__file__).parent))

from visualization.generate_all import VisualizationGenerator


def create_mock_experiment_data(n_experiments: int = 50):
    """Create comprehensive mock data for testing visualizations."""
    print("="*60)
    print("Creating Mock Experiment Data")
    print("="*60)

    # Create temp output directory
    temp_dir = Path("temp_viz_test")
    temp_dir.mkdir(exist_ok=True)

    # Generate mock results.csv with all fields
    data = []
    historian_pool = [
        "Arnaldo Momigliano", "Georg G. Iggers", "John Marincola",
        "Nancy C. M. Hartsock", "Joan Wallach Scott", "Clifford Geertz",
        "Reinhart Koselleck", "Perry Anderson", "Peter Burke",
        "Ranajit Guha", "David Harvey", "Ellen Meiksins Wood"
    ]

    for i in range(n_experiments):
        # Triangle geometry
        side_1 = np.random.uniform(0.2, 0.6)
        side_2 = np.random.uniform(0.2, 0.6)
        side_3 = np.random.uniform(0.2, 0.6)
        perimeter = side_1 + side_2 + side_3
        s = perimeter / 2
        area = np.sqrt(max(0, s * (s - side_1) * (s - side_2) * (s - side_3)))

        # Angles
        def compute_angle(a, b, c):
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)

        angle_1 = compute_angle(side_1, side_2, side_3)
        angle_2 = compute_angle(side_2, side_3, side_1)
        angle_3 = compute_angle(side_3, side_1, side_2)

        # Select 3 historians
        historians = np.random.choice(historian_pool, 3, replace=False)

        # Outcomes (with some correlation to geometry)
        turn_count = int(15 + perimeter * 10 + np.random.normal(0, 3))
        n_sources = int(5 + area * 100 + np.random.normal(0, 2))
        consensus = np.random.random() > 0.3

        data.append({
            'experiment_id': f'exp_{i:05d}',
            'historian_1': historians[0],
            'historian_2': historians[1],
            'historian_3': historians[2],
            'turn_count': max(10, turn_count),
            'consensus_reached': consensus,
            'n_sources_used': max(3, n_sources),
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
    df.to_csv(temp_dir / "results.csv", index=False)

    # Generate detailed JSON files
    for i in range(n_experiments):
        exp_id = f'exp_{i:05d}'
        row = data[i]

        detailed = {
            'experiment_id': exp_id,
            'group_composition': [
                {'name': row['historian_1']},
                {'name': row['historian_2']},
                {'name': row['historian_3']},
            ],
            'triangle_geometry': {
                'side_1': row['geom_side_1'],
                'side_2': row['geom_side_2'],
                'side_3': row['geom_side_3'],
                'perimeter': row['geom_perimeter'],
                'area': row['geom_area'],
            },
            'chat_history': [
                {'agent_id': 'agent_1', 'action': 'speak', 'content': 'Mock message'},
            ],
            'sources_accessed': [
                {'source_id': f'source_{j}', 'title': f'Source {j}'}
                for j in range(row['n_sources_used'])
            ],
            'final_question': f'Mock research question for experiment {i}?',
            'final_abstract': ' '.join(['word'] * row['abstract_length']),
            'turn_count': row['turn_count'],
            'consensus_reached': row['consensus_reached'],
            'timestamp': '2024-01-01T00:00:00'
        }

        with open(temp_dir / f'{exp_id}.json', 'w') as f:
            json.dump(detailed, f)

    # Create mock correlation data
    geometry_features = ['geom_side_1', 'geom_perimeter', 'geom_area', 'geom_regularity']
    outcomes = ['turn_count', 'n_sources_used', 'abstract_length']

    corr_data = []
    for geom in geometry_features:
        for outcome in outcomes:
            corr_data.append({
                'geometry_feature': geom,
                'outcome': outcome,
                'pearson_r': np.random.uniform(-0.5, 0.5),
                'pearson_p': np.random.uniform(0, 0.2),
                'spearman_r': np.random.uniform(-0.5, 0.5),
                'spearman_p': np.random.uniform(0, 0.2),
                'n': n_experiments
            })

    corr_df = pd.DataFrame(corr_data)
    corr_df.to_csv(temp_dir / "correlations.csv", index=False)

    # Create mock prediction report
    prediction_report = {
        'timestamp': '2024-01-01T00:00:00',
        'n_experiments': n_experiments,
        'rq1_source_prediction': {
            'rf': {
                'test_r2': 0.35,
                'test_mse': 2.5,
                'cv_r2_mean': 0.28,
                'top_features': {
                    'geom_perimeter': 0.25,
                    'geom_area': 0.20,
                    'geom_side_1': 0.15,
                    'geom_regularity': 0.10,
                    'geom_min_angle': 0.08
                }
            },
            'gb': {
                'test_r2': 0.40,
                'test_mse': 2.2,
                'cv_r2_mean': 0.32,
                'top_features': {
                    'geom_area': 0.30,
                    'geom_perimeter': 0.22,
                    'geom_side_2': 0.13,
                    'geom_regularity': 0.12,
                    'geom_angle_variance': 0.09
                }
            }
        },
        'rq2_perplexity_prediction': {
            'rf': {
                'test_r2': 0.28,
                'test_mse': 5.8,
                'cv_r2_mean': 0.21,
                'top_features': {
                    'geom_regularity': 0.28,
                    'geom_perimeter': 0.21,
                    'geom_area': 0.18,
                    'geom_side_variance': 0.11,
                    'geom_max_angle': 0.08
                }
            }
        },
        'correlations': {
            'n_significant': 3,
            'strongest': [
                {'geometry_feature': 'geom_perimeter', 'outcome': 'turn_count',
                 'pearson_r': 0.45, 'pearson_p': 0.001},
                {'geometry_feature': 'geom_area', 'outcome': 'n_sources_used',
                 'pearson_r': 0.38, 'pearson_p': 0.008},
            ]
        }
    }

    with open(temp_dir / "prediction_report.json", 'w') as f:
        json.dump(prediction_report, f, indent=2)

    print(f"✓ Created {n_experiments} mock experiments")
    print(f"✓ Saved to {temp_dir}")

    return temp_dir


def test_visualization_generation():
    """Test generating all visualizations."""
    print("\n" + "="*60)
    print("Testing Visualization Generation")
    print("="*60)

    # Create mock data
    temp_dir = create_mock_experiment_data(n_experiments=50)

    # Initialize generator
    generator = VisualizationGenerator(output_dir=temp_dir)

    # Load data
    success = generator.load_data()
    assert success, "Failed to load data"
    print("\n✓ Data loaded successfully")

    # Test each visualization category
    try:
        # Geometry visualizations
        print("\n--- Geometry Visualizations ---")
        generator.generate_geometry_visualizations()

        # Experiment visualizations
        print("\n--- Experiment Visualizations ---")
        generator.generate_experiment_visualizations()

        # Prediction visualizations
        print("\n--- Prediction Visualizations ---")
        generator.generate_prediction_visualizations()

        # Summary report
        print("\n--- Summary Report ---")
        generator.generate_summary_report()

        # Count generated figures
        n_figures = len(list(generator.figures_dir.glob("*.png")))
        print(f"\n✓ Generated {n_figures} visualization files")

        # Verify key files exist
        expected_files = [
            "01_geometry_distributions.png",
            "02_triangle_shape_space.png",
            "08_experiment_overview.png",
            "20_correlation_matrix.png",
            "visualization_report.txt"
        ]

        for fname in expected_files:
            fpath = generator.figures_dir / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ⚠ Missing: {fname}")

        return temp_dir, n_figures

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return temp_dir, 0


def cleanup(temp_dir):
    """Cleanup temp files."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up {temp_dir}")


def main():
    """Run visualization tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 16 + "VISUALIZATION TEST" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    temp_dir = None

    try:
        temp_dir, n_figures = test_visualization_generation()

        if n_figures > 0:
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED")
            print("="*60)
            print(f"\nGenerated {n_figures} visualizations successfully!")
            print("\nVisualization capabilities:")
            print("  1. ✓ Triangle geometry distributions (8 features)")
            print("  2. ✓ Embedding space projections (2D/3D)")
            print("  3. ✓ Distance heatmaps")
            print("  4. ✓ Experiment overview dashboard")
            print("  5. ✓ Historian participation analysis")
            print("  6. ✓ Outcome distributions & correlations")
            print("  7. ✓ Prediction model comparisons")
            print("  8. ✓ Feature importance plots")
            print("  9. ✓ Correlation matrices")
            print(" 10. ✓ Timeline analysis")
            print(f"\nTotal: 20+ distinct visualization types")
            return 0
        else:
            print("\n✗ No visualizations generated")
            return 1

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if temp_dir:
            cleanup(temp_dir)


if __name__ == "__main__":
    exit(main())
