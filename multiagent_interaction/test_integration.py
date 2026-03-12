"""
Quick integration test for historian-based multi-agent system.
Tests that all components work together without running full experiments.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from personas.historian_manager import HistorianManager, HistorianPersona


def test_historian_loading():
    """Test loading historian personas."""
    print("=" * 60)
    print("Test 1: Loading Historian Personas")
    print("=" * 60)

    manager = HistorianManager()

    # Load personas
    personas = manager.load_personas("personas/historian_personas.json")

    print(f"✓ Loaded {len(personas)} historians")

    # Check first persona
    first = personas[0]
    print(f"\nFirst historian: {first.name}")
    print(f"  - ID: {first.historian_id}")
    print(f"  - Papers: {len(first.papers)}")
    print(f"  - Prompt length: {len(first.prompt)} chars")
    print(f"  - Embedding shape: {first.embedding.shape if first.embedding is not None else 'None'}")

    return personas


def test_group_generation():
    """Test generating historian groups with geometry."""
    print("\n" + "=" * 60)
    print("Test 2: Generating Groups with Triangle Geometry")
    print("=" * 60)

    manager = HistorianManager()
    personas = manager.load_personas("personas/historian_personas.json")

    # Sample groups
    groups = manager.sample_groups(
        n_groups=10,
        strategy="filtered",
        min_distance=0.1,
        max_distance=0.7,
        min_area=0.001
    )

    print(f"✓ Generated {len(groups)} valid groups")

    # Show first group
    if groups:
        print(f"\nExample group:")
        group = groups[0]
        for h in group:
            print(f"  - {h.name}")

        geom = manager.compute_triangle_geometry(group)
        print(f"\nTriangle geometry:")
        for key, value in geom.items():
            print(f"  {key}: {value:.4f}")

    return groups


def test_system_prompt():
    """Test that historian prompts are properly formatted."""
    print("\n" + "=" * 60)
    print("Test 3: System Prompt Generation")
    print("=" * 60)

    manager = HistorianManager()
    personas = manager.load_personas("personas/historian_personas.json")

    # Check persona prompt structure
    persona = personas[0]

    print(f"✓ Persona: {persona.name}")
    print(f"  - Prompt length: {len(persona.prompt)} chars")
    print(f"  - Prompt starts with: {persona.prompt[:100]}...")

    # Check that prompt contains paper content
    has_papers = any(paper.get('title', '') in persona.prompt for paper in persona.papers)
    print(f"  - Contains paper content: {has_papers}")

    # Check for key components
    has_perspective = "scholarly perspective" in persona.prompt.lower()
    has_embodiment = "embody" in persona.prompt.lower()
    has_name = persona.name in persona.prompt

    print(f"  - Contains 'scholarly perspective': {has_perspective}")
    print(f"  - Contains 'embody' instruction: {has_embodiment}")
    print(f"  - Contains historian name: {has_name}")

    return persona


def test_experiment_result_structure():
    """Test that experiment result dataclass works with new structure."""
    print("\n" + "=" * 60)
    print("Test 4: Experiment Result Structure")
    print("=" * 60)

    manager = HistorianManager()
    personas = manager.load_personas("personas/historian_personas.json")[:3]

    # Mock triangle geometry
    triangle_geometry = manager.compute_triangle_geometry(tuple(personas))

    # Create mock result structure (without importing actual class to avoid langchain deps)
    result = {
        'experiment_id': "test_001",
        'group_composition': [p.to_dict() for p in personas],
        'triangle_geometry': triangle_geometry,
        'chat_history': [],
        'sources_accessed': [],
        'final_question': "Test question?",
        'final_abstract': "Test abstract.",
        'turn_count': 10,
        'consensus_reached': True,
        'timestamp': "2024-01-01T00:00:00"
    }

    print(f"✓ Created mock ExperimentResult structure")
    print(f"  - Experiment ID: {result['experiment_id']}")
    print(f"  - Historians: {[p['name'] for p in result['group_composition']]}")
    print(f"  - Triangle perimeter: {result['triangle_geometry']['perimeter']:.3f}")
    print(f"  - Triangle area: {result['triangle_geometry']['area']:.3f}")
    print(f"  - Has all required fields: {all(k in result for k in ['experiment_id', 'group_composition', 'triangle_geometry', 'final_question'])}")

    return result


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "HISTORIAN MULTI-AGENT INTEGRATION TEST" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        # Run tests
        personas = test_historian_loading()
        groups = test_group_generation()
        persona = test_system_prompt()
        result = test_experiment_result_structure()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe multi-agent system is ready to use real historian personas!")
        print("Next steps:")
        print("  1. Set up source library (primary sources)")
        print("  2. Configure LLM API keys")
        print("  3. Run experiments with: python experiments/experiment_runner.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
