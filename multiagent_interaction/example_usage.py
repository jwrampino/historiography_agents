"""
Example Usage: Demonstrates basic functionality of the multi-agent system.
Run this after initial setup to test the system.
"""

from pathlib import Path
import json

# Import components
from personas import PersonaManager, HistorianPersona
from sources import SourceLibrary, PrimarySource, create_example_library
from agents import MultiAgentDialogueSystem
from experiments import ExperimentRunner
from analysis import ExperimentAnalyzer


def example_1_create_personas():
    """Example 1: Create and explore historian personas."""
    print("="*60)
    print("Example 1: Creating Historian Personas")
    print("="*60)

    manager = PersonaManager()

    # Create full persona grid
    personas = manager.create_persona_grid()
    print(f"\nCreated {len(personas)} historian personas")

    # Show some examples
    print("\nExample personas:")
    for i, persona in enumerate(personas[:5]):
        print(f"  {i+1}. {persona.persona_id}")
        print(f"     Field: {persona.field}, Method: {persona.method}")
        print(f"     Era: {persona.era}, Orientation: {persona.theoretical_orientation}")

    # Generate groups for experiments
    groups = manager.generate_stratified_groups(group_size=3, n_samples=5)
    print(f"\nGenerated {len(groups)} sample groups")

    print("\nExample group composition:")
    for persona in groups[0]:
        print(f"  - {persona.field} historian using {persona.method} methods")

    return manager


def example_2_source_library():
    """Example 2: Create and search source library."""
    print("\n" + "="*60)
    print("Example 2: Source Library with FAISS")
    print("="*60)

    # Create example library
    library = create_example_library(n_sources=50)
    print(f"\nCreated library with {len(library.sources)} sources")

    # Test search
    queries = [
        "labor movements and workers' rights",
        "medieval trade and commerce",
        "women's political activism"
    ]

    for query in queries:
        print(f"\nSearch: '{query}'")
        results = library.search(query, k=3)

        for i, (source, score) in enumerate(results, 1):
            print(f"  {i}. {source.title} (score: {score:.3f})")

    # Show access log
    print(f"\nTotal searches logged: {len(library.access_log)}")

    return library


def example_3_single_experiment():
    """Example 3: Run a single multi-agent experiment."""
    print("\n" + "="*60)
    print("Example 3: Single Multi-Agent Experiment")
    print("="*60)

    # Setup
    persona_manager = PersonaManager()
    personas = persona_manager.create_persona_grid()[:3]  # Use first 3

    print("\nAgent personas:")
    for persona in personas:
        print(f"  - {persona.persona_id}: {persona.field} / {persona.theoretical_orientation}")

    # Note: This would actually run the dialogue with LLM calls
    # Uncomment to run (requires API key):
    """
    system = MultiAgentDialogueSystem()
    system.source_library = create_example_library(n_sources=100)

    print("\nRunning dialogue simulation...")
    final_state = system.run_experiment(personas, experiment_id="demo_001")

    print(f"\nResults:")
    print(f"  Turns: {final_state.turn_count}")
    print(f"  Consensus: {final_state.consensus_reached}")
    print(f"  Sources accessed: {len(final_state.sources_accessed)}")
    print(f"\nFinal question: {final_state.final_question}")
    """

    print("\n(Skipped actual LLM execution - requires API key)")


def example_4_analyze_results():
    """Example 4: Analyze experiment results."""
    print("\n" + "="*60)
    print("Example 4: Causal Analysis")
    print("="*60)

    output_dir = Path("outputs")

    if not output_dir.exists() or not (output_dir / "results.csv").exists():
        print("\nNo results found. Run experiments first with:")
        print("  python run_experiment.py --mode full --n-samples 5")
        return

    # Load and analyze
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(output_dir / "results.csv")
    analyzer.load_detailed_results(output_dir)

    print(f"\nLoaded {len(analyzer.results_df)} experiments")

    # Extract source patterns
    patterns = analyzer.extract_source_selection_patterns()
    print(f"Source pattern matrix: {patterns.shape}")

    # Train SAE
    print("\nTraining Sparse Autoencoder...")
    analyzer.train_sae(latent_dim=32)

    latent = analyzer.get_latent_representations()
    print(f"Latent representations: {latent.shape}")

    # Compute quality metrics
    print("\nComputing thesis quality metrics...")
    quality_df = analyzer.compute_thesis_quality_metrics()
    print("\nQuality statistics:")
    print(quality_df[['novelty_combined', 'complexity', 'quality_score']].describe())


def example_5_custom_source():
    """Example 5: Add custom primary sources."""
    print("\n" + "="*60)
    print("Example 5: Custom Primary Sources")
    print("="*60)

    library = SourceLibrary()
    library.initialize_embedding_model()
    library.create_index()

    # Create custom sources
    custom_sources = [
        PrimarySource(
            source_id="custom_001",
            title="Letter from Factory Worker, 1890",
            content="""
            Dear Sir,
            I write to protest the recent reduction in wages at the textile mill.
            We workers can scarcely afford to feed our families on the current pay,
            and a further reduction would bring us to the brink of starvation.
            The owners claim the market demands such measures, but we see their
            fine carriages and lavish homes while we struggle for bread.
            """,
            source_type="text",
            metadata={
                "year": 1890,
                "type": "correspondence",
                "location": "Manchester, England"
            }
        ),
        PrimarySource(
            source_id="custom_002",
            title="Newspaper Report: Strike at Steel Works, 1892",
            content="""
            Today marked the third week of the steel workers' strike at the
            Homestead plant. The conflict between capital and labor has reached
            a fever pitch, with both sides refusing to compromise. Management
            has brought in replacement workers, leading to violent clashes.
            The governor has dispatched the state militia to restore order.
            """,
            source_type="text",
            metadata={
                "year": 1892,
                "type": "newspaper",
                "location": "Pittsburgh, PA"
            }
        )
    ]

    library.add_sources_batch(custom_sources)
    print(f"\nAdded {len(custom_sources)} custom sources")

    # Test search
    results = library.search("labor strikes and worker protests", k=2)
    print("\nSearch results:")
    for source, score in results:
        print(f"  - {source.title} (score: {score:.3f})")

    return library


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" Multi-Agent Historian System - Example Usage")
    print("="*70)

    try:
        # Run examples
        example_1_create_personas()
        example_2_source_library()
        example_3_single_experiment()
        example_5_custom_source()
        example_4_analyze_results()

        print("\n" + "="*70)
        print(" Examples Complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Set up your .env file with API keys")
        print("  2. Run: python run_experiment.py --mode full --n-samples 5")
        print("  3. Check outputs/ directory for results")
        print("  4. Explore the analysis report\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
