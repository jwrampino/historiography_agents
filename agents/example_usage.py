"""
Example usage of the multi-agent system components.
Demonstrates how to use individual components programmatically.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_historian_manager():
    """Example: Load and manage historian personas."""
    from agents.historian_manager import HistorianManager

    logger.info("\n=== Example: Historian Manager ===\n")

    # Initialize manager
    manager = HistorianManager(n_historians=10)

    # Create personas
    personas = manager.create_historian_personas()
    print(f"Created {len(personas)} historian personas")

    # Show example persona
    example = personas[0]
    print(f"\nExample: {example.name}")
    print(f"Papers: {len(example.papers)}")
    print(f"Prompt length: {len(example.prompt)} characters")

    # Compute embeddings
    manager.compute_historian_embeddings()
    print(f"\nEmbedding shape: {example.embedding.shape}")

    # Generate triads
    triads = manager.sample_groups(
        n_groups=5,
        strategy="stratified",
        min_distance=0.1,
        max_distance=0.7
    )
    print(f"\nGenerated {len(triads)} triads")

    # Show example triad geometry
    if triads:
        geom = manager.compute_triangle_geometry(triads[0])
        print(f"\nExample triad geometry:")
        print(f"  Perimeter: {geom['perimeter']:.4f}")
        print(f"  Area: {geom['area']:.4f}")


def example_source_retrieval():
    """Example: Retrieve sources from corpus."""
    from agents.source_retrieval import SourceRetriever
    from agents.historian_manager import HistorianManager

    logger.info("\n=== Example: Source Retrieval ===\n")

    # Initialize
    retriever = SourceRetriever()
    manager = HistorianManager(n_historians=5)

    # Get a historian
    personas = manager.create_historian_personas()
    historian = personas[0]

    print(f"Retrieving sources for: {historian.name}")

    # Retrieve source packet
    try:
        packet = retriever.retrieve_source_packet(
            historian.papers,
            n_text=2,
            n_images=1
        )

        print(f"\nRetrieval query: '{packet['query']}'")
        print(f"Text sources: {len(packet['text_sources'])}")
        print(f"Image sources: {len(packet['image_sources'])}")

        if packet['text_sources']:
            print(f"\nExample text source:")
            src = packet['text_sources'][0]
            print(f"  Title: {src['title']}")
            print(f"  Institution: {src['institution']}")
            print(f"  Similarity: {src['similarity_score']:.3f}")

    except Exception as e:
        print(f"⚠ Retrieval failed (corpus may not be built): {e}")
        print("Run: python -m historian_pipeline.pipeline --query \"history\" --max-items 500")


def example_convergence_analysis():
    """Example: Analyze convergence."""
    from agents.convergence_analysis import ConvergenceAnalyzer
    from agents.historian_manager import HistorianManager
    import numpy as np

    logger.info("\n=== Example: Convergence Analysis ===\n")

    # Initialize
    analyzer = ConvergenceAnalyzer()
    manager = HistorianManager(n_historians=5)

    # Get historians
    personas = manager.create_historian_personas()
    manager.compute_historian_embeddings()

    # Simulate abstracts (in real use, these come from LLM)
    historian_embeddings = tuple([
        personas[0].embedding,
        personas[1].embedding,
        personas[2].embedding
    ])

    individual_abstracts = [
        "This analysis examines the role of class in historical transformation.",
        "Evidence suggests material conditions shaped social movements.",
        "The intersection of economic and political power defined this era."
    ]

    final_abstract = (
        "Synthesizing these perspectives, we see how material conditions "
        "and class dynamics intersected with political power to drive change."
    )

    # Analyze convergence
    metrics = analyzer.compute_convergence_metrics(
        historian_embeddings=historian_embeddings,
        individual_abstracts=individual_abstracts,
        final_abstract=final_abstract
    )

    print(f"Mean abstract distance: {metrics.mean_abstract_distance:.4f}")
    print(f"Final abstract distance: {metrics.distance_final_to_centroid:.4f}")
    print(f"Converged: {metrics.converged}")

    # Additional stats
    stats = analyzer.compute_embedding_stats(metrics)
    print(f"\nConvergence delta: {stats['convergence_delta']:.4f}")
    print(f"Mean pairwise similarity: {stats['mean_pairwise_abstract_similarity']:.3f}")


def example_storage():
    """Example: Store and retrieve experiment data."""
    from agents.storage import ExperimentStorage
    import pandas as pd

    logger.info("\n=== Example: Storage ===\n")

    # Initialize (in-memory for demo)
    storage = ExperimentStorage(db_path=":memory:")

    # Insert example triad
    storage.insert_triad(
        triad_id=1,
        historian_names=["Historian A", "Historian B", "Historian C"],
        historian_ids=["id1", "id2", "id3"],
        geometry={
            'side_1': 0.3,
            'side_2': 0.4,
            'side_3': 0.35,
            'perimeter': 1.05,
            'area': 0.05,
            'min_angle': 0.8,
            'max_angle': 1.2,
            'angle_variance': 0.02
        },
        retrieval_query="reconstruction era history"
    )

    # Insert proposal
    storage.insert_proposal(
        triad_id=1,
        historian_name="Historian A",
        position=1,
        proposal={
            'research_question': "How did reconstruction reshape power?",
            'abstract': "This research examines...",
            'selected_sources': "Source 1, Source 3"
        },
        n_text_sources=3,
        n_image_sources=2
    )

    # Insert convergence result
    storage.insert_convergence_result(
        triad_id=1,
        metrics={
            'distance_hist1_to_centroid': 0.2,
            'distance_hist2_to_centroid': 0.25,
            'distance_hist3_to_centroid': 0.22,
            'mean_historian_distance': 0.22,
            'distance_abstract1_to_centroid': 0.15,
            'distance_abstract2_to_centroid': 0.18,
            'distance_abstract3_to_centroid': 0.16,
            'mean_abstract_distance': 0.163,
            'distance_final_to_centroid': 0.12,
            'converged': True
        },
        additional_stats={
            'convergence_delta': 0.043,
            'abstract_distance_variance': 0.0002,
            'mean_pairwise_abstract_similarity': 0.85
        }
    )

    # Query data
    df = storage.get_convergence_data()
    print(f"Retrieved {len(df)} records")
    print(df.head())

    storage.close()


def main():
    """Run all examples."""
    print("="*60)
    print("MULTI-AGENT SYSTEM EXAMPLES")
    print("="*60)

    example_historian_manager()
    example_source_retrieval()
    example_convergence_analysis()
    example_storage()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
