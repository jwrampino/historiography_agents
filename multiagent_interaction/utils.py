"""
Utility functions for the multi-agent experiment system.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


def load_experiment_result(experiment_id: str, output_dir: Path = Path("outputs")) -> Dict:
    """Load a single experiment result by ID."""
    result_file = output_dir / f"{experiment_id}.json"

    if not result_file.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found in {output_dir}")

    with open(result_file, 'r') as f:
        return json.load(f)


def get_experiment_ids(output_dir: Path = Path("outputs")) -> List[str]:
    """Get list of all experiment IDs in output directory."""
    result_files = output_dir.glob("exp_*.json")
    return [f.stem for f in result_files]


def export_chat_history(experiment_id: str, output_file: Optional[Path] = None):
    """Export chat history from an experiment to readable format."""
    result = load_experiment_result(experiment_id)

    if output_file is None:
        output_file = Path(f"outputs/{experiment_id}_chat.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Chat History: {experiment_id}\n")
        f.write("="*70 + "\n\n")

        f.write("Group Composition:\n")
        for agent in result['group_composition']:
            f.write(f"  - {agent['persona_id']}: ")
            f.write(f"{agent['field']}, {agent['method']}, ")
            f.write(f"{agent['era']}, {agent['theoretical_orientation']}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Dialogue:\n")
        f.write("="*70 + "\n\n")

        for i, msg in enumerate(result['chat_history'], 1):
            f.write(f"Turn {i} - {msg['timestamp']}\n")
            f.write(f"Agent: {msg['agent_id']}\n")
            f.write(f"Action: {msg['action_type']}\n")
            f.write(f"Content: {msg['content']}\n")
            f.write("-"*70 + "\n\n")

        f.write("="*70 + "\n")
        f.write("Results:\n")
        f.write("="*70 + "\n\n")
        f.write(f"Research Question:\n{result['final_question']}\n\n")
        f.write(f"Abstract:\n{result['final_abstract']}\n\n")
        f.write(f"Sources Used: {len(result['sources_accessed'])}\n")

    print(f"Chat history exported to {output_file}")


def summarize_experiments(output_dir: Path = Path("outputs")) -> pd.DataFrame:
    """Create summary table of all experiments."""
    experiment_ids = get_experiment_ids(output_dir)

    summaries = []
    for exp_id in experiment_ids:
        result = load_experiment_result(exp_id, output_dir)

        # Extract key metrics
        summary = {
            'experiment_id': exp_id,
            'turns': result['turn_count'],
            'consensus': result['consensus_reached'],
            'sources_used': len(result['sources_accessed']),
            'question_length': len(result['final_question']),
            'abstract_length': len(result['final_abstract']),
        }

        # Add persona info
        for i, agent in enumerate(result['group_composition'][:3]):
            summary[f'agent_{i}_field'] = agent['field']
            summary[f'agent_{i}_method'] = agent['method']

        summaries.append(summary)

    df = pd.DataFrame(summaries)
    print(f"Loaded {len(df)} experiments")

    return df


def compare_experiments(exp_id1: str, exp_id2: str):
    """Compare two experiments side by side."""
    result1 = load_experiment_result(exp_id1)
    result2 = load_experiment_result(exp_id2)

    print("="*70)
    print(f"Comparing {exp_id1} vs {exp_id2}")
    print("="*70)

    # Group composition
    print("\nGroup 1:")
    for agent in result1['group_composition']:
        print(f"  {agent['field']} / {agent['theoretical_orientation']}")

    print("\nGroup 2:")
    for agent in result2['group_composition']:
        print(f"  {agent['field']} / {agent['theoretical_orientation']}")

    # Metrics
    print("\nMetrics:")
    print(f"  Turns: {result1['turn_count']} vs {result2['turn_count']}")
    print(f"  Sources: {len(result1['sources_accessed'])} vs {len(result2['sources_accessed'])}")
    print(f"  Consensus: {result1['consensus_reached']} vs {result2['consensus_reached']}")

    # Questions
    print(f"\nQuestion 1:\n{result1['final_question']}")
    print(f"\nQuestion 2:\n{result2['final_question']}")


def find_experiments_by_persona(
    field: Optional[str] = None,
    method: Optional[str] = None,
    orientation: Optional[str] = None,
    output_dir: Path = Path("outputs")
) -> List[str]:
    """Find experiments containing personas with specific characteristics."""
    experiment_ids = get_experiment_ids(output_dir)
    matching = []

    for exp_id in experiment_ids:
        result = load_experiment_result(exp_id, output_dir)

        for agent in result['group_composition']:
            match = True
            if field and agent['field'] != field:
                match = False
            if method and agent['method'] != method:
                match = False
            if orientation and agent['theoretical_orientation'] != orientation:
                match = False

            if match:
                matching.append(exp_id)
                break

    print(f"Found {len(matching)} matching experiments")
    return matching


def analyze_source_usage(output_dir: Path = Path("outputs")) -> pd.DataFrame:
    """Analyze which sources are most commonly used."""
    experiment_ids = get_experiment_ids(output_dir)

    source_counts = {}
    for exp_id in experiment_ids:
        result = load_experiment_result(exp_id, output_dir)

        for source in result['sources_accessed']:
            source_id = source['source_id']
            if source_id not in source_counts:
                source_counts[source_id] = {
                    'count': 0,
                    'title': source['title'],
                    'type': source['source_type']
                }
            source_counts[source_id]['count'] += 1

    df = pd.DataFrame(source_counts.values())
    df = df.sort_values('count', ascending=False)

    print(f"Analyzed {len(df)} unique sources across {len(experiment_ids)} experiments")
    print("\nTop 10 most used sources:")
    print(df.head(10))

    return df


def check_system_status():
    """Check if all system components are properly set up."""
    print("="*70)
    print("System Status Check")
    print("="*70)

    status = {
        'personas': False,
        'sources': False,
        'outputs': False,
        'api_key': False
    }

    # Check personas
    persona_file = Path("personas/persona_storage.json")
    if persona_file.exists():
        with open(persona_file, 'r') as f:
            data = json.load(f)
            n_personas = data.get('total_count', 0)
            print(f"✓ Personas: {n_personas} personas stored")
            status['personas'] = True
    else:
        print("✗ Personas: Not set up (run: python run_experiment.py --mode setup-only)")

    # Check source library
    source_index = Path("sources/faiss_index/faiss.index")
    if source_index.exists():
        print(f"✓ Source Library: Index found at {source_index}")
        status['sources'] = True
    else:
        print("✗ Source Library: Not set up (run: python run_experiment.py --mode setup-only)")

    # Check outputs
    output_dir = Path("outputs")
    if output_dir.exists():
        n_results = len(list(output_dir.glob("exp_*.json")))
        print(f"✓ Outputs: {n_results} experiment results found")
        status['outputs'] = n_results > 0
    else:
        print("✗ Outputs: No experiments run yet")

    # Check API key
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()

        if os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY'):
            print("✓ API Key: Configured")
            status['api_key'] = True
        else:
            print("✗ API Key: Not configured (create .env file)")
    except:
        print("✗ API Key: Cannot check (install python-dotenv)")

    print("\n" + "="*70)

    if all(status.values()):
        print("✓ System ready to run experiments!")
    else:
        print("⚠ Some components need setup")

    return status


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            check_system_status()

        elif command == "summary":
            df = summarize_experiments()
            print("\n", df)

        elif command == "sources":
            df = analyze_source_usage()

        elif command == "export" and len(sys.argv) > 2:
            experiment_id = sys.argv[2]
            export_chat_history(experiment_id)

        elif command == "compare" and len(sys.argv) > 3:
            compare_experiments(sys.argv[2], sys.argv[3])

        else:
            print("Unknown command. Available commands:")
            print("  python utils.py status")
            print("  python utils.py summary")
            print("  python utils.py sources")
            print("  python utils.py export <exp_id>")
            print("  python utils.py compare <exp_id1> <exp_id2>")
    else:
        check_system_status()
