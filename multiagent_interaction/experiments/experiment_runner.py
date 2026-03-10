"""
Experiment Runner: Orchestrates factorial design experiments.
Runs multiple combinations of historian personas and collects outputs.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import yaml

import sys
sys.path.append('..')
from personas.persona_manager import PersonaManager, HistorianPersona
from sources.source_library import SourceLibrary
from agents.multi_agent_system import MultiAgentDialogueSystem, DialogueState


@dataclass
class ExperimentResult:
    """Stores results from a single experiment run."""
    experiment_id: str
    group_composition: List[Dict]  # Persona info for each agent
    chat_history: List[Dict]  # All agent actions
    sources_accessed: List[Dict]  # Sources used
    final_question: str
    final_abstract: str
    turn_count: int
    consensus_reached: bool
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, output_dir: Path):
        """Save experiment result to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.experiment_id}.json"

        with open(output_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """Manages and executes factorial design experiments."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment runner."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.persona_manager = PersonaManager(config_path)
        self.dialogue_system = MultiAgentDialogueSystem(config_path)
        self.source_library = self.dialogue_system.source_library

        # Experiment tracking
        self.results: List[ExperimentResult] = []
        self.experiment_count = 0

        # Output directory
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict:
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_personas(self, strategy: str = "full", n_samples: Optional[int] = None):
        """
        Set up personas for experiments.

        Args:
            strategy: "full" for full factorial, "stratified" for sampling
            n_samples: Number of samples for stratified strategy
        """
        print("Setting up personas...")

        # Load or create personas
        self.persona_manager.load_personas_from_storage()

        if strategy == "full":
            # Full factorial: all possible groups
            group_size = self.config['experiment']['n_agents_per_group']
            self.groups = self.persona_manager.generate_factorial_groups(group_size)
            print(f"Generated {len(self.groups)} groups (full factorial)")

        elif strategy == "stratified":
            # Stratified sampling
            if n_samples is None:
                n_samples = 100  # Default

            group_size = self.config['experiment']['n_agents_per_group']
            self.groups = self.persona_manager.generate_stratified_groups(
                group_size=group_size,
                n_samples=n_samples,
                ensure_diversity=True
            )
            print(f"Generated {len(self.groups)} groups (stratified sampling)")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def setup_source_library(self, library_path: Optional[str] = None):
        """
        Set up the source library.

        Args:
            library_path: Path to pre-existing library, or None to create example
        """
        print("Setting up source library...")

        if library_path and Path(library_path).exists():
            self.source_library.load_index(Path(library_path))
        else:
            print("Creating example source library...")
            from sources.source_library import create_example_library
            self.source_library = create_example_library(n_sources=200)
            self.dialogue_system.source_library = self.source_library

        print(f"Source library ready with {len(self.source_library.sources)} sources")

    def run_single_experiment(
        self,
        group: Tuple[HistorianPersona, ...],
        experiment_id: Optional[str] = None
    ) -> ExperimentResult:
        """
        Run a single experiment with a group of personas.

        Args:
            group: Tuple of HistorianPersona objects
            experiment_id: Optional custom experiment ID

        Returns:
            ExperimentResult object
        """
        if experiment_id is None:
            experiment_id = f"exp_{self.experiment_count:05d}"
            self.experiment_count += 1

        print(f"\nRunning experiment: {experiment_id}")
        print(f"Group composition: {[p.persona_id for p in group]}")

        # Run dialogue
        final_state = self.dialogue_system.run_experiment(
            personas=list(group),
            experiment_id=experiment_id
        )

        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            group_composition=[p.to_dict() for p in group],
            chat_history=[msg.to_dict() for msg in final_state.messages],
            sources_accessed=[s.to_dict() for s in final_state.sources_accessed],
            final_question=final_state.final_question or "",
            final_abstract=final_state.final_abstract or "",
            turn_count=final_state.turn_count,
            consensus_reached=final_state.consensus_reached,
            timestamp=datetime.now().isoformat()
        )

        # Save result
        result.save(self.output_dir)

        # Store in memory
        self.results.append(result)

        return result

    def run_all_experiments(
        self,
        max_experiments: Optional[int] = None,
        start_from: int = 0
    ):
        """
        Run all experiments in the factorial design.

        Args:
            max_experiments: Maximum number of experiments to run (None for all)
            start_from: Index to start from (for resuming)
        """
        if not hasattr(self, 'groups'):
            raise RuntimeError("Must call setup_personas() before running experiments")

        experiments_to_run = self.groups[start_from:]
        if max_experiments:
            experiments_to_run = experiments_to_run[:max_experiments]

        print(f"\n{'='*60}")
        print(f"Starting experiment run: {len(experiments_to_run)} experiments")
        print(f"{'='*60}\n")

        for i, group in enumerate(tqdm(experiments_to_run, desc="Running experiments")):
            experiment_id = f"exp_{start_from + i:05d}"

            try:
                result = self.run_single_experiment(group, experiment_id)
                print(f"✓ {experiment_id} completed ({result.turn_count} turns)")

            except Exception as e:
                print(f"✗ {experiment_id} failed: {str(e)}")
                # Log error but continue
                self._log_error(experiment_id, group, str(e))

        print(f"\n{'='*60}")
        print(f"Experiment run complete: {len(self.results)} successful")
        print(f"{'='*60}\n")

        # Save summary
        self.save_experiment_summary()

    def save_experiment_summary(self):
        """Save summary of all experiments."""
        summary = {
            'total_experiments': len(self.results),
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'experiments': [
                {
                    'experiment_id': r.experiment_id,
                    'group': [p['persona_id'] for p in r.group_composition],
                    'turns': r.turn_count,
                    'consensus': r.consensus_reached,
                    'sources_used': len(r.sources_accessed)
                }
                for r in self.results
            ]
        }

        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved experiment summary to {summary_file}")

    def _log_error(self, experiment_id: str, group: Tuple[HistorianPersona, ...], error: str):
        """Log experiment errors."""
        error_log = self.output_dir / "errors.json"

        error_entry = {
            'experiment_id': experiment_id,
            'group': [p.persona_id for p in group],
            'error': error,
            'timestamp': datetime.now().isoformat()
        }

        # Append to error log
        errors = []
        if error_log.exists():
            with open(error_log, 'r') as f:
                errors = json.load(f)

        errors.append(error_entry)

        with open(error_log, 'w') as f:
            json.dump(errors, f, indent=2)

    def load_previous_results(self, output_dir: Optional[Path] = None):
        """Load results from previous experiment runs."""
        if output_dir is None:
            output_dir = self.output_dir

        result_files = list(output_dir.glob("exp_*.json"))
        print(f"Loading {len(result_files)} previous results...")

        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                result = ExperimentResult(**data)
                self.results.append(result)

        print(f"Loaded {len(self.results)} results")

    def get_results_dataframe(self):
        """Convert results to pandas DataFrame for analysis."""
        import pandas as pd

        records = []
        for result in self.results:
            # Extract persona characteristics for each agent
            persona_fields = [p['field'] for p in result.group_composition]
            persona_methods = [p['method'] for p in result.group_composition]
            persona_eras = [p['era'] for p in result.group_composition]
            persona_orientations = [p['theoretical_orientation'] for p in result.group_composition]

            record = {
                'experiment_id': result.experiment_id,
                'turn_count': result.turn_count,
                'consensus_reached': result.consensus_reached,
                'n_sources_used': len(result.sources_accessed),
                'question_length': len(result.final_question),
                'abstract_length': len(result.final_abstract),
            }

            # Add persona characteristics
            for i in range(len(result.group_composition)):
                record[f'agent_{i}_field'] = persona_fields[i]
                record[f'agent_{i}_method'] = persona_methods[i]
                record[f'agent_{i}_era'] = persona_eras[i]
                record[f'agent_{i}_orientation'] = persona_orientations[i]

            records.append(record)

        return pd.DataFrame(records)


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent historian experiments")
    parser.add_argument(
        '--strategy',
        choices=['full', 'stratified'],
        default='stratified',
        help='Sampling strategy for persona groups'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=50,
        help='Number of groups to sample (for stratified strategy)'
    )
    parser.add_argument(
        '--max-experiments',
        type=int,
        default=None,
        help='Maximum number of experiments to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(args.config)

    # Setup
    runner.setup_personas(strategy=args.strategy, n_samples=args.n_samples)
    runner.setup_source_library()

    # Run experiments
    runner.run_all_experiments(max_experiments=args.max_experiments)

    # Export results
    df = runner.get_results_dataframe()
    df.to_csv(runner.output_dir / "results.csv", index=False)
    print(f"Results saved to {runner.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
