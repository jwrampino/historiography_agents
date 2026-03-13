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
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from personas.historian_manager import HistorianManager, HistorianPersona
from sources.source_library import SourceLibrary
from agents.multi_agent_system import MultiAgentDialogueSystem, DialogueState


@dataclass
class ExperimentResult:
    """Stores results from a single experiment run."""
    experiment_id: str
    group_composition: List[Dict]  # Persona info for each agent
    triangle_geometry: Dict  # Geometric features of the historian triangle
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
        self.historian_manager = HistorianManager()
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

    def setup_personas(self, strategy: str = "filtered", n_samples: Optional[int] = None):
        """
        Set up personas for experiments.

        Args:
            strategy: "filtered", "stratified", or "random"
            n_samples: Number of samples for sampling strategies
        """
        print("Setting up historian personas...")

        # Load historian personas
        personas = self.historian_manager.load_personas("personas/historian_personas.json")
        print(f"Loaded {len(personas)} historians")

        # Ensure embeddings are computed
        if personas[0].embedding is None:
            print("Computing embeddings...")
            self.historian_manager.compute_historian_embeddings()

        # Sample groups with geometry constraints
        if n_samples is None:
            n_samples = 100  # Default

        self.groups = self.historian_manager.sample_groups(
            n_groups=n_samples,
            strategy=strategy,
            min_distance=0.1,
            max_distance=0.7,
            min_area=0.001
        )

        print(f"Generated {len(self.groups)} groups using '{strategy}' strategy")

    def setup_source_library(self, library_path: Optional[str] = None, use_historian_pipeline: bool = True):
        """
        Set up the source library.

        Args:
            library_path: Path to pre-existing library, or None to use historian_pipeline data
            use_historian_pipeline: If True, load from historian_pipeline data structure (default)
        """
        print("Setting up source library...")

        if use_historian_pipeline:
            # Load from historian_pipeline data
            data_dir = Path(__file__).parent.parent.parent / "data"
            if library_path:
                data_dir = Path(library_path)
            print(f"Loading from historian_pipeline data at {data_dir}")
            self.source_library.load_from_historian_pipeline(data_dir)
        elif library_path and Path(library_path).exists():
            self.source_library.load_index(Path(library_path))
        else:
            print("Creating example source library...")
            from sources.source_library import create_example_library
            self.source_library = create_example_library(n_sources=200)
            self.dialogue_system.source_library = self.source_library

        index_size = len(self.source_library.index_to_source_id) if hasattr(self.source_library, 'index_to_source_id') else self.source_library.index.ntotal if self.source_library.index else 0
        print(f"Source library ready with {index_size} sources in index")

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
        print(f"Group composition: {[p.name for p in group]}")

        # Compute triangle geometry
        triangle_geometry = self.historian_manager.compute_triangle_geometry(group)
        print(f"Triangle geometry: perimeter={triangle_geometry['perimeter']:.3f}, area={triangle_geometry['area']:.3f}")

        # Run dialogue
        final_state = self.dialogue_system.run_experiment(
            personas=list(group),
            experiment_id=experiment_id
        )

        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            group_composition=[p.to_dict() for p in group],
            triangle_geometry=triangle_geometry,
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
            # Extract historian names
            historian_names = [p['name'] for p in result.group_composition]

            record = {
                'experiment_id': result.experiment_id,
                'historian_1': historian_names[0] if len(historian_names) > 0 else None,
                'historian_2': historian_names[1] if len(historian_names) > 1 else None,
                'historian_3': historian_names[2] if len(historian_names) > 2 else None,
                'turn_count': result.turn_count,
                'consensus_reached': result.consensus_reached,
                'n_sources_used': len(result.sources_accessed),
                'question_length': len(result.final_question),
                'abstract_length': len(result.final_abstract),
            }

            # Add triangle geometry features
            geom = result.triangle_geometry
            record.update({
                'geom_side_1': geom['side_1'],
                'geom_side_2': geom['side_2'],
                'geom_side_3': geom['side_3'],
                'geom_perimeter': geom['perimeter'],
                'geom_area': geom['area'],
                'geom_min_angle': geom['min_angle'],
                'geom_max_angle': geom['max_angle'],
                'geom_angle_variance': geom['angle_variance'],
            })

            records.append(record)

        return pd.DataFrame(records)


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent historian experiments")
    parser.add_argument(
        '--strategy',
        choices=['filtered', 'stratified', 'random'],
        default='filtered',
        help='Sampling strategy for historian groups'
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
