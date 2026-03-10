"""
Persona Manager: Handles historian persona definitions and assignments.
Assumes personas are already fine-tuned and stored.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import combinations
import json
from pathlib import Path
import yaml


@dataclass
class HistorianPersona:
    """Represents a fine-tuned historian persona."""
    persona_id: str
    field: str  # e.g., "social_history"
    method: str  # e.g., "quantitative"
    era: str  # e.g., "modern"
    theoretical_orientation: str  # e.g., "marxist"

    # Reference to fine-tuned model or system prompt
    model_identifier: Optional[str] = None
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert persona to dictionary."""
        return asdict(self)

    def __hash__(self):
        return hash(self.persona_id)

    def __eq__(self, other):
        if isinstance(other, HistorianPersona):
            return self.persona_id == other.persona_id
        return False


class PersonaManager:
    """Manages historian personas and their assignments to groups."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize PersonaManager with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.personas: List[HistorianPersona] = []
        self.persona_storage_path = Path("personas/persona_storage.json")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_persona_grid(self) -> List[HistorianPersona]:
        """
        Create a grid of all possible historian personas.
        This represents the factorial design space.
        """
        personas = []
        persona_config = self.config['personas']

        fields = persona_config['fields']
        methods = persona_config['methods']
        eras = persona_config['eras']
        orientations = persona_config['theoretical_orientations']

        persona_id = 0
        for field in fields:
            for method in methods:
                for era in eras:
                    for orientation in orientations:
                        persona = HistorianPersona(
                            persona_id=f"historian_{persona_id:04d}",
                            field=field,
                            method=method,
                            era=era,
                            theoretical_orientation=orientation,
                            model_identifier=f"finetuned_{field}_{method}_{era}_{orientation}"
                        )
                        personas.append(persona)
                        persona_id += 1

        self.personas = personas
        return personas

    def load_personas_from_storage(self, storage_path: Optional[str] = None) -> List[HistorianPersona]:
        """
        Load pre-existing personas from storage.
        Assumes personas are already fine-tuned and stored.
        """
        if storage_path:
            self.persona_storage_path = Path(storage_path)

        if not self.persona_storage_path.exists():
            print(f"Storage path {self.persona_storage_path} not found. Creating persona grid instead.")
            return self.create_persona_grid()

        with open(self.persona_storage_path, 'r') as f:
            data = json.load(f)

        self.personas = [HistorianPersona(**p) for p in data['personas']]
        return self.personas

    def save_personas_to_storage(self, storage_path: Optional[str] = None):
        """Save personas to storage file."""
        if storage_path:
            self.persona_storage_path = Path(storage_path)

        self.persona_storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'personas': [p.to_dict() for p in self.personas],
            'total_count': len(self.personas)
        }

        with open(self.persona_storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_factorial_groups(self, group_size: int = 3) -> List[Tuple[HistorianPersona, ...]]:
        """
        Generate all possible combinations of personas in groups.
        For factorial design experiments.

        Args:
            group_size: Number of personas per group (default: 3)

        Returns:
            List of tuples, each containing a group of personas
        """
        if not self.personas:
            self.load_personas_from_storage()

        # Generate all combinations of group_size personas
        all_groups = list(combinations(self.personas, group_size))

        print(f"Generated {len(all_groups)} unique groups of {group_size} personas")
        return all_groups

    def generate_stratified_groups(
        self,
        group_size: int = 3,
        n_samples: int = 100,
        ensure_diversity: bool = True
    ) -> List[Tuple[HistorianPersona, ...]]:
        """
        Generate stratified sample of persona groups.
        Useful when full factorial is too large.

        Args:
            group_size: Number of personas per group
            n_samples: Number of groups to sample
            ensure_diversity: If True, ensure each group has diverse fields/methods

        Returns:
            List of sampled persona groups
        """
        import random

        if not self.personas:
            self.load_personas_from_storage()

        groups = []
        max_attempts = n_samples * 10

        for _ in range(max_attempts):
            if len(groups) >= n_samples:
                break

            # Sample group
            group = tuple(random.sample(self.personas, group_size))

            if ensure_diversity:
                # Check diversity: all different fields
                fields = [p.field for p in group]
                if len(set(fields)) < group_size:
                    continue

            if group not in groups:
                groups.append(group)

        print(f"Generated {len(groups)} stratified groups of {group_size} personas")
        return groups

    def get_persona_by_id(self, persona_id: str) -> Optional[HistorianPersona]:
        """Retrieve a persona by its ID."""
        for persona in self.personas:
            if persona.persona_id == persona_id:
                return persona
        return None

    def filter_personas(
        self,
        field: Optional[str] = None,
        method: Optional[str] = None,
        era: Optional[str] = None,
        orientation: Optional[str] = None
    ) -> List[HistorianPersona]:
        """Filter personas by characteristics."""
        filtered = self.personas

        if field:
            filtered = [p for p in filtered if p.field == field]
        if method:
            filtered = [p for p in filtered if p.method == method]
        if era:
            filtered = [p for p in filtered if p.era == era]
        if orientation:
            filtered = [p for p in filtered if p.theoretical_orientation == orientation]

        return filtered


if __name__ == "__main__":
    # Example usage
    manager = PersonaManager()

    # Create full persona grid
    personas = manager.create_persona_grid()
    print(f"Created {len(personas)} personas")

    # Save to storage
    manager.save_personas_to_storage()

    # Generate groups for factorial design
    groups = manager.generate_stratified_groups(group_size=3, n_samples=50)
    print(f"\nExample group:")
    for persona in groups[0]:
        print(f"  - {persona.persona_id}: {persona.field}, {persona.method}, {persona.era}, {persona.theoretical_orientation}")
