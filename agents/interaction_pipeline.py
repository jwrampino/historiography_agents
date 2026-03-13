"""
Interaction Pipeline: Orchestrates the two-stage multi-agent process.
Stage 1: Individual historian proposals
Stage 2: Final synthesis
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from agents.historian_manager import HistorianPersona
from agents.source_retrieval import SourceRetriever
from agents.agent_llm import AgentLLM

logger = logging.getLogger(__name__)


@dataclass
class TriadExperimentResult:
    """Container for complete triad experiment results."""
    triad_id: int

    # Historians
    historians: Tuple[HistorianPersona, HistorianPersona, HistorianPersona]
    historian_names: List[str]

    # Triangle geometry
    geometry: Dict

    # Retrieval
    retrieval_query: str
    source_packets: List[Dict]  # 3 packets, one per historian

    # Stage 1: Individual proposals
    proposals: List[Dict]  # 3 proposals

    # Stage 2: Synthesis
    synthesis: Dict

    # Success flag
    success: bool
    error: str = ""


class InteractionPipeline:
    """Manages two-stage historian interaction."""

    def __init__(
        self,
        source_retriever: SourceRetriever,
        agent_llm: AgentLLM
    ):
        """
        Initialize pipeline.

        Args:
            source_retriever: SourceRetriever instance
            agent_llm: AgentLLM instance
        """
        self.retriever = source_retriever
        self.llm = agent_llm

    def run_triad_experiment(
        self,
        triad_id: int,
        historians: Tuple[HistorianPersona, HistorianPersona, HistorianPersona],
        geometry: Dict,
        n_text_sources: int = 3,
        n_image_sources: int = 2
    ) -> TriadExperimentResult:
        """
        Run complete experiment for one triad.

        Args:
            triad_id: Unique triad identifier
            historians: Tuple of 3 HistorianPersona objects
            geometry: Triangle geometry dict
            n_text_sources: Number of text sources per historian
            n_image_sources: Number of image sources per historian

        Returns:
            TriadExperimentResult
        """
        historian_names = [h.name for h in historians]

        logger.info(
            f"Starting triad {triad_id}: "
            f"{historian_names[0]}, {historian_names[1]}, {historian_names[2]}"
        )

        # Set triad context for LLM checkpoint logging
        if hasattr(self.llm, 'set_context'):
            self.llm._current_triad_id = triad_id

        try:
            # Stage 0: Retrieve sources for each historian
            source_packets = self._retrieve_sources(
                historians, n_text_sources, n_image_sources
            )

            # Use same query for all (derived from first historian)
            retrieval_query = source_packets[0]['query']

            # Stage 1: Individual proposals
            proposals = self._generate_individual_proposals(
                historians, source_packets
            )

            # Stage 2: Synthesis
            synthesis = self._generate_synthesis(historian_names, proposals)

            result = TriadExperimentResult(
                triad_id=triad_id,
                historians=historians,
                historian_names=historian_names,
                geometry=geometry,
                retrieval_query=retrieval_query,
                source_packets=source_packets,
                proposals=proposals,
                synthesis=synthesis,
                success=True
            )

            logger.info(f"OK Triad {triad_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"✗ Triad {triad_id} failed: {e}", exc_info=True)
            return TriadExperimentResult(
                triad_id=triad_id,
                historians=historians,
                historian_names=historian_names,
                geometry=geometry,
                retrieval_query="",
                source_packets=[],
                proposals=[],
                synthesis={},
                success=False,
                error=str(e)
            )

    def _retrieve_sources(
        self,
        historians: Tuple[HistorianPersona, HistorianPersona, HistorianPersona],
        n_text: int,
        n_images: int
    ) -> List[Dict]:
        """
        Retrieve source packets for each historian.

        Args:
            historians: Tuple of 3 historians
            n_text: Number of text sources
            n_images: Number of image sources

        Returns:
            List of 3 source packet dicts
        """
        logger.info("Retrieving sources for historians...")

        source_packets = []
        for i, historian in enumerate(historians):
            logger.info(f"  Retrieving for {historian.name}...")
            packet = self.retriever.retrieve_source_packet(
                historian.papers,
                n_text=n_text,
                n_images=n_images
            )
            source_packets.append(packet)

        return source_packets

    def _generate_individual_proposals(
        self,
        historians: Tuple[HistorianPersona, HistorianPersona, HistorianPersona],
        source_packets: List[Dict]
    ) -> List[Dict]:
        """
        Generate individual proposals from each historian.

        Args:
            historians: Tuple of 3 historians
            source_packets: List of 3 source packets

        Returns:
            List of 3 proposal dicts
        """
        logger.info("Generating individual proposals...")

        proposals = []
        for i, (historian, packet) in enumerate(zip(historians, source_packets)):
            logger.info(f"  Generating proposal from {historian.name}...")

            # Format sources for prompt
            sources_text = self.retriever.format_sources_for_agent(packet)

            # Extract image paths if available
            image_paths = [
                img['local_path'] for img in packet['image_sources']
                if img.get('local_path')
            ]

            # Set context for checkpoint logging
            if hasattr(self.llm, 'set_context'):
                # We'll set triad_id later in run_triad_experiment
                self.llm.set_context(
                    triad_id=getattr(self.llm, '_current_triad_id', 0),
                    historian_name=historian.name,
                    historian_position=i + 1
                )

            # Generate proposal (with images if available)
            proposal = self.llm.generate_individual_proposal(
                historian_prompt=historian.prompt,
                source_packet_text=sources_text,
                image_paths=image_paths if image_paths else None,
                temperature=0.7
            )

            proposals.append(proposal)

        logger.info(f"Generated {len(proposals)} individual proposals")
        return proposals

    def _generate_synthesis(
        self,
        historian_names: List[str],
        proposals: List[Dict]
    ) -> Dict:
        """
        Generate final synthesis from individual proposals.

        Args:
            historian_names: List of 3 historian names
            proposals: List of 3 proposal dicts

        Returns:
            Synthesis dict
        """
        logger.info("Generating synthesis...")

        synthesis = self.llm.generate_synthesis(
            historian_names=historian_names,
            proposals=proposals,
            temperature=0.5
        )

        logger.info("Synthesis generated")
        return synthesis
