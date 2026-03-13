"""
Agent LLM Interface: Handles communication with OpenAI GPT-4o.
Used for generating individual proposals and synthesizing group abstracts.
"""

import os
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class AgentLLM:
    """Interface to OpenAI GPT-4o for historian agent reasoning."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", storage=None):
        """
        Initialize LLM interface.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (default: gpt-4o)
            storage: ExperimentStorage instance for checkpoint logging (optional)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.storage = storage
        self._client = None
        self._current_triad_id = None
        self._current_historian_name = None
        self._current_historian_position = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client with model {self.model}")
            except ImportError:
                raise RuntimeError(
                    "openai package required: pip install openai"
                )
        return self._client

    def set_context(
        self, triad_id: int, historian_name: str = "", historian_position: int = 0
    ):
        """Set context for checkpoint logging."""
        self._current_triad_id = triad_id
        self._current_historian_name = historian_name
        self._current_historian_position = historian_position

    def generate_individual_proposal(
        self,
        historian_prompt: str,
        source_packet_text: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Generate an individual historian's proposal.

        Args:
            historian_prompt: Full persona prompt for the historian
            source_packet_text: Formatted sources text
            temperature: Sampling temperature
            max_retries: Number of retry attempts

        Returns:
            Dict with keys: research_question, abstract, selected_sources
        """
        system_message = historian_prompt

        user_message = f"""You have been provided with the following primary sources:

{source_packet_text}

Based on these sources and your scholarly perspective, please:

1. Propose ONE focused research question that you find most compelling
2. Write a short abstract (3-4 sentences) outlining your interpretation
3. Identify the TWO most useful sources for your interpretation (by number)

Format your response exactly as:

RESEARCH QUESTION:
[your question here]

ABSTRACT:
[your 3-4 sentence abstract here]

SELECTED SOURCES:
[list the source numbers, e.g., "Source 1, Image 2"]"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=800
                )

                content = response.choices[0].message.content
                parsed = self._parse_individual_proposal(content)

                # Checkpoint: Log LLM interaction immediately
                if self.storage and self._current_triad_id is not None:
                    self.storage.insert_llm_interaction(
                        triad_id=self._current_triad_id,
                        interaction_type='individual_proposal',
                        historian_name=self._current_historian_name,
                        historian_position=self._current_historian_position,
                        system_prompt=system_message,
                        user_prompt=user_message,
                        llm_response=content,
                        model_name=self.model,
                        temperature=temperature
                    )

                logger.info(f"Generated individual proposal (attempt {attempt + 1})")
                return parsed

            except Exception as e:
                logger.warning(f"Proposal generation failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def generate_synthesis(
        self,
        historian_names: List[str],
        proposals: List[Dict[str, str]],
        temperature: float = 0.5,
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Generate a synthesized group abstract from individual proposals.

        Args:
            historian_names: Names of the three historians
            proposals: List of 3 proposal dicts
            temperature: Sampling temperature
            max_retries: Number of retry attempts

        Returns:
            Dict with keys: final_research_question, final_abstract, final_sources
        """
        # Build prompt with all three proposals
        proposals_text = []
        for name, prop in zip(historian_names, proposals):
            proposals_text.append(f"""
=== {name} ===
Research Question: {prop['research_question']}

Abstract:
{prop['abstract']}

Sources Used: {prop['selected_sources']}
""")

        system_message = """You are a skilled historical editor facilitating collaboration between historians with different perspectives. Your goal is to synthesize their individual proposals into a coherent shared interpretation."""

        user_message = f"""Three historians have each proposed research questions and abstracts based on the same source materials:

{''.join(proposals_text)}

Please produce:

1. A SHARED RESEARCH QUESTION that bridges their perspectives
2. A FINAL MERGED ABSTRACT (4-5 sentences) that:
   - Identifies key agreements across proposals
   - Acknowledges productive tensions or disagreements
   - Proposes a synthetic interpretation
3. SELECT 3-5 SOURCES from those mentioned that best support the merged interpretation

Format your response exactly as:

FINAL RESEARCH QUESTION:
[shared question here]

FINAL ABSTRACT:
[4-5 sentence synthesis here]

FINAL SOURCES:
[list selected sources]"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )

                content = response.choices[0].message.content
                parsed = self._parse_synthesis(content)

                # Checkpoint: Log LLM interaction immediately
                if self.storage and self._current_triad_id is not None:
                    self.storage.insert_llm_interaction(
                        triad_id=self._current_triad_id,
                        interaction_type='synthesis',
                        historian_name='group_synthesis',
                        historian_position=0,
                        system_prompt=system_message,
                        user_prompt=user_message,
                        llm_response=content,
                        model_name=self.model,
                        temperature=temperature
                    )

                logger.info(f"Generated synthesis (attempt {attempt + 1})")
                return parsed

            except Exception as e:
                logger.warning(f"Synthesis generation failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _parse_individual_proposal(self, content: str) -> Dict[str, str]:
        """Parse the LLM output for individual proposal."""
        result = {
            'research_question': '',
            'abstract': '',
            'selected_sources': ''
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line_upper = line.strip().upper()

            if 'RESEARCH QUESTION' in line_upper:
                current_section = 'research_question'
                # Check if content on same line
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue
            elif 'ABSTRACT' in line_upper and 'FINAL' not in line_upper:
                current_section = 'abstract'
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue
            elif 'SELECTED SOURCES' in line_upper or 'SOURCES USED' in line_upper:
                current_section = 'selected_sources'
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue

            # Add content to current section
            if current_section and line.strip():
                if result[current_section]:
                    result[current_section] += ' ' + line.strip()
                else:
                    result[current_section] = line.strip()

        return result

    def _parse_synthesis(self, content: str) -> Dict[str, str]:
        """Parse the LLM output for synthesis."""
        result = {
            'final_research_question': '',
            'final_abstract': '',
            'final_sources': ''
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line_upper = line.strip().upper()

            if 'FINAL RESEARCH QUESTION' in line_upper or \
               ('SHARED' in line_upper and 'QUESTION' in line_upper):
                current_section = 'final_research_question'
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue
            elif 'FINAL ABSTRACT' in line_upper or \
                 ('MERGED' in line_upper and 'ABSTRACT' in line_upper):
                current_section = 'final_abstract'
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue
            elif 'FINAL SOURCES' in line_upper or \
                 ('SELECTED SOURCES' in line_upper and 'FINAL' in line_upper):
                current_section = 'final_sources'
                if ':' in line:
                    content_part = line.split(':', 1)[1].strip()
                    if content_part:
                        result[current_section] = content_part
                continue

            # Add content to current section
            if current_section and line.strip():
                if result[current_section]:
                    result[current_section] += ' ' + line.strip()
                else:
                    result[current_section] = line.strip()

        return result
