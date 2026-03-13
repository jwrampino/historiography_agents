"""
Multi-Agent Dialogue System using LangGraph.
Enables historians to collaboratively develop research questions through
inductive and deductive reasoning, with access to primary sources.
"""

from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import yaml

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from personas.historian_manager import HistorianPersona
from sources.source_library import SourceLibrary, PrimarySource


ActionType = Literal["speak", "search", "read", "propose", "critique", "conclude"]


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    agent_id: str
    action_type: ActionType
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DialogueState:
    """State of the multi-agent dialogue."""
    experiment_id: str
    agents: List['HistorianAgent']
    messages: List[AgentAction] = field(default_factory=list)
    sources_accessed: List[PrimarySource] = field(default_factory=list)
    proposed_questions: List[Dict] = field(default_factory=list)
    proposed_abstracts: List[Dict] = field(default_factory=list)
    final_question: Optional[str] = None
    final_abstract: Optional[str] = None
    consensus_reached: bool = False
    turn_count: int = 0
    max_turns: int = 50


class HistorianAgent:
    """An agent representing a historian persona."""

    def __init__(
        self,
        persona: HistorianPersona,
        llm_config: Dict,
        source_library: SourceLibrary,
        experiment_id: str
    ):
        """Initialize historian agent."""
        self.persona = persona
        self.agent_id = persona.historian_id
        self.name = persona.name
        self.source_library = source_library
        self.experiment_id = experiment_id

        # Initialize LLM
        provider = llm_config.get('provider', 'anthropic')
        if provider == 'anthropic':
            self.llm = ChatAnthropic(
                model=llm_config.get('model', 'claude-sonnet-4-6'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 4000)
            )
        else:  # openai
            self.llm = ChatOpenAI(
                model=llm_config.get('model', 'gpt-4'),
                temperature=llm_config.get('temperature', 0.7)
            )

        # Agent's internal state
        self.sources_read: List[PrimarySource] = []
        self.working_thesis: Optional[str] = None

    def get_system_prompt(self) -> str:
        """Generate system prompt based on persona."""
        # Use the historian's actual prompt from their papers
        base_prompt = self.persona.prompt

        # Add collaboration instructions
        collaboration_instructions = """

You are collaborating with other historians to develop a novel research question and abstract.
Your goal is to:
1. Search for and analyze primary sources from the shared library
2. Engage in dialogue with your colleagues to explore ideas
3. Mix inductive reasoning (letting patterns emerge from sources) and deductive reasoning (testing hypotheses)
4. Contribute to developing a compelling, novel research question and abstract

Available actions:
- SPEAK: Share thoughts, arguments, or observations with the group
- SEARCH: Search the primary source library (provide search query)
- READ: Read a specific source in detail (provide source_id)
- PROPOSE: Propose a research question or thesis statement
- CRITIQUE: Offer constructive criticism of another's proposal
- CONCLUDE: Signal agreement with a final research question and abstract

Be intellectually curious, rigorous, and open to collaboration while maintaining your scholarly perspective."""

        return base_prompt + collaboration_instructions

    def decide_action(self, dialogue_state: DialogueState) -> AgentAction:
        """
        Decide what action to take based on current dialogue state.
        This is the agent's main decision-making function.
        """
        # Build context from recent dialogue
        recent_messages = dialogue_state.messages[-10:]  # Last 10 messages
        context = self._build_context(recent_messages, dialogue_state)

        # Construct prompt
        prompt = f"""Based on the current state of the collaborative research session:

{context}

What would you like to do next? Choose one action and provide the necessary details.

Respond in JSON format:
{{
    "action": "speak|search|read|propose|critique|conclude",
    "content": "your response or query here",
    "reasoning": "brief explanation of why you chose this action"
}}"""

        # Get LLM response
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Parse response
        try:
            response_data = json.loads(response.content)
            action_type = response_data['action']
            content = response_data['content']
            reasoning = response_data.get('reasoning', '')

            return AgentAction(
                agent_id=self.agent_id,
                action_type=action_type,
                content=content,
                metadata={'reasoning': reasoning}
            )
        except:
            # Fallback: treat as speak action
            return AgentAction(
                agent_id=self.agent_id,
                action_type="speak",
                content=response.content,
                metadata={'parse_error': True}
            )

    def execute_action(self, action: AgentAction, dialogue_state: DialogueState) -> Dict:
        """
        Execute the chosen action and return results.
        """
        result = {
            'action': action.to_dict(),
            'success': True,
            'output': None
        }

        if action.action_type == "speak":
            result['output'] = action.content

        elif action.action_type == "search":
            # Search source library
            query = action.content
            search_results = self.source_library.search(
                query=query,
                k=5,
                agent_id=self.agent_id,
                experiment_id=self.experiment_id
            )

            result['output'] = {
                'query': query,
                'results': [
                    {
                        'source_id': source.source_id,
                        'title': source.title,
                        'similarity': score,
                        'preview': source.content[:200] + "..."
                    }
                    for source, score in search_results
                ]
            }

            # Track accessed sources
            for source, _ in search_results:
                if source not in dialogue_state.sources_accessed:
                    dialogue_state.sources_accessed.append(source)

        elif action.action_type == "read":
            # Read a specific source in detail
            source_id = action.content
            source = self.source_library.get_source_by_id(source_id)

            if source:
                result['output'] = {
                    'source_id': source.source_id,
                    'title': source.title,
                    'content': source.content,
                    'metadata': source.metadata
                }
                self.sources_read.append(source)

                if source not in dialogue_state.sources_accessed:
                    dialogue_state.sources_accessed.append(source)
            else:
                result['success'] = False
                result['output'] = f"Source {source_id} not found"

        elif action.action_type == "propose":
            # Propose research question or thesis
            proposal = {
                'agent_id': self.agent_id,
                'proposal': action.content,
                'timestamp': action.timestamp,
                'sources_cited': [s.source_id for s in self.sources_read]
            }

            # Determine if it's a question or abstract
            if "research question" in action.content.lower() or "?" in action.content:
                dialogue_state.proposed_questions.append(proposal)
            else:
                dialogue_state.proposed_abstracts.append(proposal)

            result['output'] = proposal

        elif action.action_type == "critique":
            result['output'] = action.content

        elif action.action_type == "conclude":
            result['output'] = action.content
            result['consensus_signal'] = True

        return result

    def _build_context(self, recent_messages: List[AgentAction], state: DialogueState) -> str:
        """Build context string from recent dialogue."""
        context_parts = []

        # Recent dialogue
        context_parts.append("RECENT DIALOGUE:")
        for msg in recent_messages[-5:]:
            context_parts.append(f"[{msg.agent_id}] {msg.action_type}: {msg.content[:200]}")

        # Proposals so far
        if state.proposed_questions:
            context_parts.append("\nPROPOSED RESEARCH QUESTIONS:")
            for prop in state.proposed_questions[-3:]:
                context_parts.append(f"- {prop['proposal'][:200]}")

        # Sources accessed
        if state.sources_accessed:
            context_parts.append(f"\nSOURCES ACCESSED: {len(state.sources_accessed)} sources")

        # Turn count
        context_parts.append(f"\nTURN: {state.turn_count}/{state.max_turns}")

        return "\n".join(context_parts)


class MultiAgentDialogueSystem:
    """Orchestrates multi-agent dialogue using LangGraph."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the dialogue system."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.source_library = SourceLibrary(config_path)
        self.llm_config = self.config['llm']
        self.dialogue_config = self.config['dialogue']

    def _load_config(self) -> Dict:
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_dialogue_graph(self, agents: List[HistorianAgent]) -> StateGraph:
        """
        Create a LangGraph state graph for multi-agent dialogue.
        """
        workflow = StateGraph(DialogueState)

        def agent_turn(state: DialogueState) -> DialogueState:
            """Execute one turn of dialogue."""
            # Round-robin: each agent gets a turn
            current_agent_idx = state.turn_count % len(state.agents)
            agent = state.agents[current_agent_idx]

            # Agent decides action
            action = agent.decide_action(state)

            # Execute action
            result = agent.execute_action(action, state)

            # Update state
            state.messages.append(action)
            state.turn_count += 1

            # Check for consensus
            if result.get('consensus_signal', False):
                state.consensus_reached = self._check_consensus(state)

            return state

        def should_continue(state: DialogueState) -> str:
            """Determine if dialogue should continue."""
            if state.consensus_reached:
                return "finalize"
            elif state.turn_count >= state.max_turns:
                return "finalize"
            else:
                return "continue"

        def finalize(state: DialogueState) -> DialogueState:
            """Finalize the research question and abstract."""
            # Use LLM to synthesize final outputs from proposals
            state.final_question = self._synthesize_question(state)
            state.final_abstract = self._synthesize_abstract(state)
            return state

        # Build graph
        workflow.add_node("agent_turn", agent_turn)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("agent_turn")

        workflow.add_conditional_edges(
            "agent_turn",
            should_continue,
            {
                "continue": "agent_turn",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run_experiment(
        self,
        personas: List[HistorianPersona],
        experiment_id: str
    ) -> DialogueState:
        """
        Run a single experiment with a group of historian personas.
        """
        # Create agents
        agents = [
            HistorianAgent(
                persona=persona,
                llm_config=self.llm_config,
                source_library=self.source_library,
                experiment_id=experiment_id
            )
            for persona in personas
        ]

        # Initialize state
        initial_state = DialogueState(
            experiment_id=experiment_id,
            agents=agents,
            max_turns=self.dialogue_config['max_turns']
        )

        # Create and run graph
        graph = self.create_dialogue_graph(agents)
        final_state = graph.invoke(initial_state)

        return final_state

    def _check_consensus(self, state: DialogueState) -> bool:
        """Check if agents have reached consensus."""
        # Simple heuristic: consensus if we have both question and abstract proposals
        # and recent messages include multiple conclude actions
        conclude_actions = [
            msg for msg in state.messages[-len(state.agents):]
            if msg.action_type == "conclude"
        ]

        has_proposals = len(state.proposed_questions) > 0 and len(state.proposed_abstracts) > 0
        has_consensus_signals = len(conclude_actions) >= len(state.agents) // 2

        return has_proposals and has_consensus_signals

    def _synthesize_question(self, state: DialogueState) -> str:
        """Synthesize final research question from proposals."""
        if not state.proposed_questions:
            return "No research question reached."

        # For now, return the most recent proposal
        # In production, use LLM to synthesize
        return state.proposed_questions[-1]['proposal']

    def _synthesize_abstract(self, state: DialogueState) -> str:
        """Synthesize final abstract from proposals."""
        if not state.proposed_abstracts:
            return "No abstract reached."

        return state.proposed_abstracts[-1]['proposal']


if __name__ == "__main__":
    # Example usage
    from personas.historian_manager import HistorianManager

    print("Initializing multi-agent system...")

    # Load personas
    historian_manager = HistorianManager()
    personas = historian_manager.load_personas("personas/historian_personas.json")[:3]

    print(f"Loaded {len(personas)} historian personas:")
    for p in personas:
        print(f"  - {p.name}")

    # Create dialogue system
    system = MultiAgentDialogueSystem()

    print(f"\nRunning experiment with {len(personas)} agents...")
    # Note: This would actually run the dialogue
    # final_state = system.run_experiment(personas, experiment_id="test_001")
