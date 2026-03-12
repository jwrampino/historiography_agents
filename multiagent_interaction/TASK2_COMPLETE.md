# Task 2 Complete: Multi-Agent System Adapted for Real Historians

## Summary

Successfully adapted the multi-agent dialogue system from synthetic personas to real historian personas based on actual scholarly work.

## Changes Made

### 1. Updated Multi-Agent System (`agents/multi_agent_system.py`)

**Before:**
- Used synthetic `PersonaManager` with factorial dimensions (field × method × era × orientation)
- System prompts generated from abstract characteristics
- Agent ID from `persona.persona_id`

**After:**
- Uses `HistorianManager` with real scholars from OpenAlex
- System prompts use actual paper abstracts from each historian
- Agent ID from `persona.historian_id` (OpenAlex URL)
- Added `name` attribute for human-readable identification

**Key Code Change:**
```python
def get_system_prompt(self) -> str:
    # Use the historian's actual prompt from their papers
    base_prompt = self.persona.prompt  # Contains real abstracts

    # Add collaboration instructions
    collaboration_instructions = """
    You are collaborating with other historians to develop
    a novel research question and abstract...
    """

    return base_prompt + collaboration_instructions
```

### 2. Updated Experiment Runner (`experiments/experiment_runner.py`)

**Before:**
- Used `PersonaManager.generate_factorial_groups()`
- Tracked persona characteristics (field, method, era, orientation)
- No geometric features

**After:**
- Uses `HistorianManager.sample_groups()` with geometry constraints
- Tracks historian names and triangle geometry
- Three sampling strategies: `filtered`, `stratified`, `random`
- Records 8 geometric features per group:
  - `side_1`, `side_2`, `side_3` (pairwise cosine distances)
  - `perimeter`, `area` (triangle metrics)
  - `min_angle`, `max_angle`, `angle_variance` (angular features)

**Key Code Change:**
```python
# Compute triangle geometry for each group
triangle_geometry = self.historian_manager.compute_triangle_geometry(group)

# Store in ExperimentResult
result = ExperimentResult(
    experiment_id=experiment_id,
    group_composition=[p.to_dict() for p in group],
    triangle_geometry=triangle_geometry,  # NEW
    chat_history=[msg.to_dict() for msg in final_state.messages],
    sources_accessed=[s.to_dict() for s in final_state.sources_accessed],
    final_question=final_state.final_question or "",
    final_abstract=final_state.final_abstract or "",
    turn_count=final_state.turn_count,
    consensus_reached=final_state.consensus_reached,
    timestamp=datetime.now().isoformat()
)
```

### 3. Updated Data Export Format

**Before (results.csv columns):**
```
experiment_id, turn_count, consensus_reached, n_sources_used,
agent_0_field, agent_0_method, agent_0_era, agent_0_orientation,
agent_1_field, agent_1_method, agent_1_era, agent_1_orientation,
agent_2_field, agent_2_method, agent_2_era, agent_2_orientation
```

**After (results.csv columns):**
```
experiment_id, historian_1, historian_2, historian_3,
turn_count, consensus_reached, n_sources_used,
geom_side_1, geom_side_2, geom_side_3,
geom_perimeter, geom_area,
geom_min_angle, geom_max_angle, geom_angle_variance
```

## Test Results

All integration tests passed ✓

```
Test 1: Loading Historian Personas
  ✓ Loaded 25 historians from OpenAlex
  ✓ Each has real papers with abstracts
  ✓ 768-d embeddings computed

Test 2: Generating Groups with Triangle Geometry
  ✓ Generated valid triangular groups
  ✓ Applied distance constraints (0.1 to 0.7)
  ✓ Applied area constraint (min 0.001)
  ✓ Example: Joan Wallach Scott + Reinhart Koselleck + Arjun Appadurai
    - Perimeter: 1.554, Area: 0.099

Test 3: System Prompt Generation
  ✓ Prompts contain actual paper abstracts
  ✓ Include "scholarly perspective" framing
  ✓ Include "embody" instructions
  ✓ Include historian's actual name

Test 4: Experiment Result Structure
  ✓ All required fields present
  ✓ Triangle geometry properly recorded
  ✓ Serializable to JSON
```

## Example Experiment Output

```json
{
  "experiment_id": "exp_00042",
  "group_composition": [
    {
      "historian_id": "https://openalex.org/A5061949360",
      "name": "Arnaldo Momigliano",
      "papers": [...],
      "embedding": [768 floats]
    },
    {
      "historian_id": "https://openalex.org/A5025922453",
      "name": "Georg G. Iggers",
      "papers": [...],
      "embedding": [768 floats]
    },
    {
      "historian_id": "https://openalex.org/A5046659216",
      "name": "John Marincola",
      "papers": [...],
      "embedding": [768 floats]
    }
  ],
  "triangle_geometry": {
    "side_1": 0.518,
    "side_2": 0.402,
    "side_3": 0.400,
    "perimeter": 1.320,
    "area": 0.079,
    "min_angle": 0.864,
    "max_angle": 1.407,
    "angle_variance": 0.065
  },
  "chat_history": [...],
  "sources_accessed": [...],
  "final_question": "How did...",
  "final_abstract": "This study...",
  "turn_count": 23,
  "consensus_reached": true,
  "timestamp": "2024-03-12T14:30:00"
}
```

## Files Modified

1. ✓ `personas/historian_manager.py` (created in Task 1)
2. ✓ `agents/multi_agent_system.py` (adapted)
3. ✓ `experiments/experiment_runner.py` (adapted)
4. ✓ `test_integration.py` (created for verification)

## Next Steps (Task 3)

Ready to proceed to prediction modeling:
- Remove SAE + double debiasing causal inference
- Add embedding-based prediction models
- Predict source selection patterns
- Predict abstract perplexity from triangle geometry
- Feature engineering for embedding distances

## Compatibility Notes

- **Backward incompatible:** Old persona storage format cannot be loaded
- **Migration:** Re-run historian persona generation from OpenAlex data
- **Dependencies:** All existing (langchain, sentence-transformers, numpy, pandas)
- **Breaking changes:**
  - `persona.persona_id` → `persona.historian_id`
  - `persona.field/method/era/orientation` → actual papers + prompt
  - Results CSV schema changed (see above)
