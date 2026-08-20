# Third-party code

This repository deliberately ships **only our own source code**.
Wherever the agent uses concepts from prior work, we either re-implement
the relevant module from scratch (e.g. ε-greedy, ExpeL-style critique)
or rely on the external project at runtime via its own pip / git
release.

The list below states, for each external dependency, what it is used
for in the manuscript, and where to obtain it.  Reviewers can clone the
upstream repository at the listed commit / version to reproduce any
result that depends on it.

| External project | Used for | Source |
|---|---|---|
| **OpenRouter** (LLM API gateway) | Hosts the `gpt-*`, `claude-*`, `gemini-*`, `deepseek-*`, `qwen-*`, `nova-*`, `grok-*` models reported in the paper | <https://openrouter.ai> |
| **Letta** (long-term memory) | Conceptual reference for the long-term experience buffer in `src/mymcp/memory.py` | <https://github.com/letta-ai/letta> |
| **Reflexion** | Conceptual reference for `src/mymcp/reflection.py` | <https://github.com/noahshinn/reflexion> |
| **ExpeL** | Conceptual reference for `src/mymcp/expel_critique.py` | <https://github.com/LeapLabTHU/ExpeL> |
| **Generative Agents** | Conceptual reference for the rolling memory stream | <https://github.com/joonspk-research/generative_agents> |
| **POM** | Baseline (`case_actuator/baselines/pom_optimizer.py`) | <https://github.com/ninja-wm/POM> |
| **PFN-CEI / BOEngineeringBenchmark** | Baseline (`case_actuator/baselines/pfn_cei_optimizer.py`) | <https://github.com/PFNs/PFNs4BO> |
| **SMAC3** | Baseline (`case_actuator/baselines/smac_optimizer.py`) | `pip install smac` |
| **PyTorch** | Runtime dependency for PFN-CEI and POM adapters | `pip install torch` |
| **PyAEDT** | Driver for ANSYS Maxwell (Case 1 simulator) | `pip install pyaedt` |
| **MSC Adams**, **SolidWorks** | Multi-body simulator + CAD regenerator (Case 2) | Commercial — see vendor websites |

If you re-run our experiments with a different version of any of the
above, please record it in your run log; the reward / score values are
specific to the simulator outputs and may shift slightly across solver
versions.
