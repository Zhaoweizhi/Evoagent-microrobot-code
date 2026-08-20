# Case 1 — EI-core electromagnetic linear actuator

A constrained physical black-box design problem for an EI-core
electromagnetic linear actuator. The optimiser proposes the geometric
variables reported in the Supplementary Information, including magnet
dimensions, yoke/bridge dimensions, air-gap height, slot geometry,
actuator stroke and bridge-to-yoke thickness ratio. Derived quantities
such as slot-wall thickness, bridge thickness and winding-layer count
enter deterministic pre-simulation viability checks before Maxwell
evaluation.

## Files

* `run_optimization.py` — main LLM-Agent entry point.  Spawns the MCP
  server (`src/mymcp/server.py`), wires up RAG / RL / critic / human
  feedback, and runs the agent loop.
* `config/task_electromagnetic_actuator.yaml` — variable bounds,
  constraint definitions and reference operating point.
* `baselines/` — GA, BO, PSO, SMAC, BORE, PFN-CEI and POM wrappers
  corresponding to the actuator comparison reported in the paper.

## Required commercial software

* **ANSYS Electronics Desktop (Maxwell) 2023 R1+**, with the
  `pyaedt` Python binding available in the same environment.

## Running

```bash
# Full EvoAgent rollout (200 maximum optimisation iterations)
python case_actuator/run_optimization.py \
       --rag --rl --auto-iterations 200 \
       --base-url https://openrouter.ai/api/v1 \
       --model gpt-5.2 \
       --embedding-base-url https://openrouter.ai/api/v1 \
       --rag-embedding-model openai/text-embedding-3-small

# Ablation: disable retrieval
python case_actuator/run_optimization.py --no-rag --rl --auto-iterations 200

# Ablation: disable lineage-memory / RL-style scheduler
python case_actuator/run_optimization.py --rag --no-rl --auto-iterations 200

# Use a specific LLM (any OpenAI-compatible endpoint)
python case_actuator/run_optimization.py \
       --rag --rl \
       --base-url https://openrouter.ai/api/v1 \
       --model    gpt-5.2
```

A non-exhaustive list of additional flags:

| Flag | Effect |
|---|---|
| `--feedback-server` | Launch the browser dashboard for human-in-the-loop intervention |
| `--mode local` | Use a local Ollama-style endpoint instead of the cloud LLM |
| `--auto-iterations N` | Cap the agent at `N` optimisation iterations |
| `--min-iterations N` / `--convergence-window N` | Early-stopping policy |
| `--rag-dir PATH` | Use your own PDF / log corpus for retrieval |
| `--rag-embedding-model MODEL` | Embedding model for retrieval; paper runs used `openai/text-embedding-3-small` |

## Running the classical baselines

```bash
# All baselines with matched evaluation budget
python case_actuator/baselines/run_baselines.py --max-evals 200

# A single algorithm
python case_actuator/baselines/run_baselines.py --algo ga --max-evals 200
```

The baseline runner re-uses the same Maxwell simulator wrapper and
deterministic feasibility checks as the agent. The paper reports rounds to
reach score thresholds using successful physical evaluations that returned a
valid numerical score. Optional baseline dependencies are independent of the
core EvoAgent workflow and may be installed separately for specific
comparison runs.
