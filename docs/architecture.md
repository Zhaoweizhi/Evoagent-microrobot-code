# Architecture overview

The framework treats a pretrained LLM as a **reasoning policy** over a
mixed continuous / discrete design space.  At each round the policy
proposes a new design, the design is screened by a critic ensemble, and
only feasible candidates are sent to the (expensive) physical
simulator.  Feedback from the simulator and from optional human
operators is appended to a trajectory memory that is replayed into the
next prompt.

```
                          ┌──────────────┐
                          │   liter/     │   PDF / log corpus
                          │ (RAG corpus) │   (not redistributed)
                          └──────┬───────┘
                                 │ retrieval
                                 ▼
┌───────────────┐         ┌──────────────┐         ┌────────────────┐
│ trajectory    │ replay  │              │ tool    │ MCP server     │
│ memory        │────────►│  LLM agent   │────────►│ - validate_*   │
│ (experience.py)│        │  (client.py) │         │ - run_*        │
└──────┬────────┘         │              │         └──────┬─────────┘
       ▲                  └──────┬───────┘                │
       │ reward / reflection      │ proposal              │ FEA / MBD
       │                          ▼                       ▼
       │                  ┌──────────────┐         ┌────────────────┐
       │                  │ critic       │ veto    │ Maxwell /      │
       └──────────────────│ ensemble     │◄────────│ SolidWorks     │
                          │ (critic.py)  │ result  │ + Adams        │
                          └──────────────┘         └────────────────┘
                                 ▲
                                 │ optional override
                          ┌──────┴───────┐
                          │ human        │
                          │ feedback     │
                          │ (feedback*.py)│
                          └──────────────┘
```

## Module map

| File (under `src/mymcp/`) | Responsibility |
|---|---|
| `client.py` | Top-level agent loop; orchestrates prompts, tool calls, RL / RAG hooks, logging |
| `mcp_adapter.py` | Translate Model-Context-Protocol tool descriptors to OpenAI tool schemas and back |
| `server.py` | Reference MCP server skeleton; the actuator case re-uses it as-is |
| `tool/` | Individual MCP tools the agent may call (validation + simulation per case) |
| `strategy.py` | ε-greedy + discrete-jump exploration scheduler (the *RL-inspired overlay*) |
| `memory.py`, `experience.py` | Short- and long-term experience buffers; provenance metadata |
| `reward.py` | Per-round reward signal fed back to the scheduler |
| `critic.py` | Feasibility / physics critic ensemble (LLM-judges) |
| `reflection.py`, `expel_critique.py` | Reflexion / ExpeL style self-critique used at the end of each rollout |
| `meta_learning.py` | Online extraction of human-readable rules from past trajectories |
| `value_function.py` | Optional state-value head for actor-critic style updates |
| `rag.py` | Local PDF / log retrieval pipeline (FAISS or BM25 fallback) |
| `evaluator.py` | LLM-as-judge module used for the cross-model evaluation table |
| `feedback.py`, `feedback_server.py` | Human feedback ingestion + browser dashboard |

## Two cases, one agent

`src/mymcp/` is **identical** for both case studies; only the registered
MCP tools and the system prompt differ.  This is what allows the
manuscript to claim transferability across very different physical
domains.
