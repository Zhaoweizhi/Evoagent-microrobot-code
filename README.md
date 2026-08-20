# EvoAgent code release

This repository contains the source code that accompanies the manuscript

> *Cognitive evolution enables de novo design of untethered millimeter-scale robots.*

The framework couples a large language model agent with retrieved physical priors,
lineage memory, deterministic feasibility checks, critic review and optional
expert feedback. It is evaluated on the two design problems of an untethered
millimeter-scale robot:

| Folder | Design problem | Design variables | Simulator |
|---|---|---|---|
| `case_actuator/` | Actuation system: EI-core electromagnetic linear actuator | 9 design variables plus derived bridge and winding quantities | ANSYS Maxwell (PyAEDT) |
| `case_robot_leg/` | Locomotion system: leg morphology and battery placement | 5 design variables (leg lengths, foot angles, battery offset) | SolidWorks + MSC Adams |

This repository contains the source code and configuration needed to inspect
and run the EvoAgent workflow. Numerical source data for the manuscript
figures are organised separately as a source-data package; see
[`docs/reproduce.md`](docs/reproduce.md).

---

## 1. Repository layout

```
paper_code_release/
├── README.md                       # this file
├── LICENSE                         # MIT
├── pyproject.toml                  # package metadata + dependencies
├── requirements.txt                # flat list of runtime deps
├── env.example                     # template for API keys / endpoints
├── .gitignore
│
├── src/                            # importable Python source
│   ├── mymcp/                      # the LLM-Agent core (shared across cases)
│   │   ├── client.py               #   - main agent loop (tool use + RL/RAG hooks)
│   │   ├── strategy.py             #   - adaptive exploration scheduler
│   │   ├── memory.py               #   - short-term + long-term experience buffer
│   │   ├── experience.py
│   │   ├── reward.py               #   - per-round reward construction
│   │   ├── critic.py               #   - feasibility-critic ensemble
│   │   ├── reflection.py           #   - Reflexion / ExpeL style reasoning
│   │   ├── meta_learning.py        #   - online rule extraction
│   │   ├── rag.py                  #   - retrieval over physics literature
│   │   ├── feedback.py             #   - human feedback ingestion
│   │   ├── feedback_server.py      #   - browser dashboard for HITL
│   │   ├── evaluator.py            #   - LLM-as-judge review module
│   │   ├── value_function.py       #   - state-value head (actor-critic)
│   │   ├── expel_critique.py       #   - ExpeL-style critique
│   │   ├── mcp_adapter.py          #   - OpenAI <-> MCP tool-schema bridge
│   │   ├── server.py               #   - MCP server skeleton (used by case_actuator)
│   │   └── tool/
│   │       ├── constraints.py      #   - hard pre-flight constraints (Maxwell)
│   │       ├── maxwell.py          #   - PyAEDT/Maxwell simulation tool
│   │       └── robot_leg.py        #   - SolidWorks/Adams simulation tool
│   │
│   └── maxwell_*.py                # actuator-specific FEA helpers
│       ├── maxwell_pyaedt_run.py
│       ├── maxwell_pyaedt_run_E_only.py
│       ├── maxwell_viz_open.py
│       ├── maxwell_saturation_plot.py
│       └── ga_baseline.py          # standalone GA over Maxwell
│
├── case_actuator/                  # Case 1: electromagnetic actuator
│   ├── README.md
│   ├── run_optimization.py         # main entry point (LLM-Agent rollout)
│   ├── config/
│   │   └── task_electromagnetic_actuator.yaml
│   └── baselines/                  # GA, BO, PSO, SMAC, BORE,
│                                   # PFN-CEI and POM
│
├── case_robot_leg/                 # Case 2: locomotion-system design
│   ├── README.md
│   ├── run_optimization.py         # main entry point
│   ├── server.py                   # MCP server registering robot-leg tools
│   └── baselines/                  # GA, BO, PSO, SMAC, BORE,
│                                   # PFN-CEI and POM
│
└── docs/
    ├── architecture.md             # how the pieces fit together
    ├── third_party.md              # external projects used for comparison
    └── reproduce.md                # how to reproduce the figures of the paper
```

---

## 2. Installation

The agent itself only needs Python 3.10+ and a handful of pure-Python
packages.  The two case studies additionally depend on **commercial**
simulator installations:

| Case | Required commercial software |
|---|---|
| `case_actuator` | ANSYS Electronics Desktop (Maxwell) 2023 R1 or later, Python binding via `pyaedt` |
| `case_robot_leg` | SolidWorks 2022+ with Python COM, MSC Adams View 2022+ |

To install the agent core for code review, extension, and simulator-independent
baseline components:

```bash
git clone https://github.com/Zhaoweizhi/Evoagent-microrobot-code.git
cd Evoagent-microrobot-code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp env.example .env                # then fill in LLM_API_KEY etc.
```

To additionally install the optional baseline dependencies:

```bash
pip install -e ".[bo,smac]"
```

The EvoAgent core does not require the optional baseline packages. Some
baseline adapters depend on external optimiser projects with their own
version constraints, so they can also be installed in a separate environment
when reproducing a specific comparison run.

---

## 3. Quick start

After filling in `.env`, run one of the two case studies:

```bash
# Case 1 — electromagnetic actuator (requires Maxwell)
python case_actuator/run_optimization.py \
       --rag --rl --auto-iterations 200 \
       --base-url https://openrouter.ai/api/v1 \
       --model gpt-5.2 \
       --embedding-base-url https://openrouter.ai/api/v1 \
       --rag-embedding-model openai/text-embedding-3-small

# Case 2 — robot-leg morphology (requires SolidWorks + Adams)
python case_robot_leg/run_optimization.py \
       --rag --rl --auto-iterations 200 \
       --base-url https://openrouter.ai/api/v1 \
       --model gpt-5.2 \
       --embedding-base-url https://openrouter.ai/api/v1 \
       --rag-embedding-model openai/text-embedding-3-small
```

Both entry points share the same CLI surface; pass `--help` for the full
list of switches. The reported EvoAgent runs used GPT-5.2 through an
OpenAI-compatible OpenRouter endpoint and `openai/text-embedding-3-small`
for retrieval embeddings.

---

## 4. Repository scope

The code release is organised separately from the numerical source-data
package and local simulator workspaces.

* **Figure source data.** Processed CSV/JSON tables supporting the figures and
  supplementary figures are provided in the accompanying source-data package.
* **Local optimisation artefacts.** Per-run files such as `AgenticOPT_*.csv`,
  `GA_results_*.csv`, `BO_results_*.csv`, `RobotLeg_*.csv`,
  `*_experience/`, `maxwell_outputs/`, `logs/`, `*.pkl` checkpoints,
  RAG caches and rendered figure files are treated as local run outputs.
* **Third-party projects** (Letta, Reflexion, ExpeL, POM,
  BOEngineeringBenchmark and generative-agent memory references).
  We use these projects via their official releases;
  see [`docs/third_party.md`](docs/third_party.md) for exact versions
  and how to obtain them.
* **Commercial CAD / FEA assets** (Maxwell `.aedt` projects,
  SolidWorks `.SLDASM` assemblies, Adams `.cmd` templates).  The simulator
  wrappers document the expected inputs and software interfaces.

---

## 5. Citation

A BibTeX entry will be added once the manuscript is accepted.  In the
meantime please cite this repository directly:

```bibtex
@misc{evoagent_code_2026,
  title  = {EvoAgent: code release for "Cognitive evolution enables de novo
            design of untethered millimeter-scale robots"},
  author = {Zhao, Weizhi and Tao, Zhi and Li, Peihan and Wang, Xizhi and
            Wang, Qinghong and Wang, Boxian and Zhai, Yanxin and Wei, Wei and
            Wang, Shitong and Li, Shuangqi and Xu, Tiantong and Li, Haiwang},
  year   = {2026},
  url    = {https://github.com/Zhaoweizhi/Evoagent-microrobot-code}
}
```

---

## 6. License

MIT, see [`LICENSE`](LICENSE).
