# Case 2: locomotion-system design

This case optimizes five variables describing the leg geometry and battery
position of the untethered millimeter-scale robot. Each feasible design is
constructed in SolidWorks and evaluated in MSC Adams over a 1-s simulation.
Net forward displacement is used as the locomotion objective.

## Structure

- `run_optimization.py`: EvoAgent entry point.
- `server.py`: MCP server for geometry validation and simulation.
- `baselines/run_baselines.py`: common entry point for GA, BO, PSO, SMAC,
  BORE, PFN-CEI and POM.
- `baselines/base_optimizer.py`: shared parameter bounds, constraints,
  simulation calls, logging and checkpoint handling.
- `baselines/*_optimizer.py`: method-specific baseline implementations.

All methods use the same functions in `src/mymcp/tool/robot_leg.py` to validate
geometry and invoke the SolidWorks--Adams evaluator.

## Requirements

- Python 3.10 or later.
- SolidWorks 2022 or later with COM automation enabled.
- MSC Adams View 2022 or later with command-line solver access.
- The local simulator package expected by `src/mymcp/tool/robot_leg.py`.
- Optional baseline dependencies installed with:

```bash
pip install -e ".[bo,smac]"
```

PFN-CEI also requires BOEngineeringBenchmark. Set
`BOENGINEERINGBENCHMARK_DIR` to its local directory or place it at
`case_robot_leg/baselines/BOEngineeringBenchmark`.

## EvoAgent

```bash
python case_robot_leg/run_optimization.py \
  --rag --rl --auto-iterations 200 \
  --base-url https://openrouter.ai/api/v1 \
  --model gpt-5.2 \
  --embedding-base-url https://openrouter.ai/api/v1 \
  --rag-embedding-model openai/text-embedding-3-small
```

## Baselines

Run one method:

```bash
python case_robot_leg/baselines/run_baselines.py \
  --algo bo --seed 42 --max-evals 200
```

Run all seven methods:

```bash
python case_robot_leg/baselines/run_baselines.py \
  --algo all --seed 42 --max-evals 200
```

Available values for `--algo` are `ga`, `bo`, `pso`, `smac`, `bore`,
`pfncei`, `pom` and `all`. Use `--help` for checkpoint, stopping and
method-specific options.
