# TCR-Net

**Context-aware risk grading for pre-transmission power-grid file sharing**

TCR-Net grades each pending file-transmission event before it enters a shared work node. Given file-structure, semantic, context, history, and rule evidence, the model predicts one of three operational actions: **release**, **review**, or **block**. It also exports the component scores and rule flags used by the decision, so each prediction can be inspected after deployment.

This repository contains the lightweight code package used for the revised manuscript experiments. Large generated datasets, checkpoints, logs, and figures are intentionally excluded; they can be regenerated with the scripts below.

## Repository layout

```text
TCR-Net/
├── configs/
├── tcrnet/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── config.py
│   ├── experiments.py
│   ├── _export_report.py
│   ├── _plot_results.py
│   └── _latex_tools.py
├── cctnet/
├── scripts/
├── revision_experiments.py
├── tcrnet_paper_pipeline.py
├── reevaluate_all_checkpoints.py
├── REVISION_EXPERIMENTS.md
└── requirements.txt
```

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The revised experiment script automatically searches for a Python executable with `torch`, `numpy`, `sklearn`, `yaml`, and `xgboost`. If needed, set it explicitly:

```bash
PYTHON_BIN=python bash scripts/run_all_revision_experiments.sh
```

## Reproduce the revised experiments

For a quick smoke test:

```bash
QUICK=1 bash scripts/run_all_revision_experiments.sh
```

For the full revised experiment suite:

```bash
bash scripts/run_all_revision_experiments.sh
```

The full runner executes the main comparison, component ablations, attention-direction checks, history-window checks, leave-one-profile transfer, generator-seed robustness, and a frozen confirmatory evaluation. Completed tasks are skipped automatically, so interrupted runs can be resumed by launching the same command again.

Useful options:

```bash
JOBS=4 bash scripts/run_all_revision_experiments.sh
FORCE=1 bash scripts/run_all_revision_experiments.sh
INCLUDE_GENERATOR_ROBUSTNESS=0 bash scripts/run_all_revision_experiments.sh
INCLUDE_CONFIRMATORY=0 bash scripts/run_all_revision_experiments.sh
```

Default outputs are written to:

- `data/revision_experiments/`
- `exp_data/revision_2026_final/`
- `exp_data/revision_2026_confirmatory/`
- `runs/`

These generated directories are ignored by git.

## Reproduce the original paper pipeline

The original pipeline remains available for backward compatibility:

```bash
python tcrnet_paper_pipeline.py all \
  --config configs/tcr_net_paper.yaml \
  --mock-dir data/mock_station_v1 \
  --jobs 4
```

Main subcommands:

| Command | Purpose |
| --- | --- |
| `generate` | Generate the synthetic event dataset |
| `train` | Train TCR-Net, baselines, or ablations |
| `evaluate` | Evaluate a checkpoint |
| `export` | Export result tables |
| `plot` | Generate paper figures |
| `latex` | Fill LaTeX tables and figure references |
| `all` | Run generation, training, export, and plotting |

## Data model

Each sample is a pre-transmission event represented by typed file, source, destination, operator-role, timestamp, context, history, and rule fields. The generator instantiates realistic operational constraints such as allowed record category, destination, role, time, file size, and operation stage. Semi-real profile mixtures model three grid-operation scenarios: transmission inspection, renewable-station operation, and substation maintenance.

The repository releases the generator and experiment code, not operational records. Site identifiers, operator identities, device identifiers, and exact business thresholds are represented by de-identified categories or ranges.

## Notes for reviewers

- Run `QUICK=1 bash scripts/run_all_revision_experiments.sh` first to check the environment.
- Use the full runner for manuscript-level results.
- See `REVISION_EXPERIMENTS.md` for the complete experiment matrix and aggregation protocol.
- Regenerated outputs are intentionally excluded from version control.

## License

Released for academic and research purposes.
