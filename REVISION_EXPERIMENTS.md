# TCR-Net revised experiment protocol

Version: 2026-07-30

Entrypoint:

```bash
bash scripts/run_all_revision_experiments.sh
```

## Quick check

Install dependencies and run a smoke test before launching the full experiment suite:

```bash
python -c "import torch, numpy, sklearn, yaml, xgboost; print('environment ok')"
QUICK=1 bash scripts/run_all_revision_experiments.sh
```

The quick mode uses a smaller dataset, one seed per group, and two neural-network epochs. It is intended only for environment validation.

## Full run

```bash
bash scripts/run_all_revision_experiments.sh
```

The runner executes the main comparison, ablations, attention-direction checks, history-window checks, leave-one-profile transfer, generator-seed robustness, aggregation, error analysis, figure generation, and frozen confirmatory evaluation.

## Experiment matrix

| Suite | Purpose | Settings | Seeds | Jobs |
| --- | --- | --- | ---: | ---: |
| `core` | Main comparison and strong baselines | Submitted methods, XGBoost, XGBoost+Rule, GRU, GRU+Rule, RandomForest, SVM, LogisticRegression | 5 | 70 |
| `ablations` | Component contribution | Full, w/o Consistency, w/o Sequence, w/o Rule, w/o Prototype, w/o Transformer | 5 | 30 |
| `attention` | Cross-attention direction | Bidirectional, event-to-history, history-to-event, none | 3 | 12 |
| `history` | History-window sensitivity | `K in {4, 8, 12, 16}` | 3 | 12 |
| `cross_scenario` | Leave-one-profile transfer | Three target profiles with six methods | 3 | 54 |
| `generator_robustness` | Generator-seed robustness | Three generator seeds with six core methods | 1 | 18 |
| `confirmatory` | Frozen confirmatory evaluation | TCR-Net, w/o Rule, XGBoost+Rule, GRU+Rule | 5 | 20 |

Default training seeds are `42 43 44 45 46`. Targeted and transfer suites use `42 43 44`. Generator seeds are `101 202 303`; the confirmatory generator seed is `404`.

## Strong-baseline protocol

XGBoost+Rule and GRU+Rule receive the same five typed rule features as TCR-Net. XGBoost, XGBoost+Rule, GRU, and GRU+Rule are selected on validation data by comparing native three-class outputs with ordered risk-threshold outputs. Macro-F1 is the primary criterion and high-risk recall is the tie-breaker. Test labels are not used for model or threshold selection.

## Transfer protocol

The cross-scenario suite uses leave-one-profile-out evaluation:

1. Select one profile as the target.
2. Train and validate on the other two profiles.
3. Evaluate only on the target-profile test split.
4. Do not use target-profile training or validation samples.

The three profiles are transmission inspection, renewable-station operation, and substation maintenance.

## Generated data

The repository releases the data generator and experiment code. Operational identifiers and exact business thresholds are represented as de-identified categories or ranges. The released record-level samples are generated from the modeled operational logic.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTHON_BIN` | auto-detected | Python executable with all dependencies |
| `JOBS` | `2` | Maximum parallel subprocesses |
| `QUICK` | `0` | Enables smoke-test mode |
| `FORCE` | `0` | Re-runs completed tasks |
| `INCLUDE_GENERATOR_ROBUSTNESS` | `1` | Enables generator-seed robustness |
| `INCLUDE_CONFIRMATORY` | `1` | Enables frozen confirmatory evaluation |
| `DATA_ROOT` | `data/revision_experiments` | Generated dataset root |
| `OUTPUT_ROOT` | `exp_data/revision_2026_final` | Main result root |
| `FIXED_DATA_DIR` | `${DATA_ROOT}/fixed_data` | Fixed dataset used by main suites |
| `FIGURE_DIR` | `figures_auto` | Figure output directory |
| `SEEDS` | `42 43 44 45 46` | Main and ablation seeds |
| `TARGETED_SEEDS` | `42 43 44` | Attention and history seeds |
| `CROSS_SEEDS` | `42 43 44` | Transfer seeds |
| `GENERATOR_SEEDS` | `101 202 303` | Generator robustness seeds |

Example:

```bash
PYTHON_BIN=python JOBS=4 bash scripts/run_all_revision_experiments.sh
```

## Separate suite execution

```bash
python revision_experiments.py prepare
python revision_experiments.py run --suite core --jobs 4
python revision_experiments.py run --suite ablations --jobs 4
python revision_experiments.py run --suite attention --jobs 4
python revision_experiments.py run --suite history --jobs 4
python revision_experiments.py run --suite cross_scenario --jobs 4
python revision_experiments.py run --suite generator_robustness --jobs 4
python revision_experiments.py aggregate --output-root exp_data/revision_2026_final
python revision_experiments.py analyze --output-root exp_data/revision_2026_final
python scripts/plot_revision_on_original_figures.py --revision-root exp_data/revision_2026_final --figure-dir figures_auto
```

Use `--dry-run` to inspect the task matrix without training.
