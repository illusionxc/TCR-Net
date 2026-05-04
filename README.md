# TCR-Net

**Pre-Transmission Risk Grading for Power Inspection Files Using Context-Consistency Modeling**

TCR-Net is a lightweight neural network for three-class risk assessment of power inspection file transmissions. It computes three complementary risk scores — semantic consistency, temporal deviation, and rule conflict — and fuses them through calibrated thresholds into an operational grade (release / review / block). Component scores and rule flags are exported alongside the grade as auditable review evidence.

## Repository Structure

```
├── tcrnet_paper_pipeline.py    # Unified CLI entry point
├── tcrnet/                     # Core library
│   ├── config.py               YAML configuration loader
│   ├── experiments.py          Variant configuration generators
│   ├── data/synthetic.py       Synthetic data generation and loading
│   ├── models/
│   │   ├── tcrnet.py           TCR-Net model definition
│   │   └── baselines.py        Additional baselines
│   ├── training/trainer.py     Training loop, threshold calibration, evaluation
│   ├── _export_report.py       Experiment data to CSV tables (internal)
│   ├── _plot_results.py        Figure generation (internal)
│   └── _latex_tools.py         LaTeX utilities (internal)
├── configs/                    # YAML experiment configs
└── requirements.txt
```

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data
```bash
python tcrnet_paper_pipeline.py generate --config configs/tcr_net_paper.yaml
```

### Train All Variants
```bash
python tcrnet_paper_pipeline.py train --config configs/tcr_net_paper.yaml --all --jobs 4
```

### Export Results & Generate Figures
```bash
python tcrnet_paper_pipeline.py export --output-root runs/tcr_paper
python tcrnet_paper_pipeline.py plot --report-data-dir runs/tcr_paper/report_data --out-dir figures
```

### Full Pipeline
```bash
python tcrnet_paper_pipeline.py all --config configs/tcr_net_paper.yaml --jobs 4
```

## Pipeline Subcommands

| Command | Description |
|---------|-------------|
| `generate` | Generate synthetic dataset |
| `train` | Train models (use --all, --baselines, --ablations, or --single) |
| `evaluate` | Evaluate a single checkpoint |
| `export` | Export experiment results as CSV tables |
| `plot` | Generate paper figures |
| `all` | Full pipeline: generate -> train -> export -> plot |

## Model Architecture

TCR-Net processes each transmission event through:

1. **Shared Encoder** — Cross-modal Transformer aligning file-structure, semantic, and context features
2. **R_cons** — Semantic-consistency score: distance to task-class prototype + attribute prediction error
3. **R_seq** — Temporal-deviation score: Mahalanobis distance from recent behavior trajectory
4. **R_rule** — Rule-conflict score: weighted sum of 5 explicit business-rule violations
5. **Fusion** — Normalized component scores fused via learned weights, mapped to {Low, Medium, High} through validation-tuned thresholds


