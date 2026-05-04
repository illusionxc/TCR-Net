"""
TCR-Net: Pre-Transmission Risk Grading for Power Inspection Files.

Package structure:
    tcrnet/
    ├── config.py         Configuration loading
    ├── data/
    │   └── synthetic.py  Synthetic data generation
    ├── models/
    │   ├── tcrnet.py     TCR-Net model
    │   └── baselines.py  Baseline methods
    ├── training/
    │   └── trainer.py    Training / evaluation / threshold calibration
    └── experiments.py    Experiment variants (baseline / ablation)
"""

from .config import load_config
from .data.synthetic import build_dataloaders, export_mock_dataset, load_dataset
from .models.tcrnet import TCRNet
from .models.baselines import run_baseline, run_all_baselines, REGISTRY
from .training.trainer import train, evaluate
from .experiments import baseline_variants, ablation_variants

__all__ = [
    "TCRNet",
    "build_dataloaders",
    "evaluate",
    "export_mock_dataset",
    "load_config",
    "load_dataset",
    "train",
    "run_baseline",
    "run_all_baselines",
    "REGISTRY",
    "baseline_variants",
    "ablation_variants",
]
