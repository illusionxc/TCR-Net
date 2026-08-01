


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from tcrnet import (
    attention_direction_variants,
    history_window_variants,
    load_config,
    revision_ablation_variants,
    revision_core_variants,
    run_baseline,
    train,
)
from tcrnet.data.synthetic import (
    FileEventDataset,
    SCENARIO_PROFILES,
    export_mock_dataset,
    load_dataset,
)
from tcrnet.training.trainer import seed_everything


REQUIRED_SUITES = (
    "core",
    "ablations",
    "attention",
    "history",
    "cross_scenario",
)
OPTIONAL_SUITES = ("generator_robustness",)
ALL_SUITES = REQUIRED_SUITES + OPTIONAL_SUITES
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_TARGETED_SEEDS = (42, 43, 44)
DEFAULT_GENERATOR_SEEDS = (101, 202, 303)
DEFAULT_HISTORY_WINDOWS = (4, 8, 12, 16)
DEFAULT_SCENARIOS = (
    "transmission_inspection",
    "renewable_station",
    "substation_maintenance",
)
PRIMARY_METRICS = (
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "high_recall",
    "far",
    "high_f1",
)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value.strip())
    return value.strip("_").lower()


def _json_safe(value: Any) -> Any:

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _payload_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_commit(project_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not-a-git-repository"


def _dataset_complete(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "metadata.json")
    )


def _with_quick_overrides(config: dict, quick: bool) -> dict:
    cfg = deepcopy(config)
    if quick:
        cfg["paper_train"]["epochs"] = 2
        cfg["paper_train"]["early_stop_patience"] = 1
        cfg["paper_train"]["log_interval"] = 1000
        cfg["baseline"]["gru_epochs"] = 2
        cfg["baseline"]["gru_early_stop_patience"] = 1
        cfg["baseline"]["xgb_n_estimators"] = 30
        cfg["baseline"]["rf_n_estimators"] = 30
    return cfg


def _set_common_task_config(
    config: dict,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    quick: bool,
) -> dict:
    cfg = _with_quick_overrides(config, quick)
    cfg["seed"] = int(seed)
    cfg["output_dir"] = str(output_dir.resolve())
    cfg["data"]["source"] = "station_jsonl"
    cfg["data"]["data_dir"] = str(data_dir.resolve())
    return cfg


def _prepare_one_dataset(
    config: dict,
    seed: int,
    output_dir: Path,
    force: bool,
) -> dict:
    if _dataset_complete(output_dir) and not force:
        return {
            "status": "skipped",
            "seed": seed,
            "output_dir": str(output_dir.resolve()),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    files = export_mock_dataset(config, seed=seed, output_dir=output_dir)
    return {
        "status": "generated",
        "seed": seed,
        "output_dir": str(output_dir.resolve()),
        "files": files,
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    data_root = Path(args.data_root)
    quick = bool(args.quick)
    if quick:
        train_samples, val_samples, test_samples = 128, 64, 64
    else:
        train_samples = int(args.scenario_train_samples)
        val_samples = int(args.scenario_val_samples)
        test_samples = int(args.scenario_test_samples)

    rows = []


    fixed_cfg = deepcopy(base)
    fixed_cfg["data"].update(
        source="synthetic",
        data_dir=None,
        scenario="base",
        history_crop_length=None,
    )
    if quick:
        fixed_cfg["data"].update(
            train_samples=128, val_samples=64, test_samples=64,
        )
    rows.append(_prepare_one_dataset(
        fixed_cfg,
        seed=int(args.base_generator_seed),
        output_dir=data_root / "fixed_data",
        force=args.force,
    ))


    history_cfg = deepcopy(base)
    history_cfg["data"].update(
        source="synthetic",
        data_dir=None,
        scenario="base",
        history_length=max(args.history_windows),
        history_crop_length=None,
    )
    if quick:
        history_cfg["data"].update(
            train_samples=128, val_samples=64, test_samples=64,
        )
    rows.append(_prepare_one_dataset(
        history_cfg,
        seed=int(args.base_generator_seed),
        output_dir=data_root / f"history_kmax_{max(args.history_windows)}",
        force=args.force,
    ))


    for index, scenario in enumerate(args.scenarios):
        if scenario not in SCENARIO_PROFILES:
            raise ValueError(
                f"Unknown scenario {scenario!r}, choices: {sorted(SCENARIO_PROFILES)}"
            )
        scenario_cfg = deepcopy(base)
        scenario_cfg["data"].update(
            source="synthetic",
            data_dir=None,
            scenario=scenario,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            history_crop_length=None,
        )
        rows.append(_prepare_one_dataset(
            scenario_cfg,
            seed=int(args.base_generator_seed) + index * 1000,
            output_dir=data_root / "scenarios" / scenario,
            force=args.force,
        ))


    for generator_seed in args.generator_seeds:
        generator_cfg = deepcopy(base)
        generator_cfg["data"].update(
            source="synthetic",
            data_dir=None,
            scenario="base",
            history_crop_length=None,
        )
        if quick:
            generator_cfg["data"].update(
                train_samples=128, val_samples=64, test_samples=64,
            )
        rows.append(_prepare_one_dataset(
            generator_cfg,
            seed=int(generator_seed),
            output_dir=data_root / "generator_seeds" / str(generator_seed),
            force=args.force,
        ))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "quick": quick,
        "datasets": rows,
        "note": (
            "scenarios are controlled synthetic profiles, not empirical "
            "real-world station distributions"
        ),
    }
    _json_dump(data_root / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _task(
    *,
    suite: str,
    method: str,
    kind: str,
    seed: int,
    config: dict,
    output_dir: Path,
    data_variant: str,
    source_data_dirs: Iterable[Path] | None = None,
    target_data_dir: Path | None = None,
) -> dict:
    task_id = "__".join([
        _slug(suite), _slug(method), _slug(data_variant), f"seed_{seed}",
    ])
    payload = {
        "task_id": task_id,
        "suite": suite,
        "method": method,
        "kind": kind,
        "seed": int(seed),
        "config": config,
        "output_dir": str(output_dir.resolve()),
        "data_variant": data_variant,
    }
    if source_data_dirs is not None:
        payload["source_data_dirs"] = [
            str(Path(path).resolve()) for path in source_data_dirs
        ]
    if target_data_dir is not None:
        payload["target_data_dir"] = str(Path(target_data_dir).resolve())
    return payload


def _build_standard_tasks(
    suite: str,
    base: dict,
    fixed_data_dir: Path,
    data_root: Path,
    output_root: Path,
    seeds: tuple[int, ...],
    targeted_seeds: tuple[int, ...],
    history_windows: tuple[int, ...],
    quick: bool,
) -> list[dict]:
    tasks: list[dict] = []

    if suite == "core":
        for seed in seeds:
            seed_root = output_root / suite / f"seed_{seed}"
            common = _set_common_task_config(
                base, seed, fixed_data_dir, seed_root, quick,
            )
            for method, variant in revision_core_variants(common).items():
                out = seed_root / _slug(method)
                variant["seed"] = seed
                variant["output_dir"] = str(out.resolve())
                tasks.append(_task(
                    suite=suite, method=method, kind="neural", seed=seed,
                    config=variant, output_dir=out, data_variant="fixed_data",
                ))
            for method in (
                "XGBoost", "XGBoost+Rule", "GRU", "GRU+Rule",
                "RandomForest", "SVM",
                "LogisticRegression",
            ):
                out = seed_root / _slug(method)
                cfg = _set_common_task_config(
                    base, seed, fixed_data_dir, out, quick,
                )
                tasks.append(_task(
                    suite=suite, method=method, kind="baseline", seed=seed,
                    config=cfg, output_dir=out, data_variant="fixed_data",
                ))

    elif suite == "ablations":
        for seed in seeds:
            seed_root = output_root / suite / f"seed_{seed}"
            common = _set_common_task_config(
                base, seed, fixed_data_dir, seed_root, quick,
            )
            for method, variant in revision_ablation_variants(common).items():
                out = seed_root / _slug(method)
                variant["seed"] = seed
                variant["output_dir"] = str(out.resolve())
                tasks.append(_task(
                    suite=suite, method=method, kind="neural", seed=seed,
                    config=variant, output_dir=out, data_variant="fixed_data",
                ))

    elif suite == "attention":
        for seed in targeted_seeds:
            seed_root = output_root / suite / f"seed_{seed}"
            common = _set_common_task_config(
                base, seed, fixed_data_dir, seed_root, quick,
            )
            for method, variant in attention_direction_variants(common).items():
                out = seed_root / _slug(method)
                variant["seed"] = seed
                variant["output_dir"] = str(out.resolve())
                tasks.append(_task(
                    suite=suite, method=method, kind="neural", seed=seed,
                    config=variant, output_dir=out, data_variant="fixed_data",
                ))

    elif suite == "history":
        history_data = data_root / f"history_kmax_{max(history_windows)}"
        if not _dataset_complete(history_data):
            raise FileNotFoundError(
                f"History-window data does not exist: {history_data}; run prepare first"
            )
        for seed in targeted_seeds:
            seed_root = output_root / suite / f"seed_{seed}"
            common = _set_common_task_config(
                base, seed, history_data, seed_root, quick,
            )
            common["data"]["history_length"] = max(history_windows)
            variants = history_window_variants(common, history_windows)
            for method, variant in variants.items():
                out = seed_root / _slug(method)
                variant["seed"] = seed
                variant["output_dir"] = str(out.resolve())
                tasks.append(_task(
                    suite=suite, method=method, kind="neural", seed=seed,
                    config=variant, output_dir=out,
                    data_variant=_slug(method),
                ))

    else:
        raise ValueError(f"Unsupported standard suite: {suite}")

    return tasks


def _build_cross_scenario_tasks(
    base: dict,
    data_root: Path,
    output_root: Path,
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    quick: bool,
) -> list[dict]:
    tasks = []
    scenario_root = data_root / "scenarios"
    for target in scenarios:
        target_dir = scenario_root / target
        sources = [scenario_root / item for item in scenarios if item != target]
        for path in sources + [target_dir]:
            if not _dataset_complete(path):
                raise FileNotFoundError(
                    f"Cross-profile data does not exist: {path}; run prepare first"
                )
        for seed in seeds:
            fold_root = (
                output_root / "cross_scenario" / f"target_{_slug(target)}"
                / f"seed_{seed}"
            )
            common = _set_common_task_config(
                base, seed, sources[0], fold_root, quick,
            )
            neural_variants = revision_core_variants(common)
            for method in ("TCR-Net", "Unified-Multimodal"):
                out = fold_root / _slug(method)
                cfg = neural_variants[method]
                cfg["seed"] = seed
                cfg["output_dir"] = str(out.resolve())
                tasks.append(_task(
                    suite="cross_scenario", method=method, kind="neural",
                    seed=seed, config=cfg, output_dir=out,
                    data_variant=f"target={target}",
                    source_data_dirs=sources, target_data_dir=target_dir,
                ))
            for method in (
                "XGBoost", "XGBoost+Rule", "GRU", "GRU+Rule",
            ):
                out = fold_root / _slug(method)
                cfg = _set_common_task_config(
                    base, seed, sources[0], out, quick,
                )
                tasks.append(_task(
                    suite="cross_scenario", method=method, kind="baseline",
                    seed=seed, config=cfg, output_dir=out,
                    data_variant=f"target={target}",
                    source_data_dirs=sources, target_data_dir=target_dir,
                ))
    return tasks


def _build_generator_tasks(
    base: dict,
    data_root: Path,
    output_root: Path,
    generator_seeds: tuple[int, ...],
    quick: bool,
) -> list[dict]:
    tasks = []
    train_seed = int(base["seed"])
    for generator_seed in generator_seeds:
        data_dir = data_root / "generator_seeds" / str(generator_seed)
        if not _dataset_complete(data_dir):
            raise FileNotFoundError(
                f"Generator-seed data does not exist: {data_dir}; run prepare first"
            )
        root = (
            output_root / "generator_robustness"
            / f"generator_seed_{generator_seed}"
        )
        common = _set_common_task_config(
            base, train_seed, data_dir, root, quick,
        )
        variants = revision_core_variants(common)
        for method in ("TCR-Net", "Unified-Multimodal"):
            out = root / _slug(method)
            cfg = variants[method]
            cfg["output_dir"] = str(out.resolve())
            tasks.append(_task(
                suite="generator_robustness", method=method, kind="neural",
                seed=train_seed, config=cfg, output_dir=out,
                data_variant=f"generator_seed={generator_seed}",
            ))
        for method in (
            "XGBoost", "XGBoost+Rule", "GRU", "GRU+Rule",
        ):
            out = root / _slug(method)
            cfg = _set_common_task_config(
                base, train_seed, data_dir, out, quick,
            )
            tasks.append(_task(
                suite="generator_robustness", method=method,
                kind="baseline", seed=train_seed, config=cfg,
                output_dir=out,
                data_variant=f"generator_seed={generator_seed}",
            ))
    return tasks


def build_tasks(args: argparse.Namespace) -> list[dict]:
    base = load_config(args.config)
    if args.lambda_proto is not None:
        base.setdefault("revision_optimization", {})["lambda_proto"] = float(
            args.lambda_proto
        )
    fixed_data = Path(args.fixed_data_dir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    seeds = tuple(args.seeds[:1] if args.quick else args.seeds)
    targeted = tuple(
        args.targeted_seeds[:1] if args.quick else args.targeted_seeds
    )
    cross_seeds = tuple(
        args.cross_seeds[:1] if args.quick else args.cross_seeds
    )

    if args.suite in {"core", "ablations", "attention", "history"}:
        return _build_standard_tasks(
            args.suite, base, fixed_data, data_root, output_root,
            seeds, targeted, tuple(args.history_windows), args.quick,
        )
    if args.suite == "cross_scenario":
        return _build_cross_scenario_tasks(
            base, data_root, output_root, cross_seeds,
            tuple(args.scenarios), args.quick,
        )
    if args.suite == "generator_robustness":
        generator_seeds = tuple(
            args.generator_seeds[:1]
            if args.quick else args.generator_seeds
        )
        return _build_generator_tasks(
            base, data_root, output_root, generator_seeds, args.quick,
        )
    raise ValueError(f"Unknown experiment suite: {args.suite}")


def _build_merged_loaders(task: dict):
    source_dirs = [Path(path) for path in task["source_data_dirs"]]
    target_dir = Path(task["target_data_dir"])
    source_train, source_val = [], []
    meta = None
    for source_dir in source_dirs:
        train_samples, val_samples, _, source_meta = load_dataset(source_dir)
        source_train.extend(train_samples)
        source_val.extend(val_samples)
        if meta is None:
            meta = source_meta
        else:
            for key in (
                "num_intents", "num_objects", "num_sources", "state_dim",
                "context_dim", "history_feature_dim", "sequence_length",
            ):
                if int(meta[key]) != int(source_meta[key]):
                    raise ValueError(
                        f"Cross-profile metadata mismatch: {key} "
                        f"{meta[key]} != {source_meta[key]}"
                    )
    _, _, target_test, target_meta = load_dataset(target_dir)
    for key in (
        "num_intents", "num_objects", "num_sources", "state_dim",
        "context_dim", "history_feature_dim", "sequence_length",
    ):
        if int(meta[key]) != int(target_meta[key]):
            raise ValueError(
                f"Target-profile metadata mismatch: {key} "
                f"{meta[key]} != {target_meta[key]}"
            )
    cfg = task["config"]
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    generator = torch.Generator().manual_seed(int(task["seed"]))
    common = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(
            FileEventDataset(source_train), shuffle=True,
            generator=generator, **common,
        ),
        DataLoader(FileEventDataset(source_val), shuffle=False, **common),
        DataLoader(FileEventDataset(target_test), shuffle=False, **common),
        meta,
    )


def _normal_loaders(task: dict):
    from tcrnet.data.synthetic import build_dataloaders
    return build_dataloaders(task["config"], int(task["seed"]))


def _normalize_baseline_result(result: dict) -> dict:
    if "error" in result:
        raise RuntimeError(result["error"])
    return {
        "val": result.get("val_metrics", {}),
        "test": result.get("test_metrics", {}),
        "runtime": result.get("runtime", {}),
        "model_config": result.get("model_config", {}),
        "thresholds": result.get("thresholds", {}),
        "_predictions": result.get("predictions"),
    }


def _loader_slice_metadata(loader: DataLoader) -> list[dict]:
    rows = []
    for batch in loader:
        batch_size = int(batch["initial_state"].shape[0])
        for index in range(batch_size):
            row = {}
            for field in (
                "intent_id", "object_type", "source_type",
                "anomaly_label", "anomaly_type",
            ):
                value = batch.get(field)
                row[field] = (
                    int(value[index].item()) if value is not None else ""
                )
            rows.append(row)
    return rows


def _export_baseline_details(
    output_dir: Path,
    loaders: tuple,
    predictions: dict | None,
) -> None:
    if not predictions:
        return
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for split, loader in (("val", loaders[1]), ("test", loaders[2])):
        payload = predictions.get(split)
        if not payload:
            continue
        metadata = _loader_slice_metadata(loader)
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        if not (len(metadata) == len(y_true) == len(y_pred)):
            raise ValueError(
                f"{split} Per-sample result length mismatch: "
                f"metadata={len(metadata)}, y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}"
            )
        rows = []
        for index, (meta, true_value, pred_value) in enumerate(
            zip(metadata, y_true, y_pred)
        ):
            rows.append({
                "sample_id": index,
                **meta,
                "risk_label_true": int(true_value),
                "risk_label_pred": int(pred_value),
            })
        path = details_dir / f"{split}_details.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_task(task: dict) -> dict:
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    project_dir = Path(__file__).resolve().parent
    config_sha256 = _payload_sha256(task["config"])
    git_commit = _git_commit(project_dir)
    started = time.time()
    try:
        seed_everything(int(task["seed"]))
        loaders = (
            _build_merged_loaders(task)
            if "source_data_dirs" in task
            else _normal_loaders(task)
        )
        config = task["config"]
        if task["kind"] == "neural":
            metrics = train(config, external_loaders=loaders)
        elif task["kind"] == "baseline":
            device_name = config.get("device", "auto")
            device = torch.device(
                "cuda"
                if device_name == "auto" and torch.cuda.is_available()
                else device_name if device_name != "auto" else "cpu"
            )
            baseline_result = run_baseline(
                task["method"], config,
                loaders[0], loaders[1], loaders[2], device,
            )
            metrics = _normalize_baseline_result(baseline_result)
            predictions = metrics.pop("_predictions", None)
            _export_baseline_details(output_dir, loaders, predictions)
        else:
            raise ValueError(f"Unknown task kind: {task['kind']}")

        result = {
            "status": "completed",
            "task_id": task["task_id"],
            "suite": task["suite"],
            "method": task["method"],
            "kind": task["kind"],
            "seed": task["seed"],
            "data_variant": task["data_variant"],
            "output_dir": str(output_dir.resolve()),
            "metrics": metrics,
            "runtime_seconds": float(time.time() - started),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "task_id": task["task_id"],
            "suite": task["suite"],
            "method": task["method"],
            "kind": task["kind"],
            "seed": task["seed"],
            "data_variant": task["data_variant"],
            "output_dir": str(output_dir.resolve()),
            "runtime_seconds": float(time.time() - started),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    _json_dump(result_path, result)
    return result


def _worker_mode(task_file: str) -> None:
    task = json.loads(Path(task_file).read_text(encoding="utf-8"))
    result = run_task(task)
    print(json.dumps(
        {
            key: result.get(key)
            for key in (
                "task_id", "status", "suite", "method", "seed",
                "data_variant", "runtime_seconds", "error",
            )
            if key in result
        },
        ensure_ascii=False,
    ))
    if result["status"] != "completed":
        raise SystemExit(1)


def _existing_result(task: dict) -> dict | None:
    path = Path(task["output_dir"]) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cmd_run(args: argparse.Namespace) -> None:
    tasks = build_tasks(args)
    if args.methods:
        wanted = set(args.methods)
        available = {task["method"] for task in tasks}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(
                f"Suite {args.suite!r} does not contain methods/settings: {unknown}; "
                f"choices: {sorted(available)}"
            )
        tasks = [task for task in tasks if task["method"] in wanted]
    output_root = Path(args.output_root)
    manifest_dir = output_root / "manifests" / args.suite
    task_dir = manifest_dir / "tasks"
    log_dir = output_root / "logs" / args.suite
    task_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    runnable = []
    skipped = []
    for task in tasks:
        existing = _existing_result(task)
        if (
            not args.force
            and existing is not None
            and existing.get("status") == "completed"
            and existing.get("config_sha256")
            == _payload_sha256(task["config"])
        ):
            skipped.append(task)
            continue
        task_path = task_dir / f"{task['task_id']}.json"
        _json_dump(task_path, task)
        runnable.append((task, task_path))

    _json_dump(
        manifest_dir / "suite_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "suite": args.suite,
            "quick": args.quick,
            "jobs": args.jobs,
            "task_count": len(tasks),
            "runnable_count": len(runnable),
            "skipped_count": len(skipped),
            "tasks": tasks,
        },
    )
    print(
        f"[{args.suite}] tasks={len(tasks)} "
        f"run={len(runnable)} skipped={len(skipped)} jobs={args.jobs}"
    )
    if args.dry_run:
        for task, _ in runnable:
            print(
                f"  {task['task_id']}: {task['kind']} "
                f"{task['method']} {task['data_variant']}"
            )
        return
    if not runnable:
        print(f"[{args.suite}] All tasks are already complete.")
        return

    pending = list(runnable)
    running: list[dict] = []
    failures = []
    while pending or running:
        while pending and len(running) < max(1, int(args.jobs)):
            task, task_path = pending.pop(0)
            log_path = log_dir / f"{task['task_id']}.log"
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                [
                    sys.executable, str(Path(__file__).resolve()),
                    "--worker-task", str(task_path),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent),
                text=True,
            )
            running.append({
                "task": task, "proc": proc, "log_file": log_file,
                "log_path": log_path,
            })
            print(f"  [start] {task['task_id']} pid={proc.pid}")

        remaining = []
        for item in running:
            return_code = item["proc"].poll()
            if return_code is None:
                remaining.append(item)
                continue
            item["log_file"].close()
            result = _existing_result(item["task"])
            if return_code != 0 or not result or result.get("status") != "completed":
                failures.append({
                    "task_id": item["task"]["task_id"],
                    "return_code": return_code,
                    "log": str(item["log_path"].resolve()),
                    "result": result,
                })
                print(
                    f"  [fail] {item['task']['task_id']} "
                    f"log={item['log_path']}"
                )
            else:
                print(
                    f"  [done] {item['task']['task_id']} "
                    f"{result['runtime_seconds']:.1f}s"
                )
        running = remaining
        if pending or running:
            time.sleep(1.0)

    if failures:
        _json_dump(manifest_dir / "failures.json", failures)
        raise SystemExit(
            f"[{args.suite}] {len(failures)} tasks failed;"
            f"see {manifest_dir / 'failures.json'}"
        )
    print(f"[{args.suite}] All tasks finished.")


def _extract_row(result: dict, path: Path) -> dict:
    metrics = result.get("metrics", {})
    test_metrics = metrics.get("test", {})
    runtime = metrics.get("runtime", {})
    return {
        "suite": result["suite"],
        "method": result["method"],
        "kind": result["kind"],
        "seed": result["seed"],
        "data_variant": result["data_variant"],
        **{
            metric: test_metrics.get(metric, "")
            for metric in PRIMARY_METRICS
        },
        "runtime_seconds": result.get(
            "runtime_seconds", runtime.get("seconds", "")
        ),
        "epochs_ran": runtime.get("epochs_ran", ""),
        "result_file": str(path.resolve()),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def cmd_aggregate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    result_paths = sorted(output_root.glob("**/result.json"))
    rows, failures = [], []
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            failures.append({"path": str(path.resolve()), **result})
            continue
        rows.append(_extract_row(result, path))

    aggregate_dir = output_root / "aggregated"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    raw_path = aggregate_dir / "raw_results.csv"
    raw_fields = [
        "suite", "method", "kind", "seed", "data_variant",
        *PRIMARY_METRICS, "runtime_seconds", "epochs_ran", "result_file",
    ]
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        group_variant = (
            "generator_seed_sweep"
            if row["suite"] == "generator_robustness"
            else str(row["data_variant"])
        )
        key = (
            str(row["suite"]), str(row["method"]),
            str(row["kind"]), group_variant,
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (suite, method, kind, data_variant), group in sorted(grouped.items()):
        aggregate_row = {
            "suite": suite,
            "method": method,
            "kind": kind,
            "data_variant": data_variant,
            "n_runs": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
            "replicates": ",".join(
                str(row["data_variant"]) for row in group
            ),
        }
        for metric in PRIMARY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row[metric] not in {"", None}
            ]
            mean_value, std_value = _mean_std(values)
            aggregate_row[f"{metric}_mean"] = mean_value
            aggregate_row[f"{metric}_std"] = std_value
        aggregate_rows.append(aggregate_row)

    aggregate_path = aggregate_dir / "aggregate_results.csv"
    aggregate_fields = [
        "suite", "method", "kind", "data_variant", "n_runs", "seeds",
        "replicates",
        *[
            field
            for metric in PRIMARY_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    _json_dump(aggregate_dir / "failures.json", failures)
    _json_dump(
        aggregate_dir / "aggregation_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "result_files": len(result_paths),
            "completed_results": len(rows),
            "failed_results": len(failures),
            "groups": len(aggregate_rows),
            "raw_results": str(raw_path.resolve()),
            "aggregate_results": str(aggregate_path.resolve()),
        },
    )
    print(
        f"[aggregate] completed={len(rows)} failed={len(failures)} "
        f"groups={len(aggregate_rows)}"
    )
    print(f"[aggregate] raw: {raw_path.resolve()}")
    print(f"[aggregate] mean+/-std: {aggregate_path.resolve()}")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_field(row: dict, name: str) -> int:
    return int(float(row[name]))


def _slice_statistics(rows: list[dict]) -> dict:
    count = len(rows)
    correct = sum(
        _int_field(row, "risk_label_true")
        == _int_field(row, "risk_label_pred")
        for row in rows
    )
    low_rows = [
        row for row in rows if _int_field(row, "risk_label_true") == 0
    ]
    risk_rows = [
        row for row in rows if _int_field(row, "risk_label_true") > 0
    ]
    high_rows = [
        row for row in rows if _int_field(row, "risk_label_true") == 2
    ]
    false_alarms = sum(
        _int_field(row, "risk_label_pred") > 0 for row in low_rows
    )
    false_releases = sum(
        _int_field(row, "risk_label_pred") == 0 for row in risk_rows
    )
    high_misses = sum(
        _int_field(row, "risk_label_pred") < 2 for row in high_rows
    )
    return {
        "n": count,
        "accuracy": correct / count if count else 0.0,
        "error_rate": (count - correct) / count if count else 0.0,
        "low_n": len(low_rows),
        "far": false_alarms / len(low_rows) if low_rows else "",
        "risk_n": len(risk_rows),
        "false_release_rate": (
            false_releases / len(risk_rows) if risk_rows else ""
        ),
        "high_n": len(high_rows),
        "high_miss_rate": (
            high_misses / len(high_rows) if high_rows else ""
        ),
    }


def _error_type(true_value: int, pred_value: int) -> str:
    if true_value == 2 and pred_value < 2:
        return "high_risk_miss"
    if true_value > 0 and pred_value == 0:
        return "false_release"
    if true_value == 0 and pred_value > 0:
        return "false_alarm"
    return "other_misclassification"


def cmd_analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    selected_suites = set(args.suites)
    detail_paths = sorted(output_root.glob("**/details/test_details.csv"))
    records = []
    skipped = []
    for detail_path in detail_paths:
        result_path = detail_path.parent.parent / "result.json"
        if not result_path.exists():
            skipped.append({
                "details": str(detail_path.resolve()),
                "reason": "missing result.json",
            })
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "completed"
            or result.get("suite") not in selected_suites
        ):
            continue
        with detail_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                records.append({
                    "suite": result["suite"],
                    "method": result["method"],
                    "seed": result["seed"],
                    "data_variant": result["data_variant"],
                    "result_file": str(result_path.resolve()),
                    **row,
                })

    dimensions = (
        "risk_label_true",
        "anomaly_type",
        "intent_id",
        "object_type",
        "source_type",
    )
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in records:
        for dimension in dimensions:
            if dimension not in row or row[dimension] == "":
                continue
            key = (
                str(row["suite"]),
                str(row["method"]),
                str(row["data_variant"]),
                dimension,
                str(row[dimension]),
            )
            grouped.setdefault(key, []).append(row)

    slice_rows = []
    for key, group in sorted(grouped.items()):
        suite, method, data_variant, dimension, slice_value = key
        run_ids = {
            (str(row["seed"]), str(row["result_file"]))
            for row in group
        }
        slice_rows.append({
            "suite": suite,
            "method": method,
            "data_variant": data_variant,
            "slice_dimension": dimension,
            "slice_value": slice_value,
            "n_runs": len(run_ids),
            **_slice_statistics(group),
        })

    confusion_groups: dict[tuple[str, str, str, int, int], int] = {}
    error_rows = []
    preserved_fields = (
        "sample_id", "intent_id", "object_type", "source_type",
        "anomaly_label", "anomaly_type", "risk_label_true",
        "risk_label_pred", "R_cons", "R_seq", "R_rule", "R_total",
    )
    for row in records:
        true_value = _int_field(row, "risk_label_true")
        pred_value = _int_field(row, "risk_label_pred")
        confusion_key = (
            str(row["suite"]),
            str(row["method"]),
            str(row["data_variant"]),
            true_value,
            pred_value,
        )
        confusion_groups[confusion_key] = (
            confusion_groups.get(confusion_key, 0) + 1
        )
        if true_value != pred_value:
            error_rows.append({
                "suite": row["suite"],
                "method": row["method"],
                "seed": row["seed"],
                "data_variant": row["data_variant"],
                "error_type": _error_type(true_value, pred_value),
                **{
                    field: row.get(field, "")
                    for field in preserved_fields
                },
                "result_file": row["result_file"],
            })
    error_priority = {
        "high_risk_miss": 0,
        "false_release": 1,
        "false_alarm": 2,
        "other_misclassification": 3,
    }
    error_rows.sort(key=lambda row: (
        error_priority[row["error_type"]],
        str(row["suite"]),
        str(row["method"]),
        int(row["seed"]),
    ))
    confusion_rows = [
        {
            "suite": key[0],
            "method": key[1],
            "data_variant": key[2],
            "risk_label_true": key[3],
            "risk_label_pred": key[4],
            "count": count,
        }
        for key, count in sorted(confusion_groups.items())
    ]

    analysis_dir = output_root / "error_analysis"
    slice_fields = [
        "suite", "method", "data_variant", "slice_dimension",
        "slice_value", "n_runs", "n", "accuracy", "error_rate",
        "low_n", "far", "risk_n", "false_release_rate",
        "high_n", "high_miss_rate",
    ]
    confusion_fields = [
        "suite", "method", "data_variant",
        "risk_label_true", "risk_label_pred", "count",
    ]
    error_fields = [
        "suite", "method", "seed", "data_variant", "error_type",
        *preserved_fields, "result_file",
    ]
    _write_csv(
        analysis_dir / "slice_metrics.csv", slice_rows, slice_fields,
    )
    _write_csv(
        analysis_dir / "confusion_matrix.csv",
        confusion_rows,
        confusion_fields,
    )
    _write_csv(
        analysis_dir / "error_cases.csv", error_rows, error_fields,
    )
    _json_dump(
        analysis_dir / "analysis_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_suites": sorted(selected_suites),
            "detail_files_discovered": len(detail_paths),
            "records_analyzed": len(records),
            "slice_rows": len(slice_rows),
            "error_cases": len(error_rows),
            "skipped": skipped,
            "note": (
                "Rows are pooled across selected completed runs; n_runs "
                "reports the number of contributing result files."
            ),
        },
    )
    print(
        f"[analyze] detail_files={len(detail_paths)} "
        f"records={len(records)} slices={len(slice_rows)} "
        f"errors={len(error_rows)}"
    )
    print(f"[analyze] output: {analysis_dir.resolve()}")


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/tcr_net_paper.yaml")
    parser.add_argument(
        "--fixed-data-dir",
        default="data/revision_experiments/fixed_data",
    )
    parser.add_argument(
        "--data-root", default="data/revision_experiments",
    )
    parser.add_argument(
        "--output-root", default="exp_data/revision_2026",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--targeted-seeds", type=int, nargs="+",
        default=list(DEFAULT_TARGETED_SEEDS),
    )
    parser.add_argument(
        "--cross-seeds", type=int, nargs="+", default=[42],
    )
    parser.add_argument(
        "--generator-seeds", type=int, nargs="+",
        default=list(DEFAULT_GENERATOR_SEEDS),
    )
    parser.add_argument(
        "--history-windows", type=int, nargs="+",
        default=list(DEFAULT_HISTORY_WINDOWS),
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Run only the listed methods/settings; names must match the suite manifest exactly",
    )
    parser.add_argument(
        "--lambda-proto",
        type=float,
        default=None,
        help=(
            "Override only the prototype-anchor loss weight for TCR-style variants;"
            "used for validation-set candidate screening and does not change independent strong baselines"
        ),
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified runner for revised TCR-Net experiments",
    )
    parser.add_argument("--worker-task", default=None, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="Prepare revised experiment data")
    prepare.add_argument("--config", default="configs/tcr_net_paper.yaml")
    prepare.add_argument(
        "--data-root", default="data/revision_experiments",
    )
    prepare.add_argument(
        "--history-windows", type=int, nargs="+",
        default=list(DEFAULT_HISTORY_WINDOWS),
    )
    prepare.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
    )
    prepare.add_argument(
        "--generator-seeds", type=int, nargs="+",
        default=list(DEFAULT_GENERATOR_SEEDS),
    )
    prepare.add_argument("--base-generator-seed", type=int, default=42)
    prepare.add_argument("--scenario-train-samples", type=int, default=2048)
    prepare.add_argument("--scenario-val-samples", type=int, default=512)
    prepare.add_argument("--scenario-test-samples", type=int, default=1024)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--quick", action="store_true")

    run = subparsers.add_parser("run", help="Run the selected experiment suite")
    run.add_argument("--suite", choices=ALL_SUITES, required=True)
    _add_common_run_args(run)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate all experiment results")
    aggregate.add_argument(
        "--output-root", default="exp_data/revision_2026",
    )

    analyze = subparsers.add_parser(
        "analyze", help="Generate systematic error-analysis tables",
    )
    analyze.add_argument(
        "--output-root", default="exp_data/revision_2026",
    )
    analyze.add_argument(
        "--suites", nargs="+", default=["core", "cross_scenario"],
    )

    args = parser.parse_args()
    if args.worker_task:
        _worker_mode(args.worker_task)
        return
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
