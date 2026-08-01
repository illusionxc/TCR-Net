


from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from tcrnet import load_config, baseline_variants, ablation_variants, train
from tcrnet.data.synthetic import export_mock_dataset, build_dataloaders


def cmd_generate(args: argparse.Namespace) -> None:

    config = load_config(args.config)
    seed = int(config["seed"])
    mock_dir = Path(args.mock_dir)
    mock_dir.mkdir(parents=True, exist_ok=True)
    files = export_mock_dataset(config, seed=seed, output_dir=mock_dir)
    summary = {"seed": seed, "output_dir": str(mock_dir.resolve()), "files": files}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[generate] Dataset saved to {mock_dir.resolve()}")


def _is_complete(output_dir: str | Path) -> bool:

    out = Path(output_dir)
    return (out / "best.pt").exists() or (out / "best_tcr.pt").exists()


def _load_metrics(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    for name in ("metrics.json", "metrics_tcr.json"):
        p = out / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _retarget_output_dirs(variants: dict, output_root: Path, suite: str) -> dict:

    retargeted = {}
    for name, cfg in variants.items():
        c = deepcopy(cfg)
        out = Path(str(c["output_dir"]))
        parts = out.parts
        if "baselines" in parts:
            suffix = parts[parts.index("baselines") + 1:]
        elif "ablations" in parts:
            suffix = parts[parts.index("ablations") + 1:]
        else:
            suffix = (out.name,)
        c["output_dir"] = str(output_root / suite / Path(*suffix))
        retargeted[name] = c
    return retargeted


def _train_worker(task: dict) -> dict:

    name = task["name"]
    suite = task["suite"]
    cfg = task["config"]
    force = bool(task.get("force", False))
    output_dir = Path(cfg["output_dir"])

    try:
        if not force and _is_complete(output_dir):
            metrics = _load_metrics(output_dir)
            return {"name": name, "suite": suite, "output_dir": str(output_dir), "status": "skipped", "metrics": metrics}
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = train(cfg)
        return {"name": name, "suite": suite, "output_dir": str(output_dir), "status": "trained", "metrics": metrics}
    except Exception as exc:
        return {"name": name, "suite": suite, "output_dir": str(output_dir), "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def _write_summary(path: Path, mock_dir: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for row in rows:
        if row["status"] == "failed":
            continue
        experiments[row["name"]] = {"output_dir": row["output_dir"], "metrics": row["metrics"], "status": row["status"]}
    path.write_text(json.dumps({"mock_dir": str(mock_dir.resolve()), "experiments": experiments}, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> None:


    config = load_config(args.config)
    mock_dir = Path(args.mock_dir)


    if args.single:
        cfg = deepcopy(config)
        cfg["data"]["source"] = "station_jsonl"
        cfg["data"]["data_dir"] = str(mock_dir.resolve())
        if args.output_dir:
            cfg["output_dir"] = args.output_dir
        result = train(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[train] Training finished. Test Macro-F1: {result.get('test', {}).get('macro_f1', 'N/A')}")
        return


    base_cfg = deepcopy(config)
    base_cfg["data"]["source"] = "station_jsonl"
    base_cfg["data"]["data_dir"] = str(mock_dir.resolve())

    output_root = Path(args.output_root or "exp_data/TCRNet_Final_Figures/tcr_paper")
    jobs = max(1, int(args.jobs or 4))
    force = args.force


    tasks = []
    want_baselines = args.all or args.baselines
    want_ablations = args.all or args.ablations
    want_deep = args.all or (args.deep and not args.skip_deep)

    if want_baselines:
        for name, cfg in _retarget_output_dirs(baseline_variants(base_cfg), output_root, "baselines").items():
            tasks.append({"name": name, "suite": "baselines", "config": cfg, "force": force})
    if want_ablations:
        for name, cfg in _retarget_output_dirs(ablation_variants(base_cfg), output_root, "ablations").items():
            tasks.append({"name": name, "suite": "ablations", "config": cfg, "force": force})
    if want_deep:
        tasks.append({"name": "deep_baselines", "suite": "deep", "config": base_cfg, "force": force})

    if not tasks:
        print("[train] No training mode selected. Use --all / --baselines / --ablations / --deep")
        return

    print(json.dumps({"tasks": len(tasks), "jobs": jobs}, ensure_ascii=False))


    deep_tasks = [t for t in tasks if t["suite"] == "deep"]
    normal_tasks = [t for t in tasks if t["suite"] != "deep"]
    all_rows = []


    if normal_tasks:
        from tcrnet_paper_pipeline import _train_worker as worker_fn

        if jobs > 1:

            run_dir = Path(f"runs/parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            task_dir = run_dir / "tasks"
            result_dir = run_dir / "results"
            log_dir = run_dir / "logs"
            task_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            pending = []
            for idx, task in enumerate(normal_tasks):
                safe_name = f"{idx:02d}_{task['suite']}_{task['name']}".replace("/", "_").replace(" ", "_")
                tp = task_dir / f"{safe_name}.json"
                rp = result_dir / f"{safe_name}.json"
                tp.write_text(json.dumps({"task": task, "result_path": str(rp)}), encoding="utf-8")
                pending.append({"task": task, "result_path": rp, "log_path": log_dir / f"{safe_name}.log", "safe_name": safe_name})

            running = []
            while pending or running:
                while pending and len(running) < jobs:
                    item = pending.pop(0)
                    lf = item["log_path"].open("w", encoding="utf-8")
                    proc = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(task_dir / f"{item['safe_name']}.json")],
                        stdout=lf, stderr=subprocess.STDOUT, text=True,
                        cwd=os.getcwd(),
                    )
                    item["proc"] = proc
                    item["log_f"] = lf
                    running.append(item)
                    print(f"  [start] {item['task']['name']} (pid={proc.pid})")
                still = []
                for item in running:
                    if item["proc"].poll() is None:
                        still.append(item)
                        continue
                    item["log_f"].close()
                    if item["result_path"].exists():
                        row = json.loads(item["result_path"].read_text(encoding="utf-8"))
                    else:
                        row = {"name": item["task"]["name"], "suite": item["task"]["suite"], "output_dir": item["task"]["config"]["output_dir"], "status": "failed", "error": f"worker exited unexpectedly"}
                    all_rows.append(row)
                    status = row.get("status", "unknown")
                    err = row.get("error", "")
                    log_path = item["log_path"]
                    if status == "failed":
                        print(f"  [done] {row.get('name')}: FAILED")
                        if err:
                            print(f"         Error: {err[:300]}")
                        print(f"         Log: {log_path}")
                    else:
                        print(f"  [done] {row.get('name')}: {status}")
                running = still
                if pending or running:
                    time.sleep(2.0)
        else:

            for task in normal_tasks:
                row = _train_worker(task)
                all_rows.append(row)
                status = row.get("status", "unknown")
                err = row.get("error", "")
                if status == "failed" and err:
                    print(f"  [done] {task['name']}: FAILED ({err[:200]})")
                else:
                    print(f"  [done] {task['name']}: {status}")


        baseline_rows = [r for r in all_rows if r["suite"] == "baselines"]
        ablation_rows = [r for r in all_rows if r["suite"] == "ablations"]
        _write_summary(output_root / "baselines" / "summary.json", mock_dir, baseline_rows)
        _write_summary(output_root / "ablations" / "summary.json", mock_dir, ablation_rows)


    if deep_tasks:
        print("[train] Running deep baselines ...")
        try:
            from tcrnet.models.baselines import run_all_baselines
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dl_cfg = deepcopy(deep_tasks[0]["config"])
            dl_cfg["data"]["source"] = "station_jsonl"
            dl_cfg["data"]["data_dir"] = str(mock_dir.resolve())
            train_l, val_l, test_l, meta = build_dataloaders(dl_cfg, int(dl_cfg["seed"]))
            deep_results = run_all_baselines(dl_cfg, train_l, val_l, test_l, device)
            (output_root / "baselines" / "deep_baselines_summary.json").write_text(
                json.dumps(deep_results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  => Deep baselines finished.")
        except Exception as e:
            print(f"  => Deep baselines failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"[train] Training finished. Output directory: {output_root}")


def cmd_evaluate(args: argparse.Namespace) -> None:

    from tcrnet.training.trainer import evaluate as evaluate_model
    config = load_config(args.config)
    config["data"]["source"] = "station_jsonl"
    config["data"]["data_dir"] = str(Path(args.mock_dir).resolve())
    results = evaluate_model(config, checkpoint_path=args.checkpoint)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[evaluate] Test Macro-F1: {results.get('test', {}).get('macro_f1', 'N/A')}")


def cmd_export(args: argparse.Namespace) -> None:

    from tcrnet._export_report import run_all_exports
    config = load_config(args.config)
    output_root = Path(args.output_root)
    detail_csv = args.detail_csv
    if detail_csv is None:

        for cand_name in ["tcr_net", "tcr_net_full"]:
            cand = output_root / "baselines" / cand_name / "details" / "test_details.csv"
            if cand.exists():
                detail_csv = str(cand)
                break
        if detail_csv is None:
            print("[export] Warning: TCR-Net test_details.csv was not found; fig_5_12 will be skipped")
            detail_csv = ""

    mock_dir = args.mock_dir
    if mock_dir is None:
        mock_dir = config["data"].get("data_dir", "data/mock_station_v1")
    mock_dir = str(Path(mock_dir).resolve())
    result = run_all_exports(
        config_path=args.config,
        output_root=args.output_root,
        detail_csv=detail_csv,
        mock_data_dir=mock_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[export] Reports exported to {result['report_data_dir']}")


def cmd_plot(args: argparse.Namespace) -> None:

    from tcrnet._plot_results import plot_all_figures


    if args.report_data_dir:
        report_dir = args.report_data_dir
    elif args.baseline_summary:
        report_dir = str(Path(args.baseline_summary).resolve().parent.parent / "report_data")
    else:
        print("[plot] Specify --report-data-dir or --baseline-summary")
        return


    detail_csv = args.detail_csv
    if not detail_csv:
        for cand_name in ["tcr_net", "tcr_net_full"]:
            cand = Path(report_dir).parent / "baselines" / cand_name / "details" / "test_details.csv"
            if cand.exists():
                detail_csv = str(cand)
                break

    generated = plot_all_figures(
        report_data_dir=report_dir,
        output_dir=args.out_dir,
        detail_csv=detail_csv or None,
    )
    print(f"[plot] Finished. Generated {len(generated)} figures in {args.out_dir}")


def cmd_latex(args: argparse.Namespace) -> None:

    from tcrnet._latex_tools import fill_tables_and_figures, insert_figures

    if args.latex_cmd == "fill":
        result = fill_tables_and_figures(
            tex_path=args.tex,
            report_data_dir=args.report_data_dir,
            figure_source_dir=args.figure_source_dir,
            figure_dest_dir=args.figure_dest_dir,
            latex_figure_dir=args.latex_figure_dir,
            compact=args.compact,
        )
        print(f"[latex fill] Replaced {result['tables_replaced']} tables, "
              f"{result['figures_replaced']} figures, "
              f"{result['captions_updated']} captions")
    elif args.latex_cmd == "insert":
        count = insert_figures(
            tex_path=args.tex,
            figure_dir=args.figure_dir,
            ext="pdf",
        )
        print(f"[latex insert] Replaced {count} figure blocks")
    else:
        print(f"[latex] Unknown subcommand: {args.latex_cmd}")


def cmd_all(args: argparse.Namespace) -> None:

    output_root = Path(args.output_root or "exp_data/TCRNet_Final_Figures/tcr_paper")
    figure_dir = Path(args.figure_dir or "exp_data/TCRNet_Final_Figures/figures")

    print("=" * 60)
    print("TCR-Net full paper experiment pipeline")
    print("=" * 60)


    print("\n[Step 1/4] Generate synthetic data ...")
    gen_args = argparse.Namespace(config=args.config, mock_dir=args.mock_dir)
    cmd_generate(gen_args)


    print("\n[Step 2/4] Train models ...")
    train_args = argparse.Namespace(
        config=args.config, mock_dir=args.mock_dir,
        single=False, all=True, baselines=False, ablations=False, deep=True,
        skip_deep=False, output_root=str(output_root), jobs=args.jobs,
        force=args.force, output_dir=None,
    )
    cmd_train(train_args)


    print("\n[Step 3/4] Export experiment data ...")
    export_args = argparse.Namespace(
        config=args.config, output_root=str(output_root),
        detail_csv=None, mock_dir=args.mock_dir,
    )
    cmd_export(export_args)


    print("\n[Step 4/4] Generate figures ...")
    plot_args = argparse.Namespace(
        report_data_dir=str(output_root / "report_data"),
        baseline_summary=str(output_root / "baselines" / "summary.json"),
        out_dir=str(figure_dir),
        detail_csv=str(output_root / "baselines" / "tcr_net_full" / "details" / "test_details.csv"),
    )
    cmd_plot(plot_args)

    print("\n" + "=" * 60)
    print(f"Full pipeline finished.")
    print(f"  Models: {output_root}")
    print(f"  Figures: {figure_dir}")
    print("=" * 60)


def _worker_mode(config_path: str) -> None:

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = _train_worker(payload["task"])
    Path(payload["result_path"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ["suite", "name", "status", "output_dir", "error"] if k in result}))
    if result["status"] == "failed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for TCR-Net paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  generate    Generate the synthetic dataset
  train       Train models (--single / --baselines / --ablations / --deep / --all)
  evaluate    Evaluate a single checkpoint
  export      Export experiment data as CSV reports
  plot        Generate manuscript figures
  latex       LaTeX integration (fill | insert)
  all         generate + train + export + plot

Examples:
  python tcrnet_paper_pipeline.py all --config configs/tcr_net_paper.yaml

  python tcrnet_paper_pipeline.py train --config configs/tcr_net_paper.yaml --baselines

  python tcrnet_paper_pipeline.py export --output-root exp_data/TCRNet_Final_Figures/tcr_paper
  python tcrnet_paper_pipeline.py plot --baseline-summary exp_data/TCRNet_Final_Figures/tcr_paper/baselines/summary.json --out-dir figures
        """,
    )
    parser.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")


    p = sub.add_parser("generate", help="Generate the synthetic dataset")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")


    p = sub.add_parser("train", help="Train models")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--output-root", default=None)
    p.add_argument("--output-dir", default=None, help="Output directory for single-model mode")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--single", action="store_true", help="Train a single model; requires --output-dir")
    p.add_argument("--baselines", action="store_true", help="Train seven baseline variants")
    p.add_argument("--ablations", action="store_true", help="Train six ablation variants")
    p.add_argument("--deep", action="store_true", help="Train four deep baselines")
    p.add_argument("--skip-deep", action="store_true", help="Skip deep baselines")
    p.add_argument("--all", action="store_true", help="Train all variants: baselines, ablations, and deep baselines")


    p = sub.add_parser("evaluate", help="Evaluate a single checkpoint")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--checkpoint", required=True)


    p = sub.add_parser("export", help="Export experiment data as CSV reports")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--output-root", required=True)
    p.add_argument("--detail-csv", default=None)
    p.add_argument("--mock-dir", default=None)


    p = sub.add_parser("plot", help="Generate manuscript figures")
    p.add_argument("--report-data-dir", default=None)
    p.add_argument("--baseline-summary", default=None)
    p.add_argument("--detail-csv", default=None)
    p.add_argument("--out-dir", default="outputs/figures")


    p = sub.add_parser("latex", help="LaTeX integration")
    p.add_argument("latex_cmd", choices=["fill", "insert"])
    p.add_argument("--tex", required=True)
    p.add_argument("--report-data-dir", default=None)
    p.add_argument("--figure-source-dir", default=None)
    p.add_argument("--figure-dest-dir", default=None)
    p.add_argument("--latex-figure-dir", default="figures_auto")
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--compact", action="store_true", help="Compact mode; remove secondary tables and figures")


    p = sub.add_parser("all", help="Full pipeline")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--output-root", default=None)
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")

    args = parser.parse_args()


    if args.worker_config:
        _worker_mode(args.worker_config)
        return


    route = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "plot": cmd_plot,
        "latex": cmd_latex,
        "all": cmd_all,
    }
    if args.command in route:
        route[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
