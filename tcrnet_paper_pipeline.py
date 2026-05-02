#!/usr/bin/env python3
"""
TCR-Net  ——  train/evaluate/export/plot 。


----
    # 1. 
    python tcrnet_paper_pipeline.py generate --config configs/tcr_net_paper.yaml \\
        --mock-dir data/mock_station_v1

    # 2. 
    python tcrnet_paper_pipeline.py train --config configs/tcr_net_paper.yaml \\
        --mock-dir data/mock_station_v1 [--baselines] [--ablations] [--all] [--jobs 4]

    # 3.  CSV 
    python tcrnet_paper_pipeline.py export --config configs/tcr_net_paper.yaml \\
        --output-root exp_data/TCRNet_Final_Figures/tcr_paper

    # 4. 
    python tcrnet_paper_pipeline.py plot --baseline-summary <path> --out-dir <path>

    # 5. LaTeX 
    python tcrnet_paper_pipeline.py latex fill --tex paper.tex --report-data-dir <dir>
    python tcrnet_paper_pipeline.py latex insert --tex paper.tex --figure-dir <dir>

    # 6. 
    python tcrnet_paper_pipeline.py all --config ... --mock-dir ...

:
  generate    
  train       （: --single, --baselines, --ablations, --deep, --all）
  evaluate     checkpoint
  export       CSV 
  plot        
  latex       LaTeX （: fill, insert）
  all         generate + train + export + plot 


--------
。， tcrnet/ 
。（ _ ） tcrnet/__init__.py 。
"""

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


# ═══════════════════════════════════════════════════════════════════
# generate  
# ═══════════════════════════════════════════════════════════════════

def cmd_generate(args: argparse.Namespace) -> None:
    """..."""
    config = load_config(args.config)
    seed = int(config["seed"])
    mock_dir = Path(args.mock_dir)
    mock_dir.mkdir(parents=True, exist_ok=True)
    files = export_mock_dataset(config, seed=seed, output_dir=mock_dir)
    summary = {"seed": seed, "output_dir": str(mock_dir.resolve()), "files": files}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[generate]  {mock_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════
# train  
# ═══════════════════════════════════════════════════════════════════

def _is_complete(output_dir: str | Path) -> bool:
    """..."""
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
    """..."""
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
    """..."""
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
    """
"""
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
        print(f"[train] 。Test Macro-F1: {result.get('test', {}).get('macro_f1', 'N/A')}")
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
        print("[train] 。 --all / --baselines / --ablations / --deep")
        return

    print(json.dumps({"tasks": len(tasks), "jobs": jobs}, ensure_ascii=False))


    deep_tasks = [t for t in tasks if t["suite"] == "deep"]
    normal_tasks = [t for t in tasks if t["suite"] != "deep"]
    all_rows = []

    # ----  TCR-Net  ----
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
                            print(f"         : {err[:300]}")
                        print(f"         : {log_path}")
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

        #  summary
        baseline_rows = [r for r in all_rows if r["suite"] == "baselines"]
        ablation_rows = [r for r in all_rows if r["suite"] == "ablations"]
        _write_summary(output_root / "baselines" / "summary.json", mock_dir, baseline_rows)
        _write_summary(output_root / "ablations" / "summary.json", mock_dir, ablation_rows)

    # ----  ----
    if deep_tasks:
        print("[train]  ...")
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
            print(f"  => 。")
        except Exception as e:
            print(f"  => : {e}")
            import traceback
            traceback.print_exc()

    print(f"[train] 。: {output_root}")


# ═══════════════════════════════════════════════════════════════════
# evaluate   checkpoint
# ═══════════════════════════════════════════════════════════════════

def cmd_evaluate(args: argparse.Namespace) -> None:
    """..."""
    from tcrnet.training.trainer import evaluate as evaluate_model
    config = load_config(args.config)
    config["data"]["source"] = "station_jsonl"
    config["data"]["data_dir"] = str(Path(args.mock_dir).resolve())
    results = evaluate_model(config, checkpoint_path=args.checkpoint)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[evaluate] Test Macro-F1: {results.get('test', {}).get('macro_f1', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════
# export  
# ═══════════════════════════════════════════════════════════════════

def cmd_export(args: argparse.Namespace) -> None:
    """..."""
    from tcrnet._export_report import run_all_exports
    config = load_config(args.config)
    output_root = Path(args.output_root)
    detail_csv = args.detail_csv
    if detail_csv is None:
        #  TCR-Net  detail CSV tcr_net  tcr_net_full
        for cand_name in ["tcr_net", "tcr_net_full"]:
            cand = output_root / "baselines" / cand_name / "details" / "test_details.csv"
            if cand.exists():
                detail_csv = str(cand)
                break
        if detail_csv is None:
            print("[export] :  TCR-Net  test_details.csv，fig_5_12 ")
            detail_csv = ""
    #  mock  --mock-dir
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
    print(f"[export]  {result['report_data_dir']}")


# ═══════════════════════════════════════════════════════════════════
# plot  
# ═══════════════════════════════════════════════════════════════════

def cmd_plot(args: argparse.Namespace) -> None:
    """..."""
    from tcrnet._plot_results import plot_all_figures

    #  report_data 
    if args.report_data_dir:
        report_dir = args.report_data_dir
    elif args.baseline_summary:
        report_dir = str(Path(args.baseline_summary).resolve().parent.parent / "report_data")
    else:
        print("[plot]  --report-data-dir  --baseline-summary")
        return

    #  detail CSV fig4 
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
    print(f"[plot] 。 {len(generated)}  {args.out_dir}")


# ═══════════════════════════════════════════════════════════════════
# latex  LaTeX 
# ═══════════════════════════════════════════════════════════════════

def cmd_latex(args: argparse.Namespace) -> None:
    """..."""
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
        print(f"[latex fill]  {result['tables_replaced']} , "
              f"{result['figures_replaced']} , "
              f"{result['captions_updated']} ")
    elif args.latex_cmd == "insert":
        count = insert_figures(
            tex_path=args.tex,
            figure_dir=args.figure_dir,
            ext="pdf",
        )
        print(f"[latex insert]  {count} ")
    else:
        print(f"[latex] : {args.latex_cmd}")


# ═══════════════════════════════════════════════════════════════════
# all  
# ═══════════════════════════════════════════════════════════════════

def cmd_all(args: argparse.Namespace) -> None:
    """..."""
    output_root = Path(args.output_root or "exp_data/TCRNet_Final_Figures/tcr_paper")
    figure_dir = Path(args.figure_dir or "exp_data/TCRNet_Final_Figures/figures")

    print("=" * 60)
    print("TCR-Net ")
    print("=" * 60)

    # Step 1: 
    print("\n[Step 1/4]  ...")
    gen_args = argparse.Namespace(config=args.config, mock_dir=args.mock_dir)
    cmd_generate(gen_args)

    # Step 2: 
    print("\n[Step 2/4]  ...")
    train_args = argparse.Namespace(
        config=args.config, mock_dir=args.mock_dir,
        single=False, all=True, baselines=False, ablations=False, deep=True,
        skip_deep=False, output_root=str(output_root), jobs=args.jobs,
        force=args.force, output_dir=None,
    )
    cmd_train(train_args)

    # Step 3: 
    print("\n[Step 3/4]  ...")
    export_args = argparse.Namespace(
        config=args.config, output_root=str(output_root),
        detail_csv=None, mock_dir=args.mock_dir,
    )
    cmd_export(export_args)

    # Step 4: 
    print("\n[Step 4/4]  ...")
    plot_args = argparse.Namespace(
        report_data_dir=str(output_root / "report_data"),
        baseline_summary=str(output_root / "baselines" / "summary.json"),
        out_dir=str(figure_dir),
        detail_csv=str(output_root / "baselines" / "tcr_net_full" / "details" / "test_details.csv"),
    )
    cmd_plot(plot_args)

    print("\n" + "=" * 60)
    print(f"！")
    print(f"  : {output_root}")
    print(f"  : {figure_dir}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# Worker 
# ═══════════════════════════════════════════════════════════════════

def _worker_mode(config_path: str) -> None:
    """..."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = _train_worker(payload["task"])
    Path(payload["result_path"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ["suite", "name", "status", "output_dir", "error"] if k in result}))
    if result["status"] == "failed":
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCR-Net ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
:
  generate    
  train       （--single / --baselines / --ablations / --deep / --all）
  evaluate     checkpoint
  export       CSV 
  plot        
  latex       LaTeX （fill | insert）
  all         generate + train + export + plot 

:

  python tcrnet_paper_pipeline.py all --config configs/tcr_net_paper.yaml

  #  baseline 
  python tcrnet_paper_pipeline.py train --config configs/tcr_net_paper.yaml --baselines


  python tcrnet_paper_pipeline.py export --output-root exp_data/TCRNet_Final_Figures/tcr_paper
  python tcrnet_paper_pipeline.py plot --baseline-summary exp_data/TCRNet_Final_Figures/tcr_paper/baselines/summary.json --out-dir figures
        """,
    )
    parser.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")

    # generate
    p = sub.add_parser("generate", help="")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")

    # train
    p = sub.add_parser("train", help="")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--output-root", default=None)
    p.add_argument("--output-dir", default=None, help="")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--single", action="store_true", help="（ --output-dir）")
    p.add_argument("--baselines", action="store_true", help=" 7  baseline ")
    p.add_argument("--ablations", action="store_true", help=" 6  ablation ")
    p.add_argument("--deep", action="store_true", help=" 4 ")
    p.add_argument("--skip-deep", action="store_true", help="")
    p.add_argument("--all", action="store_true", help="（baselines + ablations + deep）")

    # evaluate
    p = sub.add_parser("evaluate", help=" checkpoint")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--checkpoint", required=True)

    # export
    p = sub.add_parser("export", help=" CSV ")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--output-root", required=True)
    p.add_argument("--detail-csv", default=None)
    p.add_argument("--mock-dir", default=None)

    # plot
    p = sub.add_parser("plot", help="")
    p.add_argument("--report-data-dir", default=None)
    p.add_argument("--baseline-summary", default=None)
    p.add_argument("--detail-csv", default=None)
    p.add_argument("--out-dir", default="outputs/figures")

    # latex
    p = sub.add_parser("latex", help="LaTeX ")
    p.add_argument("latex_cmd", choices=["fill", "insert"])
    p.add_argument("--tex", required=True)
    p.add_argument("--report-data-dir", default=None)
    p.add_argument("--figure-source-dir", default=None)
    p.add_argument("--figure-dest-dir", default=None)
    p.add_argument("--latex-figure-dir", default="figures_auto")
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--compact", action="store_true", help="（/）")

    # all
    p = sub.add_parser("all", help="")
    p.add_argument("--config", default="configs/tcr_net_paper.yaml")
    p.add_argument("--mock-dir", default="data/mock_station_v1")
    p.add_argument("--output-root", default=None)
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")

    args = parser.parse_args()

    # Worker 
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
