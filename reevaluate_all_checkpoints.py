


from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch


def _resolve_evaluator(cfg: dict, ckpt_path: Path):

    try:
        from tcrnet import evaluate as evaluate_fn
        evaluator_name = "tcrnet"
    except ImportError:
        evaluator_name = None


    if evaluator_name == "tcrnet":
        try:

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            has_unified = any("unified_head" in k for k in ckpt.get("model_state", {}).keys())
            if has_unified:

                from tcrnet import evaluate as eval_fn
                return eval_fn, "tcrnet"
            else:

                from tcrnet.training.trainer import evaluate as eval_fn
                return _tcrnet_compat_evaluate, "tcrnet_compat"
        except Exception:
            pass


    from cctnet.tcrnet_paper_trainer import evaluate_tcrnet_paper as eval_fn
    return eval_fn, "cctnet"


def _tcrnet_compat_evaluate(cfg: dict, checkpoint_path: str):

    from tcrnet import TCRNet, load_config
    from tcrnet.data.synthetic import build_dataloaders
    from tcrnet.training.trainer import (
        _collect_split_outputs, _apply_decision_calibration,
        _summarize, _export_details,
    )
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_l, val_l, test_l, meta = build_dataloaders(cfg, int(cfg["seed"]))
    model = TCRNet(cfg, meta).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt["model_state"]


    if not any("unified_head" in k for k in model_state):
        own_state = model.state_dict()
        for k in own_state:
            if "unified_head" in k and k not in model_state:
                model_state[k] = own_state[k]

    model.load_state_dict(model_state, strict=False)
    stats = ckpt["score_stats"]
    calib = ckpt.get("decision_calibration", {
        "mix_logit": 0.0, "component_weights": [1/3, 1/3, 1/3],
        "eta_1": float(ckpt["thresholds"]["eta_1"]),
        "eta_2": float(ckpt["thresholds"]["eta_2"]),
    })
    eta = {"eta_1": float(calib["eta_1"]), "eta_2": float(calib["eta_2"])}

    results = {}
    details_dir = Path(cfg["output_dir"]) / "details"
    for split, loader in [("train", train_l), ("val", val_l), ("test", test_l)]:
        payload = _collect_split_outputs(model, loader, device, cfg)
        score, pred = _apply_decision_calibration(payload, stats, calib)
        m = _summarize(payload["y3"], pred)
        m.update({
            "avg_R_cons": float(payload["R_cons"].mean()),
            "avg_R_seq": float(payload["R_seq"].mean()),
            "avg_R_rule": float(payload["R_rule"].mean()),
            "avg_R_total": float(score.mean()),
            "thresholds": {"eta_1": eta["eta_1"], "eta_2": eta["eta_2"]},
        })
        results[split] = m
        _export_details(split, details_dir, payload, score, pred, eta)

    results["thresholds"] = {"eta_1": eta["eta_1"], "eta_2": eta["eta_2"]}
    results["decision_calibration"] = {
        "mix_logit": float(calib["mix_logit"]),
        "component_weights": [float(x) for x in calib["component_weights"]],
    }
    return results


def _iter_targets(output_root: Path) -> Iterable[tuple[str, Path]]:
    full = output_root / "full"
    if (full / "best_tcr.pt").exists():
        yield "Full", full
    for parent_name in ("baselines", "ablations"):
        parent = output_root / parent_name
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            if (child / "best_tcr.pt").exists():
                yield f"{parent_name}/{child.name}", child


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate every TCR-Net checkpoint.")
    parser.add_argument("--config", default="configs/tcr_net_paper.yaml")
    parser.add_argument("--mock-dir", default="data/mock_station_v1")
    parser.add_argument("--output-root", default="exp_data/TCRNet_Final_Figures/tcr_paper")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    targets = list(_iter_targets(output_root))
    print(json.dumps({
        "output_root": str(output_root),
        "targets": [name for name, _ in targets],
    }, ensure_ascii=False, indent=2))

    for name, ckpt_dir in targets:
        ckpt_path = ckpt_dir / "best_tcr.pt"
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = deepcopy(ckpt_blob["config"])
        cfg["data"]["source"] = "station_jsonl"
        cfg["data"]["data_dir"] = args.mock_dir
        cfg["output_dir"] = str(ckpt_dir)

        print(f"[reevaluate] {name} -> {ckpt_path}")
        try:
            eval_fn, engine = _resolve_evaluator(cfg, ckpt_path)
            results = eval_fn(cfg, checkpoint_path=str(ckpt_path))
            (ckpt_dir / "metrics_tcr.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print(f"[reevaluate] OK ({engine})")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[reevaluate] FAILED {name}: {exc!r}")


if __name__ == "__main__":
    main()
