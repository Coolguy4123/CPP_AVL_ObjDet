"""Step 2: choose the fusion settings on the tuning split and save fused detections.

Reads    runs/fusion/detections/{ground_truth,thermal,rgb}.npz   (from detect.py)
Writes   runs/fusion/fusion_config.json     best config + split used + inputs
         runs/fusion/tuning_log.csv         every config tried
         runs/fusion/detections/fusion.npz  fused detections for ALL pairs

Only the tuning split is used for the search; evaluate.py reports on the
held-out test split.  The staged search is:
    1. method (NMS, Soft-NMS, WBF avg/max/box_avg) x IoU threshold
    2. thermal:RGB weight ratio
    3. score floor before fusion
    4. IoU refinement around the best
scored by macro mAP over the classes that have GT.

Usage
    python Testing_Scripts/fusion/tune_fusion.py
    python Testing_Scripts/fusion/tune_fusion.py --config runs/fusion/fusion_config.json   # re-fuse only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fusion_common import (CONFIG_FILE, DEFAULT_SEED, DEFAULT_TEST_FRAC, GT_FILE, OUT_DIR,
                           class_models_for, config_label, det_file, fuse_image, load_detections,
                           load_records, macro_map, make_split, match_dataset, save_records)


# Worker state, loaded once per process (avoids pickling the detections per task).
_W: dict = {}


def _init_worker(ids: list[int]) -> None:
    gt = load_records(GT_FILE)
    thermal = load_detections(det_file("thermal"), gt["stems"])
    rgb = load_detections(det_file("rgb"), gt["stems"])
    class_models = class_models_for([thermal["class_names"], rgb["class_names"]])
    _W.update(
        ids=ids,
        gts=[gt["records"][i] for i in ids],
        thermal=[thermal["records"][i] for i in ids],
        rgb=[rgb["records"][i] for i in ids],
        class_models=class_models,
        classes=sorted({n for r in gt["records"] for n in r["names"]} & set(class_models)),
    )


def _score(cfg: dict) -> dict:
    start = time.time()
    fused = [fuse_image([t, r], cfg, _W["class_models"]) for t, r in zip(_W["thermal"], _W["rgb"])]
    matches = match_dataset(fused, _W["gts"], _W["classes"], list(range(len(fused))))
    map50, map50_95 = macro_map(matches)
    return {"cfg": cfg, "map50": map50, "map50_95": map50_95, "seconds": time.time() - start}


def search(pool, metric: str, log: list[dict]) -> dict:
    def run(cfgs: list[dict], best: dict | None = None) -> dict:
        """Score cfgs in parallel; return the best of them and the incoming best."""
        for res in pool.map(_score, cfgs):
            log.append({"config": config_label(res["cfg"]), "map50": res["map50"],
                        "map50_95": res["map50_95"], **res["cfg"]})
            print(f"  {config_label(res['cfg']):<64} mAP50={res['map50']:.4f} "
                  f"mAP50-95={res['map50_95']:.4f} ({res['seconds']:.0f}s)")
            if best is None or res[metric] > best[metric]:
                best = res
        return best

    base = {"weights": (1.0, 1.0), "skip_thr": 0.001, "conf_type": "avg"}
    print("Stage 1: method x IoU threshold (equal weights)")
    stage1 = []
    for iou in (0.5, 0.55, 0.6, 0.7):
        stage1.append({**base, "method": "nms", "iou_thr": iou})
        stage1.append({**base, "method": "soft_nms", "iou_thr": iou})
        for conf_type in ("avg", "max", "box_avg"):
            stage1.append({**base, "method": "wbf", "iou_thr": iou, "conf_type": conf_type})
    best = run(stage1)

    print("Stage 2: thermal:RGB weights")
    ratios = [(wt, wr) for wt, wr in itertools.product((1.0, 1.5, 2.0, 3.0), repeat=2)
              if (wt, wr) != (1.0, 1.0) and not (wt > 1 and wr > 1)]
    best = run([{**best["cfg"], "weights": w} for w in ratios], best)

    print("Stage 3: score floor")
    best = run([{**best["cfg"], "skip_thr": s} for s in (0.01, 0.03, 0.05, 0.1)], best)

    print("Stage 4: IoU refinement")
    ious = sorted({round(best["cfg"]["iou_thr"] + d, 3) for d in (-0.05, -0.025, 0.025, 0.05)})
    ious = [i for i in ious if 0.3 <= i <= 0.9]
    best = run([{**best["cfg"], "iou_thr": i} for i in ious], best)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split seed")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC, help="held-out fraction")
    parser.add_argument("--metric", choices=("map50", "map50_95"), default="map50_95")
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--config", type=Path, default=None,
                        help="skip the search and fuse with this fusion_config.json")
    args = parser.parse_args()

    gt = load_records(GT_FILE)
    thermal = load_detections(det_file("thermal"), gt["stems"])
    rgb = load_detections(det_file("rgb"), gt["stems"])
    class_models = class_models_for([thermal["class_names"], rgb["class_names"]])

    if args.config:
        saved = json.loads(args.config.read_text(encoding="utf-8"))
        best_cfg = saved["config"]
        best_cfg["weights"] = tuple(best_cfg["weights"])
        record = saved
        print(f"Using {args.config}: {config_label(best_cfg)}")
    else:
        split = make_split(len(gt["stems"]), args.seed, args.test_frac)
        print(f"{len(gt['stems'])} pairs -> tuning on {len(split['tune'])}, "
              f"{len(split['test'])} held out for evaluate.py | {args.jobs} worker(s)")
        log: list[dict] = []
        start = time.time()
        with ProcessPoolExecutor(args.jobs, initializer=_init_worker, initargs=(split["tune"],)) as pool:
            best = search(pool, args.metric, log)
        best_cfg = best["cfg"]
        print(f"\nBest on tuning split: {config_label(best_cfg)}  "
              f"mAP50={best['map50']:.4f} mAP50-95={best['map50_95']:.4f}  "
              f"({(time.time() - start) / 60:.1f} min)")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "tuning_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log[0]))
            writer.writeheader()
            writer.writerows(log)
        record = {
            "config": {**best_cfg, "weights": list(best_cfg["weights"])},
            "label": config_label(best_cfg),
            "tune_metric": args.metric,
            "tune_scores": {"map50": best["map50"], "map50_95": best["map50_95"]},
            "split": {"seed": args.seed, "test_frac": args.test_frac,
                      "n_tune": len(split["tune"]), "n_test": len(split["test"])},
            "inputs": {"thermal": thermal["meta"], "rgb": rgb["meta"]},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        CONFIG_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Saved {CONFIG_FILE} and {OUT_DIR / 'tuning_log.csv'}")

    print("Fusing all pairs with the chosen config")
    fused = [fuse_image([t, r], best_cfg, class_models)
             for t, r in zip(thermal["records"], rgb["records"])]
    out = det_file("fusion")
    save_records(out, gt["stems"], fused, list(class_models),
                 {"modality": "fusion", "config": record["config"], "label": config_label(best_cfg),
                  "split": record.get("split"), "inputs": record.get("inputs"),
                  "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
