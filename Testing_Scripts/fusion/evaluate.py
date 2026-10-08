"""Step 3: per-class evaluation of thermal, RGB and fused detections.

Reads    runs/fusion/detections/*.npz          (from detect.py / tune_fusion.py)
Writes   runs/fusion/eval/<split>/summary.md   tables ready to quote
         runs/fusion/eval/<split>/per_class.csv
         runs/fusion/eval/<split>/comparisons.csv
         runs/fusion/eval/<split>/other_classes.csv
         runs/fusion/eval/<split>/results.json

Every class with ground truth is evaluated on its own, next to its GT box/image
count:
    AP50, AP75, AP50-95                COCO-style, 101-point interpolation
    95% CI for AP50 / AP50-95          bootstrap over images (--bootstrap draws)
    P, R, F1, TP, FP, FN               at a confidence threshold picked on the
                                       *tuning* split (best F1, IoU 0.5) and
                                       applied unchanged to the evaluated split
    paired differences                 e.g. fusion - thermal, same bootstrap draws,
                                       with 95% CI and P(diff <= 0)
Classes without GT get detection counts only.

The split must match the one used by tune_fusion.py; it is read from
runs/fusion/fusion_config.json when present.  Report the "test" split -- the
fusion settings were chosen on "tune", so "all" and "tune" are optimistic for
fusion.

Usage
    python Testing_Scripts/fusion/evaluate.py
    python Testing_Scripts/fusion/evaluate.py --split all --bootstrap 2000
    python Testing_Scripts/fusion/evaluate.py --models thermal rgb      # before tuning
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import json
import time
from pathlib import Path

import numpy as np

from fusion_common import (CONFIG_FILE, DEFAULT_SEED, DEFAULT_TEST_FRAC, EVAL_DIR, GT_FILE,
                           average_precision, best_f1_conf, det_file, load_detections,
                           load_records, make_split, match_dataset, operating_point)


def fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{digits}f}"


def fmt_ci(v: float | None, lo: float | None, hi: float | None, signed: bool = False) -> str:
    if v is None or np.isnan(v):
        return "n/a"
    f = "+.3f" if signed else ".3f"
    return f"{v:{f}} [{lo:{f}}, {hi:{f}}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("test", "tune", "all"), default="test")
    parser.add_argument("--models", nargs="+", default=None,
                        help="detection files in runs/fusion/detections (default: thermal rgb fusion, if present)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap draws for CIs (0 = off)")
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--count-conf", type=float, default=0.25,
                        help="confidence for detection counts of classes without GT")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which cannot print the Δ in the tables.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- inputs -----------------------------------------------------------
    gt = load_records(GT_FILE)
    stems, gts = gt["stems"], gt["records"]
    names = args.models or [m for m in ("thermal", "rgb", "fusion") if det_file(m).exists()]
    if not names:
        raise SystemExit("No detection files found; run detect.py first.")
    dets = {m: load_detections(det_file(m), stems) for m in names}

    seed, test_frac, fusion_label = DEFAULT_SEED, DEFAULT_TEST_FRAC, None
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        seed, test_frac, fusion_label = cfg["split"]["seed"], cfg["split"]["test_frac"], cfg["label"]
    split = make_split(len(stems), seed, test_frac)
    ids = split[args.split]
    thr_ids = split["tune"] if args.split == "test" else ids
    thr_source = "tune split" if args.split == "test" else f"{args.split} split (same as evaluated)"

    # ---- ground truth per class ------------------------------------------
    gt_boxes: dict[str, int] = {}
    gt_images: dict[str, int] = {}
    for i in ids:
        for c in set(gts[i]["names"]):
            gt_images[c] = gt_images.get(c, 0) + 1
        for c in gts[i]["names"]:
            gt_boxes[c] = gt_boxes.get(c, 0) + 1
    classes = sorted(gt_boxes, key=lambda c: -gt_boxes[c])
    trained = {m: set(d["class_names"]) for m, d in dets.items()}
    common = [c for c in classes if all(c in trained[m] for m in names)]

    print(f"Split: {args.split} ({len(ids)} pairs, seed={seed}, test_frac={test_frac})")
    print(f"Models: {names}" + (f" | fusion = {fusion_label}" if "fusion" in names and fusion_label else ""))
    for c in classes:
        print(f"  {c:<10} {gt_boxes[c]:>6} GT boxes in {gt_images[c]:>5} images  "
              f"trained in: {', '.join(m for m in names if c in trained[m]) or 'none'}")

    # ---- matching ----------------------------------------------------------
    matches, thresholds = {}, {}
    for m in names:
        start = time.time()
        cls_m = [c for c in classes if c in trained[m]]
        matches[m] = match_dataset(dets[m]["records"], gts, cls_m, ids)
        thr_matches = matches[m] if thr_ids is ids else match_dataset(dets[m]["records"], gts, cls_m, thr_ids)
        thresholds[m] = {c: best_f1_conf(thr_matches[c]) for c in cls_m}
        print(f"Matched {m} in {time.time() - start:.0f}s")

    # ---- point estimates ---------------------------------------------------
    point: dict[str, dict[str, dict]] = {m: {} for m in names}
    for m in names:
        for c, mt in matches[m].items():
            ap = average_precision(mt)
            point[m][c] = {"ap50": ap[0], "ap75": ap[5], "ap50_95": float(ap.mean()),
                           **operating_point(mt, thresholds[m][c])}

    # ---- bootstrap (images resampled; same draws for every model -> paired) -
    boot = {m: {c: {"ap50": [], "ap50_95": []} for c in matches[m]} for m in names}
    boot_macro = {m: {"ap50": [], "ap50_95": []} for m in names}
    if args.bootstrap:
        rng = np.random.default_rng(args.boot_seed)
        start = time.time()
        for b in range(args.bootstrap):
            w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
            for m in names:
                for c, mt in matches[m].items():
                    ap = average_precision(mt, w)
                    boot[m][c]["ap50"].append(ap[0])
                    boot[m][c]["ap50_95"].append(np.nanmean(ap) if not np.all(np.isnan(ap)) else np.nan)
                if common:
                    boot_macro[m]["ap50"].append(np.nanmean([boot[m][c]["ap50"][-1] for c in common]))
                    boot_macro[m]["ap50_95"].append(np.nanmean([boot[m][c]["ap50_95"][-1] for c in common]))
            if (b + 1) % 100 == 0:
                print(f"  bootstrap {b + 1}/{args.bootstrap} ({time.time() - start:.0f}s)", end="\r")
        print()

    def ci(values) -> tuple[float, float]:
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    for m in names:
        for c, p in point[m].items():
            for k in ("ap50", "ap50_95"):
                p[f"{k}_lo"], p[f"{k}_hi"] = ci(boot[m][c][k]) if args.bootstrap else (np.nan, np.nan)

    macro = {}
    for m in names:
        macro[m] = {k: float(np.mean([point[m][c][k] for c in common])) if common else np.nan
                    for k in ("ap50", "ap50_95")}
        for k in ("ap50", "ap50_95"):
            macro[m][f"{k}_lo"], macro[m][f"{k}_hi"] = ci(boot_macro[m][k]) if args.bootstrap else (np.nan, np.nan)

    # ---- paired comparisons ------------------------------------------------
    pairs = [(a, b) for a, b in itertools.combinations(names, 2)]
    pairs = [(b, a) if b == "fusion" else (a, b) for a, b in pairs]  # fusion first
    comparisons = []
    for a, b in pairs:
        for c in [*common, "macro (common classes)"]:
            is_macro = c not in matches[a]
            row = {"class": c, "comparison": f"{a} - {b}"}
            for k in ("ap50", "ap50_95"):
                va = macro[a][k] if is_macro else point[a][c][k]
                vb = macro[b][k] if is_macro else point[b][c][k]
                row[f"d_{k}"] = va - vb
                if args.bootstrap:
                    da = np.asarray(boot_macro[a][k] if is_macro else boot[a][c][k], float)
                    db = np.asarray(boot_macro[b][k] if is_macro else boot[b][c][k], float)
                    d = da - db
                    row[f"d_{k}_lo"], row[f"d_{k}_hi"] = ci(d)
                    d = d[~np.isnan(d)]
                    row[f"p_{k}_le_0"] = float((d <= 0).mean()) if len(d) else np.nan
            comparisons.append(row)

    # ---- detection counts for classes without GT ---------------------------
    other: dict[str, dict[str, int]] = {}
    for m in names:
        for i in ids:
            r = dets[m]["records"][i]
            for n, s in zip(r["names"], r["scores"]):
                if s >= args.count_conf and n not in gt_boxes:
                    other.setdefault(n, {}).setdefault(m, 0)
                    other[n][m] += 1

    # ---- write -------------------------------------------------------------
    out = EVAL_DIR / args.split
    out.mkdir(parents=True, exist_ok=True)

    per_class_rows = []
    for c in classes:
        for m in names:
            row = {"split": args.split, "class": c, "gt_boxes": gt_boxes[c], "gt_images": gt_images[c],
                   "model": m, "trained": c in trained[m]}
            if c in point[m]:
                row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in point[m][c].items()})
                row["detections_at_conf"] = row["tp"] + row["fp"]
            per_class_rows.append(row)
    fields = list(dict.fromkeys(k for r in per_class_rows for k in r))
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_class_rows)
    if comparisons:
        with (out / "comparisons.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            w.writeheader()
            w.writerows(comparisons)
    with (out / "other_classes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", *[f"{m}_detections" for m in names]])
        for n in sorted(other, key=lambda n: -sum(other[n].values())):
            w.writerow([n, *[other[n].get(m, 0) for m in names]])

    # summary.md
    md = [f"# Thermal / RGB / Fusion per-class evaluation ({args.split} split)", "",
          f"- Pairs evaluated: **{len(ids)}** (split seed {seed}, test fraction {test_frac})",
          f"- Detection files: " + ", ".join(f"`{m}` ({Path(dets[m]['meta']['weights']).name if 'weights' in dets[m]['meta'] else dets[m]['meta'].get('label', '')})"
                                             for m in names),
          f"- Fusion config (chosen on tune split): `{fusion_label}`" if fusion_label else "",
          f"- P/R/F1 threshold: best F1 at IoU 0.5 on the {thr_source}",
          f"- 95% CIs: {args.bootstrap} bootstrap resamples of images" if args.bootstrap else "- CIs: off",
          ""]

    md += ["## AP50 per class", ""]
    rows = [[c, f"{gt_boxes[c]} ({gt_images[c]})",
             *[fmt_ci(point[m][c]["ap50"], point[m][c]["ap50_lo"], point[m][c]["ap50_hi"])
               if c in point[m] else "not trained" for m in names]] for c in classes]
    if common:
        rows.append([f"**mean ({', '.join(common)})**", "",
                     *[fmt_ci(macro[m]["ap50"], macro[m]["ap50_lo"], macro[m]["ap50_hi"]) for m in names]])
    md += [md_table(["Class", "GT boxes (images)", *names], rows), ""]

    md += ["## AP50-95 per class", ""]
    rows = [[c, f"{gt_boxes[c]} ({gt_images[c]})",
             *[fmt_ci(point[m][c]["ap50_95"], point[m][c]["ap50_95_lo"], point[m][c]["ap50_95_hi"])
               if c in point[m] else "not trained" for m in names]] for c in classes]
    if common:
        rows.append([f"**mean ({', '.join(common)})**", "",
                     *[fmt_ci(macro[m]["ap50_95"], macro[m]["ap50_95_lo"], macro[m]["ap50_95_hi"]) for m in names]])
    md += [md_table(["Class", "GT boxes (images)", *names], rows), ""]

    md += ["## Precision / recall / F1 per class (IoU 0.5)", ""]
    rows = []
    for c in classes:
        for m in names:
            if c not in point[m]:
                rows.append([c, str(gt_boxes[c]), m, *["n/a"] * 7])
                continue
            p = point[m][c]
            rows.append([c, str(gt_boxes[c]), m, fmt(p["conf"]), str(p["tp"]), str(p["fp"]), str(p["fn"]),
                         fmt(p["precision"]), fmt(p["recall"]), fmt(p["f1"])])
    md += [md_table(["Class", "GT", "Model", "Conf", "TP", "FP", "FN", "P", "R", "F1"], rows), ""]

    if comparisons:
        md += ["## Paired differences (bootstrap 95% CI)", "",
               "Positive = first model better.  A CI that excludes 0 is a reliable difference.", ""]
        rows = [[r["class"], r["comparison"],
                 fmt_ci(r["d_ap50"], r.get("d_ap50_lo"), r.get("d_ap50_hi"), signed=True),
                 fmt(r.get("p_ap50_le_0")),
                 fmt_ci(r["d_ap50_95"], r.get("d_ap50_95_lo"), r.get("d_ap50_95_hi"), signed=True),
                 fmt(r.get("p_ap50_95_le_0"))] for r in comparisons]
        md += [md_table(["Class", "Comparison", "ΔAP50", "P(Δ≤0)", "ΔAP50-95", "P(Δ≤0)"], rows), ""]

    if other:
        md += [f"## Detections of classes without GT (conf ≥ {args.count_conf})", ""]
        rows = [[n, *[str(other[n].get(m, 0)) for m in names]]
                for n in sorted(other, key=lambda n: -sum(other[n].values()))]
        md += [md_table(["Class", *names], rows), ""]

    summary = "\n".join(line for line in md if line is not None)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    (out / "results.json").write_text(json.dumps(clean({
        "split": args.split, "n_pairs": len(ids), "seed": seed, "test_frac": test_frac,
        "models": {m: dets[m]["meta"] for m in names}, "fusion": fusion_label,
        "threshold_source": thr_source, "bootstrap": args.bootstrap,
        "gt": {c: {"boxes": gt_boxes[c], "images": gt_images[c]} for c in classes},
        "per_class": point, "macro_common_classes": {"classes": common, **macro},
        "comparisons": comparisons, "other_classes": other,
    }), indent=2), encoding="utf-8")

    print()
    print(summary)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
