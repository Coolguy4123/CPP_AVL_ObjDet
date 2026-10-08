"""Step 1: run the thermal and RGB detectors once and save every detection.

Writes
    runs/fusion/detections/ground_truth.npz   aligned GT (bicycle read as the models' "bike")
    runs/fusion/detections/thermal.npz         thermal detections, all trained classes
    runs/fusion/detections/rgb.npz             RGB detections, all trained classes

Detections are kept down to conf=0.001 (needed for mAP) and stored in the
annotation frame.  An existing file is reused unless --overwrite is given, so
each modality only ever needs to be run once per set of weights.

Usage
    python Testing_Scripts/fusion/detect.py --device 0
    python Testing_Scripts/fusion/detect.py --modality rgb --overwrite --device 0
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from fusion_common import (ALIGN, GT_FILE, GT_LABEL_TO_MODEL_NAME, IMAGE_SUFFIX, MODALITIES,
                           RGB_WEIGHTS, THERMAL_WEIGHTS, build_ground_truth, det_file,
                           load_records, save_records)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ground_truth(align_dir: Path, overwrite: bool, limit: int | None) -> tuple[list[str], np.ndarray]:
    if GT_FILE.exists() and not overwrite:
        gt = load_records(GT_FILE)
        print(f"Using existing {GT_FILE} ({len(gt['stems'])} pairs)")
        return gt["stems"], gt["sizes"]

    print(f"Reading annotations from {align_dir}")
    stems, records, sizes = build_ground_truth(align_dir)
    if limit:
        stems, records, sizes = stems[:limit], records[:limit], sizes[:limit]
    if not stems:
        raise SystemExit(f"No thermal/RGB/XML triplets found in {align_dir}")
    classes = sorted({n for r in records for n in r["names"]})
    save_records(GT_FILE, stems, records, classes,
                 {"align_dir": str(align_dir), "gt_label_to_model_name": GT_LABEL_TO_MODEL_NAME,
                  "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                 sizes=sizes)
    print(f"Saved {GT_FILE}: {len(stems)} pairs")
    for c in classes:
        print(f"  {c:<10} {sum(n == c for r in records for n in r['names']):>6} GT boxes")
    return stems, sizes


def run_detector(modality: str, weights: Path, stems: list[str], sizes: np.ndarray,
                 align_dir: Path, args) -> None:
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgsz = getattr(args, f"{modality}_imgsz")
    if imgsz is None:
        imgsz = int((model.ckpt or {}).get("train_args", {}).get("imgsz", 640))
    class_names = [model.names[i] for i in sorted(model.names)]
    print(f"[{modality}] {weights} imgsz={imgsz} {len(class_names)} classes: {class_names}")

    paths = [align_dir / "JPEGImages" / f"{s}{IMAGE_SUFFIX[modality]}" for s in stems]
    records = []
    start = time.time()
    for b in range(0, len(paths), args.batch):
        results = model.predict([str(p) for p in paths[b:b + args.batch]], imgsz=imgsz,
                                conf=args.conf, iou=args.iou, max_det=args.max_det,
                                device=args.device, verbose=False)
        for res, (fw, fh) in zip(results, sizes[b:b + args.batch]):
            ih, iw = res.orig_shape
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            # Boxes are stored in the annotation frame in case image sizes differ.
            xyxy *= np.array([fw / iw, fh / ih, fw / iw, fh / ih], dtype=np.float32)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            records.append({"boxes": xyxy,
                            "scores": res.boxes.conf.cpu().numpy().astype(np.float32),
                            "names": [model.names[c] for c in cls]})
        done = min(b + args.batch, len(paths))
        rate = done / max(time.time() - start, 1e-9)
        print(f"  {done}/{len(paths)}  {rate:.1f} img/s", end="\r")
    print()

    meta = {
        "modality": modality,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": str(args.device),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - start, 1),
    }
    out = det_file(modality)
    save_records(out, stems, records, class_names, meta)
    print(f"Saved {out} ({sum(len(r['names']) for r in records)} detections)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--align-dir", type=Path, default=ALIGN)
    parser.add_argument("--thermal-weights", type=Path, default=THERMAL_WEIGHTS)
    parser.add_argument("--rgb-weights", type=Path, default=RGB_WEIGHTS)
    parser.add_argument("--thermal-imgsz", type=int, default=None, help="default: model's training imgsz")
    parser.add_argument("--rgb-imgsz", type=int, default=None, help="default: model's training imgsz")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6, help="per-model NMS IoU")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run even if the output file exists (also rebuilds GT)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N pairs only, for a quick test (combine with FUSION_OUT_DIR)")
    args = parser.parse_args()

    stems, sizes = ground_truth(args.align_dir, args.overwrite, args.limit)
    for modality in args.modality:
        out = det_file(modality)
        if out.exists() and not args.overwrite:
            print(f"[{modality}] {out} exists, skipping (use --overwrite to re-run)")
            continue
        run_detector(modality, getattr(args, f"{modality}_weights"), stems, sizes, args.align_dir, args)


if __name__ == "__main__":
    main()
