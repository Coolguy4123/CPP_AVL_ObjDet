"""Shared code for the thermal/RGB fusion pipeline.

    detect.py       run each detector once  -> runs/fusion/detections/{thermal,rgb}.npz
    tune_fusion.py  search fusion settings  -> runs/fusion/fusion_config.json, detections/fusion.npz
    evaluate.py     per-class metrics       -> runs/fusion/eval/<split>/...

Detection files all share one format (see save_records), so evaluate.py treats
thermal, RGB and fused detections identically.
"""

from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ALIGN = ROOT / "align"
THERMAL_WEIGHTS = ROOT / "thermal_training" / "weights" / "best_s1.pt"
RGB_WEIGHTS = ROOT / "rgb_training" / "yolov8s_rgb" / "weights" / "best.pt"

# FUSION_OUT_DIR lets a quick test run write somewhere other than the real results.
OUT_DIR = Path(os.environ.get("FUSION_OUT_DIR", ROOT / "runs" / "fusion"))
DET_DIR = OUT_DIR / "detections"
GT_FILE = DET_DIR / "ground_truth.npz"
CONFIG_FILE = OUT_DIR / "fusion_config.json"
EVAL_DIR = OUT_DIR / "eval"

MODALITIES = ("thermal", "rgb")
IMAGE_SUFFIX = {"thermal": "_PreviewData.jpeg", "rgb": "_RGB.jpg"}

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0, 1, 101)

DEFAULT_SEED = 0
DEFAULT_TEST_FRAC = 0.5

# Model class names are used exactly as trained.  The aligned XML files spell
# one class differently from the models, so only that GT label is translated to
# the model's name (the models' class names are never changed).
GT_LABEL_TO_MODEL_NAME = {
    "bicycle": "bike",
}


def det_file(name: str) -> Path:
    return DET_DIR / f"{name}.npz"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def gt_name(name: str) -> str:
    name = name.strip()
    return GT_LABEL_TO_MODEL_NAME.get(name, name)


def parse_voc(xml_path: Path) -> tuple[int, int, dict]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))
    boxes, names = [], []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
        names.append(gt_name(obj.find("name").text or ""))
    return width, height, {"boxes": np.asarray(boxes, np.float32).reshape(-1, 4), "names": names}


def build_ground_truth(align_dir: Path) -> tuple[list[str], list[dict], np.ndarray]:
    """Return stems, per-image GT records and (N, 2) annotation frame sizes for
    every annotation that has both a thermal and an RGB image."""
    images = align_dir / "JPEGImages"
    stems, records, sizes = [], [], []
    for xml_path in sorted((align_dir / "Annotations").glob("*_PreviewData.xml")):
        stem = xml_path.stem.replace("_PreviewData", "")
        if not all((images / f"{stem}{IMAGE_SUFFIX[m]}").exists() for m in MODALITIES):
            continue
        width, height, record = parse_voc(xml_path)
        stems.append(stem)
        records.append(record)
        sizes.append((width, height))
    return stems, records, np.asarray(sizes, np.int32).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Detection / GT file format
# ---------------------------------------------------------------------------

def save_records(path: Path, stems: list[str], records: list[dict], class_names: list[str],
                 meta: dict, **extra: np.ndarray) -> None:
    """Save per-image boxes (xyxy, annotation frame) as one flat compressed npz.

    records: [{"boxes": (n, 4), "names": [str] * n, optional "scores": (n,)}]
    """
    lookup = {n: i for i, n in enumerate(class_names)}
    counts = np.array([len(r["names"]) for r in records], np.int64)
    boxes = (np.concatenate([r["boxes"].reshape(-1, 4) for r in records]) if records
             else np.zeros((0, 4))).astype(np.float32)
    arrays = {
        "stems": np.array(stems, dtype=str),
        "counts": counts,
        "boxes": boxes,
        "class_ids": np.array([lookup[n] for r in records for n in r["names"]], np.int32),
        "class_names": np.array(class_names, dtype=str),
        "meta": np.array(json.dumps(meta)),
        **extra,
    }
    if records and "scores" in records[0]:
        arrays["scores"] = np.concatenate([r["scores"] for r in records]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_records(path: Path) -> dict:
    """Inverse of save_records.  Returns stems, records, class_names, meta (+ extras)."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    offsets = np.concatenate([[0], np.cumsum(data["counts"])])
    names = np.array(data["class_names"].tolist() + [""], dtype=object)[data["class_ids"]]
    scores = data.get("scores")
    records = []
    for i in range(len(data["stems"])):
        a, b = offsets[i], offsets[i + 1]
        r = {"boxes": data["boxes"][a:b], "names": names[a:b].tolist()}
        if scores is not None:
            r["scores"] = scores[a:b]
        records.append(r)
    out = {k: v for k, v in data.items()
           if k not in ("stems", "counts", "boxes", "class_ids", "class_names", "meta", "scores")}
    out.update(stems=data["stems"].tolist(), records=records,
               class_names=data["class_names"].tolist(), meta=json.loads(str(data["meta"])))
    return out


def load_detections(path: Path, gt_stems: list[str]) -> dict:
    det = load_records(path)
    if det["stems"] != gt_stems:
        raise SystemExit(f"{path} was made for a different image list than {GT_FILE}; "
                         f"re-run detect.py (and tune_fusion.py) so they match.")
    return det


def make_split(n: int, seed: int = DEFAULT_SEED, test_frac: float = DEFAULT_TEST_FRAC) -> dict[str, list[int]]:
    """Deterministic image-level split shared by tuning and evaluation."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_frac))
    return {"tune": sorted(idx[n_test:]), "test": sorted(idx[:n_test]), "all": list(range(n))}


# ---------------------------------------------------------------------------
# Boxes and fusion
# ---------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return boxes[keep], scores[keep]


def soft_nms(boxes, scores, iou_thr, sigma=0.5, min_score=1e-3):
    """Gaussian Soft-NMS; iou_thr gates which overlaps get decayed."""
    boxes, scores = boxes.copy(), scores.copy()
    kept_b, kept_s = [], []
    while len(scores):
        i = int(np.argmax(scores))
        kept_b.append(boxes[i])
        kept_s.append(scores[i])
        boxes = np.delete(boxes, i, axis=0)
        scores = np.delete(scores, i)
        if not len(scores):
            break
        ious = box_iou(kept_b[-1][None], boxes)[0]
        decay = np.where(ious > iou_thr, np.exp(-(ious ** 2) / sigma), 1.0)
        scores = scores * decay
        mask = scores >= min_score
        boxes, scores = boxes[mask], scores[mask]
    return np.asarray(kept_b, dtype=np.float32).reshape(-1, 4), np.asarray(kept_s, dtype=np.float32)


def wbf(boxes, scores, model_ids, weights, iou_thr, conf_type, capable):
    """Weighted Boxes Fusion (Solovyev et al.) for a single class in one image.

    capable: ids of the models trained on this class.  The "avg" score is
    normalised by those models only, so a class that only one modality knows is
    not penalised for the other modality's silence.

    conf_type:
        "avg"     fused score = sum(w*s) / sum(capable weights)   rewards cross-modal agreement
        "max"     fused score = max weighted score                keeps single-modality hits strong
        "box_avg" fused score = mean(w*s) over the cluster
    """
    weights = np.asarray(weights, dtype=np.float32)
    ws = scores * weights[model_ids]
    order = np.argsort(-ws)
    boxes, ws, model_ids = boxes[order], ws[order], model_ids[order]

    fused = np.zeros((0, 4), dtype=np.float32)
    members: list[list[int]] = []
    for i in range(len(boxes)):
        if len(fused):
            ious = box_iou(boxes[i:i + 1], fused)[0]
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                members[j].append(i)
                idx = members[j]
                w = ws[idx][:, None]
                fused[j] = (boxes[idx] * w).sum(0) / w.sum()
                continue
        members.append([i])
        fused = np.vstack([fused, boxes[i:i + 1]])

    out_scores = np.empty(len(members), dtype=np.float32)
    for j, idx in enumerate(members):
        s = ws[idx]
        n_models = len(set(model_ids[idx].tolist()))
        if conf_type == "max":
            out_scores[j] = s.max() / weights[capable].max()
        elif conf_type == "box_avg":
            out_scores[j] = s.mean() / weights[model_ids[idx]].mean()
        else:  # avg: one box per model at most counts toward agreement
            out_scores[j] = s.sum() * min(n_models, len(capable)) / len(idx) / weights[capable].sum()
    return fused, np.clip(out_scores, 0, 1)


def fuse_image(dets: list[dict], cfg: dict, class_models: dict[str, list[int]]) -> dict:
    """Fuse the per-modality detections of one image according to cfg.

    class_models maps each class name to the ids of the models trained on it.
    """
    weights = np.asarray(cfg["weights"], dtype=np.float32)
    all_b, all_s, all_m, all_n = [], [], [], []
    for m, d in enumerate(dets):
        keep = d["scores"] >= cfg["skip_thr"]
        all_b.append(d["boxes"][keep])
        all_s.append(d["scores"][keep])
        all_m.append(np.full(int(keep.sum()), m, dtype=int))
        all_n.extend(n for n, k in zip(d["names"], keep) if k)
    boxes = np.concatenate(all_b)
    scores = np.concatenate(all_s)
    model_ids = np.concatenate(all_m)
    names = np.asarray(all_n, dtype=object)

    out_b, out_s, out_n = [], [], []
    for name in sorted(set(all_n)):
        sel = names == name
        b, s, m = boxes[sel], scores[sel], model_ids[sel]
        if cfg["method"] == "wbf":
            fb, fs = wbf(b, s, m, weights, cfg["iou_thr"], cfg["conf_type"], class_models[name])
        else:
            s = s * weights[m] / weights[class_models[name]].max()
            fb, fs = (nms if cfg["method"] == "nms" else soft_nms)(b, s, cfg["iou_thr"])
        out_b.append(fb)
        out_s.append(fs)
        out_n.extend([name] * len(fs))
    if not out_b:
        return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32), "names": []}
    boxes, scores = np.concatenate(out_b), np.concatenate(out_s)
    top = np.argsort(-scores, kind="stable")[:cfg.get("max_det", 300)]
    return {"boxes": boxes[top], "scores": scores[top], "names": [out_n[i] for i in top]}


def class_models_for(class_lists: list[list[str]]) -> dict[str, list[int]]:
    all_classes = list(dict.fromkeys(c for cl in class_lists for c in cl))
    return {c: [m for m, cl in enumerate(class_lists) if c in cl] for c in all_classes}


def config_label(cfg: dict) -> str:
    w = "/".join(f"{x:g}" for x in cfg["weights"])
    extra = f" conf={cfg['conf_type']}" if cfg["method"] == "wbf" else ""
    return f"{cfg['method']} iou={cfg['iou_thr']:g} w(T/RGB)={w} skip={cfg['skip_thr']:g}{extra}"


# ---------------------------------------------------------------------------
# Matching and metrics (COCO-style: greedy matching, 101-point interpolated AP)
# ---------------------------------------------------------------------------

def match_dataset(preds: list[dict], gts: list[dict], classes: list[str], ids: list[int]) -> dict:
    """Match predictions to GT on the images in ids, per class.

    Returns {class: {"scores", "tp" (N, T) bool, "img" (local image index),
    "n_gt" (per-image GT counts)}} with detections sorted by descending score.
    """
    T = len(IOU_THRESHOLDS)
    acc = {c: {"scores": [], "tp": [], "img": [], "n_gt": np.zeros(len(ids), np.int64)} for c in classes}
    for k, i in enumerate(ids):
        p, g = preds[i], gts[i]
        pn = np.asarray(p["names"], dtype=object)
        gn = np.asarray(g["names"], dtype=object)
        for c in classes:
            gt = g["boxes"][gn == c] if len(gn) else np.zeros((0, 4), np.float32)
            acc[c]["n_gt"][k] = len(gt)
            if not len(pn):
                continue
            sel = pn == c
            if not sel.any():
                continue
            ps, pb = p["scores"][sel], p["boxes"][sel]
            order = np.argsort(-ps, kind="stable")
            ps, pb = ps[order], pb[order]
            tp = np.zeros((len(ps), T), dtype=bool)
            if len(gt):
                ious = box_iou(pb, gt)
                rows = np.where(ious.max(1) >= IOU_THRESHOLDS[0])[0]
                for t, thr in enumerate(IOU_THRESHOLDS):
                    taken = np.zeros(len(gt), dtype=bool)
                    for r in rows:
                        cand = np.where(~taken & (ious[r] >= thr), ious[r], -1)
                        j = int(np.argmax(cand))
                        if cand[j] >= 0:
                            taken[j] = True
                            tp[r, t] = True
            acc[c]["scores"].append(ps)
            acc[c]["tp"].append(tp)
            acc[c]["img"].append(np.full(len(ps), k, np.int64))

    out = {}
    for c, d in acc.items():
        if d["scores"]:
            scores = np.concatenate(d["scores"])
            order = np.argsort(-scores, kind="stable")
            out[c] = {"scores": scores[order], "tp": np.concatenate(d["tp"])[order],
                      "img": np.concatenate(d["img"])[order], "n_gt": d["n_gt"]}
        else:
            out[c] = {"scores": np.zeros(0, np.float32), "tp": np.zeros((0, T), bool),
                      "img": np.zeros(0, np.int64), "n_gt": d["n_gt"]}
    return out


def average_precision(m: dict, image_weights: np.ndarray | None = None) -> np.ndarray:
    """AP at each IoU threshold for one class's matches.

    image_weights (per local image) supports the bootstrap: an image drawn k
    times counts k times, which equals duplicating it.
    """
    T = len(IOU_THRESHOLDS)
    if image_weights is None:
        n_gt = m["n_gt"].sum()
        w = np.ones(len(m["scores"]))
    else:
        n_gt = (m["n_gt"] * image_weights).sum()
        w = image_weights[m["img"]].astype(np.float64)
    if n_gt <= 0:
        return np.full(T, np.nan)
    if not len(w):
        return np.zeros(T)
    wt = w[:, None]
    ctp = np.cumsum(m["tp"] * wt, 0)
    cfp = np.cumsum(~m["tp"] * wt, 0)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    ap = np.zeros(T)
    for t in range(T):
        p = np.maximum.accumulate(prec[::-1, t])[::-1]
        idx = np.searchsorted(rec[:, t], RECALL_POINTS, side="left")
        ap[t] = np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0).mean()
    return ap


def best_f1_conf(m: dict) -> float:
    """Confidence threshold that maximises F1 at IoU 0.5."""
    n_gt = m["n_gt"].sum()
    if not len(m["scores"]) or n_gt == 0:
        return 1.0
    ctp = np.cumsum(m["tp"][:, 0])
    prec = ctp / np.arange(1, len(ctp) + 1)
    rec = ctp / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(m["scores"][int(np.argmax(f1))])


def operating_point(m: dict, conf: float) -> dict:
    """TP/FP/FN, precision, recall, F1 at IoU 0.5 for detections with score >= conf."""
    keep = m["scores"] >= conf
    tp = int(m["tp"][keep, 0].sum())
    fp = int(keep.sum()) - tp
    n_gt = int(m["n_gt"].sum())
    fn = n_gt - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12)}


def macro_map(matches: dict) -> tuple[float, float]:
    """(mAP50, mAP50-95) averaged over classes that have GT."""
    aps = [average_precision(m) for m in matches.values() if m["n_gt"].sum() > 0]
    if not aps:
        return 0.0, 0.0
    aps = np.array(aps)
    return float(aps[:, 0].mean()), float(aps.mean())
