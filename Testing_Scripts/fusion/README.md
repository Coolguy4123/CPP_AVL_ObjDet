# Thermal + RGB Sensor Fusion: Detection, Tuning and Per-Class Evaluation

This folder holds three scripts that run one after another. Each step saves its
results to a file, so a later step never repeats an earlier one.

```bash
python Testing_Scripts/fusion/detect.py --device 0   # 1. inference, once per set of weights
python Testing_Scripts/fusion/tune_fusion.py         # 2. choose fusion settings, save fused detections
python Testing_Scripts/fusion/evaluate.py            # 3. per-class metrics (seconds to minutes, re-run freely)
```

| Step | Reads | Writes (under `runs/fusion/`) | Re-run when |
|---|---|---|---|
| `detect.py` | `align/`, model weights | `detections/ground_truth.npz`, `detections/thermal.npz`, `detections/rgb.npz` | weights change (`--overwrite`, or `--modality rgb --overwrite` for one model) |
| `tune_fusion.py` | the three files above | `fusion_config.json`, `tuning_log.csv`, `detections/fusion.npz` | detections change, or you want a different search |
| `evaluate.py` | any `detections/*.npz` | `eval/<split>/summary.md`, `per_class.csv`, `comparisons.csv`, `other_classes.csv`, `results.json` | any time |

`fusion_common.py` holds the shared code: data loading, file format, fusion
algorithms and metrics.

To fuse again with a saved config without repeating the search:
`tune_fusion.py --config runs/fusion/fusion_config.json`.

---

## 1. Data: `align/`

The `align` folder holds co-registered thermal/RGB pairs. Co-registered means a
pixel in one image and the same pixel in the other show the same point in the
scene, so a single set of boxes is correct for both images.

| File | Contents |
|---|---|
| `align/JPEGImages/FLIR_xxxxx_PreviewData.jpeg` | thermal image (640×512) |
| `align/JPEGImages/FLIR_xxxxx_RGB.jpg` | RGB image (640×512) |
| `align/Annotations/FLIR_xxxxx_PreviewData.xml` | shared ground truth (Pascal VOC) |

The set has 5,142 pairs, with about 24.7k `car`, 13.1k `person`, 2.9k `bicycle`
and 108 `dog` boxes.

## 2. Models and classes

| Modality | Weights | Input size | Classes |
|---|---|---|---|
| Thermal | `thermal_training/weights/best_s1.pt` | 832 | 15 |
| RGB | `rgb_training/yolov8s_rgb/weights/best.pt` | 1024 | 80 (the same 15 plus 65 RGB-only) |

**Every class a model was trained on is detected and saved, under the model's own
class name.** Nothing is merged or renamed. The single translation is applied to
the ground truth only: the XML files label bicycles `bicycle`, while both models
call that class `bike`. The script reads `bicycle` as `bike` so those boxes can be
scored. This is `GT_LABEL_TO_MODEL_NAME` in `fusion_common.py`.

- Classes both models know are fused across the two sensors.
- RGB-only classes (`license plate`, `rider`, `face`, ...) pass through from RGB.
  Their scores are not reduced because thermal has nothing for them.

AP needs ground truth, so it is computed for the classes labelled in `align`
(`car`, `person`, `bike`, `dog`). Every other class gets detection counts per
model.

## 3. Why fuse thermal and RGB?

| Situation | Thermal | RGB |
|---|---|---|
| Night, low light, headlight glare | ✅ sees heat | ❌ dark or washed out |
| Smoke, light fog, hard shadows | ✅ mostly unaffected | ❌ contrast drops |
| Traffic-light colour, sign text | ❌ no colour or texture | ✅ |
| Cold or parked objects at ambient temperature | ❌ low contrast | ✅ |
| Fine detail (plates, faces) | ❌ low resolution | ✅ |

When the two sensors' errors are not correlated, fusion can raise **recall**,
because one sensor recovers what the other misses. It can also raise
**precision and localisation**, because boxes both sensors agree on are more
trustworthy and averaging the two gives tighter coordinates.

## 4. How the fusion works

The pipeline uses **late (decision-level) fusion**. Each detector runs on its own
image, and only their output boxes are combined. Neither model needs retraining,
and if one camera fails the system still has the other detector's output.

```
thermal img ──► thermal YOLO ──┐
                               ├──► per-class box fusion ──► fused detections
RGB img     ──► RGB YOLO ──────┘
```

1. **Inference** (`detect.py`): each model runs at `conf=0.001` and keeps up to
   300 boxes per image. The low threshold preserves the full precision/recall
   curve that mAP needs. Boxes are stored in the annotation frame.
2. **Grouping**: for each image, the boxes from both models are pooled, boxes
   below the score floor (`skip_thr`) are dropped, and the rest are grouped by
   class name.
3. **Combining**: each class is fused with one of the methods below. Each method
   takes modality weights `w_thermal` and `w_rgb`, which set how much to trust
   each sensor.
   - **NMS**: each score is multiplied by its weight. The top box is kept and any
     box overlapping it with IoU > `iou_thr` is removed. The best box wins and
     the other is discarded.
   - **Soft-NMS**: works like NMS, but an overlapping box's score is reduced by
     `exp(-IoU²/0.5)` instead of the box being removed. This helps in crowded
     scenes.
   - **WBF (Weighted Boxes Fusion, Solovyev et al. 2021)**: overlapping boxes are
     clustered (IoU > `iou_thr`). Each cluster's box is the **score-weighted
     average of its corners**: `Σ(w·s·box)/Σ(w·s)`. The cluster's score depends
     on `conf_type`:
     - `avg`: `Σ(w·s) · min(#models in cluster, #capable) / #boxes / Σ w_capable`.
       A box found by only one sensor is scaled down. This **rewards agreement
       between the sensors** and suppresses false positives from a single
       sensor.
     - `max`: the highest weighted score in the cluster. A detection from one
       sensor stays strong, which favours recall when one camera is blind.
     - `box_avg`: the mean weighted score of the cluster.

     `capable` is the set of models trained on that class, so RGB-only classes
     are normalised by RGB alone.

## 5. Tuning protocol (`tune_fusion.py`)

The pairs are split at the image level, with a fixed seed, into a **tune half**
and a **test half** (`--seed 0`, `--test-frac 0.5`). **Only the tune half is used
to choose the settings.** The split is saved in `fusion_config.json`, and
`evaluate.py` reads it from there so both scripts use the same split. The search
runs in four stages, scored by the mean mAP50-95 over the GT classes:

| Stage | Searched |
|---|---|
| 1 | method ∈ {NMS, Soft-NMS, WBF-avg, WBF-max, WBF-box_avg} × IoU ∈ {0.5, 0.55, 0.6, 0.7} |
| 2 | thermal:RGB weight from 1:3 to 3:1 |
| 3 | score floor ∈ {0.01, 0.03, 0.05, 0.1} |
| 4 | IoU ±0.025 / ±0.05 around the best |

Configs run in parallel (`--jobs`). Every config tried is written to
`tuning_log.csv`.

## 6. Evaluation (`evaluate.py`)

Each GT class is evaluated **separately** and reported next to its GT count
(boxes and images) for every model.

| Metric | How it is computed |
|---|---|
| AP50, AP75, AP50-95 | COCO-style: greedy matching, IoU 0.50 to 0.95, 101-point interpolated precision |
| 95% CI | bootstrap over images (`--bootstrap 1000`). Resampling whole images keeps the objects within an image together |
| P, R, F1, TP, FP, FN | at IoU 0.5, at each model's best-F1 confidence **chosen on the tune split** and applied unchanged to the test split, so the threshold does not leak test data |
| Paired difference | e.g. fusion − thermal, computed on the **same bootstrap draws**, with a 95% CI and P(Δ ≤ 0) |
| Mean over classes | mean AP over the GT classes every model is trained on, with a CI |
| Other classes | detection counts at conf ≥ 0.25 for classes without GT |

`summary.md` contains the same tables in Markdown, ready to copy.

## 7. Options

| Script | Flag | Default | Meaning |
|---|---|---|---|
| detect | `--modality` | thermal rgb | which detectors to run |
| detect | `--device` | auto | `0` for the first GPU, `cpu` for CPU |
| detect | `--thermal-imgsz`, `--rgb-imgsz` | training size | inference resolution |
| detect | `--overwrite` | off | redo detections and GT |
| detect | `--limit N` | – | first N pairs only (quick test) |
| tune_fusion | `--seed`, `--test-frac` | 0, 0.5 | split |
| tune_fusion | `--metric` | `map50_95` | `map50` or `map50_95` |
| tune_fusion | `--jobs` | CPUs−1 (max 8) | parallel workers |
| tune_fusion | `--config` | – | skip the search and fuse with a saved config |
| evaluate | `--split` | `test` | `test`, `tune` or `all` |
| evaluate | `--models` | thermal rgb fusion | detection files to compare |
| evaluate | `--bootstrap` | 1000 | resamples for CIs (0 turns them off) |

To run a quick test without touching the real results, set `FUSION_OUT_DIR` to a
scratch folder and pass `--limit 300` to `detect.py`.

## 9. Caveats

- Late fusion assumes the two images are registered. If the boxes from the two
  sensors are offset, WBF merges them less often; the tuner's IoU search
  partially compensates.