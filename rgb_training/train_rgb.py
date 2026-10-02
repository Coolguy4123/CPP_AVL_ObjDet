"""Train a YOLO object detector on the RGB COCO exports.

The RGB annotations are stored as COCO JSON files.  On the first run this
script converts them to the YOLO label layout expected by Ultralytics, then
starts training with RGB-safe augmentations.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# Helps reduce allocator fragmentation during large 1024px batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RGB_TRAIN = ROOT / "images_rgb_train"
RGB_VAL = ROOT / "images_rgb_val"
DATASET = Path(__file__).resolve().parent / "dataset"
RUNS_DIR = ROOT / "runs"

MODEL = "yolov8s.pt"
EPOCHS = 200
IMAGE_SIZE = 1024
REQUIRE_RTX_5090 = True


def configure_rtx_5090() -> int:
    """Select and validate the RTX 5090 CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build for the RTX 5090."
        )

    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    print(f"GPU: {device_name} | compute capability: {capability}")
    if REQUIRE_RTX_5090 and "5090" not in device_name:
        raise RuntimeError(
            f"Expected an RTX 5090 on CUDA device {device_index}, found: {device_name}"
        )

    # Use fast TF32 tensor cores for FP32 operations where supported.
    torch.set_float32_matmul_precision("high")
    return device_index


def _category_map(categories: list[dict]) -> dict[int, int]:
    """Map COCO category IDs to contiguous YOLO class IDs."""
    ordered = sorted(categories, key=lambda category: category["id"])
    return {category["id"]: index for index, category in enumerate(ordered)}


def _convert_split(split_dir: Path, split: str) -> list[str]:
    annotation_file = split_dir / "coco.json"
    with annotation_file.open(encoding="utf-8") as file:
        coco = json.load(file)

    category_to_class = _category_map(coco["categories"])
    images_by_id = {image["id"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images_by_id}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    image_root = split_dir / "data"
    output_images = DATASET / "images" / split
    output_labels = DATASET / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_names = []
    for image in coco["images"]:
        source = image_root / Path(image["file_name"]).name
        if not source.exists():
            raise FileNotFoundError(f"RGB image listed in COCO JSON is missing: {source}")

        destination = output_images / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

        label_lines = []
        width, height = image["width"], image["height"]
        for annotation in annotations_by_image.get(image["id"], []):
            x, y, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            x_center = (x + box_width / 2) / width
            y_center = (y + box_height / 2) / height
            label_lines.append(
                f"{category_to_class[annotation['category_id']]} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        (output_labels / f"{source.stem}.txt").write_text(
            "\n".join(label_lines), encoding="utf-8"
        )
        image_names.append(source.name)

    return [category["name"] for category in sorted(coco["categories"], key=lambda c: c["id"])]


def prepare_dataset() -> Path:
    train_names = _convert_split(RGB_TRAIN, "train")
    val_names = _convert_split(RGB_VAL, "val")
    if train_names != val_names:
        raise ValueError(f"Train/validation class lists differ: {train_names} != {val_names}")

    data_yaml = DATASET / "rgb.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(DATASET),
                "train": "images/train",
                "val": "images/val",
                "names": train_names,
                "nc": len(train_names),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    device = configure_rtx_5090()
    data_yaml = prepare_dataset()
    train_config = {
        "data": str(data_yaml),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": -1,
        "workers": 8,
        "patience": 5,
        "device": device,
        "project": str(RUNS_DIR),
        "name": "yolov8s_rgb",
        "exist_ok": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        # Basic RGB augmentation: modest geometric and colour variation.
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.50,
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": 0.50,
        "mixup": 0.0,
        "box": 7.5,
        "cls": 0.7,
        "dfl": 1.5,
        "amp": True,
    }

    print(f"Training RGB detector on RTX 5090 (CUDA device {device})")
    print(f"Dataset config: {data_yaml}")
    model = YOLO(MODEL)
    results = model.train(**train_config)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
