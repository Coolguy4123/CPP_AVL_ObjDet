# RGB object detection training

Run from the repository root with:

```powershell
python rgb_training/train_rgb.py
```

The script reads `images_rgb_train/coco.json` and `images_rgb_val/coco.json`,
converts the annotations to YOLO format under `rgb_training/dataset/`, and
trains a YOLOv8s detector. The generated dataset is safe to delete and will be
recreated on the next run.

RGB-specific augmentation is intentionally modest: horizontal flips, small
rotations/translations, scale variation, HSV colour variation, and limited
mosaic augmentation. Validation images are not augmented.
