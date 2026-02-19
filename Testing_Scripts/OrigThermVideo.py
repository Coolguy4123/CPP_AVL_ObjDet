import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# COLOR MAPPING FOR EACH CLASS
CLASS_COLORS = {
    'person': (0, 255, 255),      # Yellow
    'car': (0, 255, 0),           # Green
    'bus': (255, 0, 0),           # Blue
    'truck': (255, 0, 255),       # Magenta
    'motor': (0, 165, 255),       # Orange
    'bike': (255, 255, 0),        # Cyan
    'light': (128, 0, 128),       # Purple
    'sign': (0, 255, 128),        # Spring Green
    'hydrant': (128, 128, 0),     # Teal
    'skateboard': (255, 192, 203),# Pink
    'stroller': (255, 255, 255),  # White
    'other_vehicle': (128, 128, 128), # Gray
}

# === CONFIGURATIONS ===
WEIGHTS = "best.pt"
IMG_SIZE = 1024
CONF_THRES = 0.6
IOU_THRES = 0.35
MAX_DET = 50
# CONF_THRES = 0.7
# IOU_THRES = 0.55
# MAX_DET = 200

VIDEO_PATH = "t1.mp4"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
# =======================

# === MAIN INFERENCE PORTION ===
model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# Video writer setup
fps_in = cap.get(cv2.CAP_PROP_FPS)
if fps_in <= 0 or np.isnan(fps_in):
    fps_in = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUT_PATH = "output_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_PATH, fourcc, fps_in, (width, height))


# == FPS Settings ==
frame_count = 0
fps_display = 0.0
window_start = time.time()
window_sec = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't receive frame. Exiting ...")
        break

    # If thermal video is single-channel, YOLO expects 3 channels:
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=False,
        device=device
    )

    r = results[0]
    names = r.names

    # Draw detections on the same frame
    display_frame = frame.copy()

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[cls]
            label = f"{class_name} {conf:.2f}"

            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, y1 - th - 8)
            cv2.rectangle(display_frame, (x1, y_text), (x1 + tw, y_text + th + 8), color, -1)
            cv2.putText(display_frame, label, (x1, y_text + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # == FPS DISPLAY ==
    frame_count += 1
    now = time.time()
    elapsed = now - window_start
    if elapsed >= window_sec:
        fps_display = frame_count / elapsed
        frame_count = 0
        window_start = now

    fps = f"FPS: {fps_display:.1f}"
    cv2.putText(display_frame, fps, (7, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (100, 250, 0), 3)

    # Write frame to output video
    writer.write(display_frame)

    cv2.imshow('Thermal Video Detection', display_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
writer.release()

cv2.destroyAllWindows()
