import cv2
import time
import numpy as np
from ultralytics import YOLO

WEIGHTS = "best.pt"
IMG_SIZE = 832         
CONF_THRES = 0.58       # higher = fewer false positives
IOU_THRES = 0.55        # stronger NMS = fewer duplicate boxes
MAX_DET = 200           # avoid flooding output

model = YOLO(WEIGHTS)

# FOR CAMERA USAGES
#===================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# -- FOR FPS SETTINGS --
frame_count = 0
fps_display = 0.0
window_start = time.time()
window_sec = 0.1  # Update 10 times per second (Smaller value -> Faster FPS changes)

while True:
    ret, frame = cap.read()

    # CANNOT READ FRAME EXCEPTION
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # -- YOLO INFERENCE --
    results = model.predict(
        source = frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=False
    )

    r = results[0]
    names = r.names

    # Draw detections
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()   # (N,4) x1,y1,x2,y2
        confs = r.boxes.conf.cpu().numpy()   # (N,)
        clss  = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{names[cls]} {conf:.2f}"

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            # label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # -- FPS --
    font = cv2.FONT_HERSHEY_COMPLEX

    frame_count += 1
    now = time.time()
    elapsed = now - window_start

    if elapsed >= window_sec:
        fps_display = frame_count / elapsed
        frame_count = 0
        window_start = now

    fps = f"FPS: {fps_display:.1f}"

    cv2.putText(frame,       # Name of the frame
                fps,         # The FPS string
                (7,50),      # Location (x, y)
                1,           # Font scale
                font,        # Font style
                (100,250,0), # Text color (B,G,R) (0-255,0-255,0-255)
                3)           # Font Thickness


    # - SHOW FRAME
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# FOR VIDEOS
# -------------------------------------------------------
# cap = cv2.VideoCapture('vtest.avi')
 
# while cap.isOpened():
#     ret, frame = cap.read()
 
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) == ord('q'):
#         break
 
# cap.release()
# cv2.destroyAllWindows()