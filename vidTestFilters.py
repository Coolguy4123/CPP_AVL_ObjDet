import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# ** CREATES A GRAYSCALE THERMAL - LIKE FILTER **
def rgb_to_thermal_simulation(frame):
    """
    Convert RGB camera feed to thermal-like appearance
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # INVERT
    inverted = 255 - gray
    
    # 2: Edge-aware smoothing
    # Preserve human silhouettes better
    smooth1 = cv2.bilateralFilter(inverted, 9, 75, 75)
    
    # 3: Moderate contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth1)
    
    # 4: Detect potential human regions
    # Use multiple brightness levels to capture different heat signatures
    
    # High heat (face, exposed skin)
    _, high_heat = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    
    # Medium heat (body through clothes)
    _, medium_heat = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
    
    # Low heat (hair, periphery)
    _, low_heat = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY)
    
    # 5: Combine heat levels with different weights
    # This helps capture full human silhouette including hair
    heat_combined = cv2.addWeighted(high_heat, 0.5, medium_heat, 0.3, 0)
    heat_combined = cv2.addWeighted(heat_combined, 1.0, low_heat, 0.2, 0)
    
    # 6: Morphological operations to connect body parts
    # Helps connect head (with hair) to body
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    heat_combined = cv2.morphologyEx(heat_combined, cv2.MORPH_CLOSE, kernel)
    
    # 7: Blend back with enhanced image
    thermal = cv2.addWeighted(enhanced, 0.7, heat_combined, 0.3, 0)
    
    # 8: Boost overall brightness for better person detection 
    thermal = cv2.normalize(thermal, None, 30, 245, cv2.NORM_MINMAX)
    
    # 9: Final light smoothing
    thermal = cv2.GaussianBlur(thermal, (3, 3), 0)
    
    # Convert to 3-channel
    thermal_3ch = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
    
    return thermal_3ch, thermal

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
CONF_THRES = 0.4
IOU_THRES = 0.55
MAX_DET = 200

# CHANGE THIS TO YOUR VIDEO PATH
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

frame_count = 0
fps_display = 0.0
window_start = time.time()
window_sec = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't receive frame. Exiting ...")
        break
    
    # APPLY THERMAL SIMULATION FILTER
    thermal_3ch, thermal_display = rgb_to_thermal_simulation(frame)
    
    # INFERENCE
    results = model.predict(
        source=thermal_3ch,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=False,
        device=device
    )
    
    r = results[0]
    names = r.names
    
    # ===== DRAW DETECTIONS =====
    # Draw on the thermal-simulated frame for consistent visualization
    display_frame = thermal_3ch.copy()

    # Create seperate RGB display frame
    rgb_display_frame = frame.copy()
    
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[cls]
            label = f"{class_name} {conf:.2f}"
            
            # Get color for this class (Default is Green)
            color = CLASS_COLORS.get(class_name, (0, 255, 0))
            
            # bbox on thermal frame
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # label background on thermal frame
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            
            # label text on thermal frame
            cv2.putText(display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # bbox on RGB frame
            cv2.rectangle(rgb_display_frame, (x1, y1), (x2, y2), color, 2)
            
            # label background on RGB frame
            cv2.rectangle(rgb_display_frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            
            # label text on RGB frame
            cv2.putText(rgb_display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # ===== FPS DISPLAY =====
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
    cv2.putText(rgb_display_frame, fps, (7, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (100, 250, 0), 3)
    
    cv2.imshow('Thermal Detection', display_frame)
    cv2.imshow('RGB Detection', rgb_display_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
