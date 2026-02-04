import cv2
import numpy as np
from ultralytics import YOLO
import time

def rgb_to_thermal_simulation(frame):
    """
    Convert RGB camera feed to thermal-like appearance
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # === STEP 1: INVERT ===
    inverted = 255 - gray
    
    # === STEP 2: Edge-aware smoothing ===
    # Preserve human silhouettes better
    smooth1 = cv2.bilateralFilter(inverted, 9, 75, 75)
    
    # === STEP 3: Moderate contrast (not too aggressive) ===
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth1)
    
    # === STEP 4: Detect potential human regions ===
    # Use multiple brightness levels to capture different heat signatures
    
    # High heat (face, exposed skin)
    _, high_heat = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    
    # Medium heat (body through clothes)
    _, medium_heat = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
    
    # Low heat (hair, periphery)
    _, low_heat = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY)
    
    # === STEP 5: Combine heat levels with different weights ===
    # This helps capture full human silhouette including hair
    heat_combined = cv2.addWeighted(high_heat, 0.5, medium_heat, 0.3, 0)
    heat_combined = cv2.addWeighted(heat_combined, 1.0, low_heat, 0.2, 0)
    
    # === STEP 6: Morphological operations to connect body parts ===
    # This helps connect head (with hair) to body
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    heat_combined = cv2.morphologyEx(heat_combined, cv2.MORPH_CLOSE, kernel)
    
    # === STEP 7: Blend back with enhanced image ===
    thermal = cv2.addWeighted(enhanced, 0.7, heat_combined, 0.3, 0)
    
    # === STEP 8: Boost overall brightness for better person detection ===
    thermal = cv2.normalize(thermal, None, 30, 245, cv2.NORM_MINMAX)
    
    # === STEP 9: Final light smoothing ===
    thermal = cv2.GaussianBlur(thermal, (3, 3), 0)
    
    # Convert to 3-channel
    thermal_3ch = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
    
    return thermal_3ch, thermal


# ===== MAIN INFERENCE CODE WITH THERMAL FILTER =====

WEIGHTS = "best.pt"
IMG_SIZE = 832
CONF_THRES = 0.5
IOU_THRES = 0.55
MAX_DET = 200

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0
fps_display = 0.0
window_start = time.time()
window_sec = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # ===== APPLY THERMAL SIMULATION FILTER =====
    thermal_3ch, thermal_display = rgb_to_thermal_simulation(frame)
    
    # ===== YOLO INFERENCE ON THERMAL-SIMULATED FRAME =====
    results = model.predict(
        source=thermal_3ch,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=False
    )
    
    r = results[0]
    names = r.names
    
    # ===== DRAW DETECTIONS =====
    # Draw on the thermal-simulated frame for consistent visualization
    display_frame = thermal_3ch.copy()
    
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{names[cls]} {conf:.2f}"
            
            # bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            
            # label text
            cv2.putText(display_frame, label, (x1, y1 - 5), 
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
    
    # ===== SHOW BOTH ORIGINAL AND THERMAL SIDE-BY-SIDE =====
    # Resize for side-by-side comparison
    h, w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (w//2, h//2))
    display_resized = cv2.resize(display_frame, (w//2, h//2))
    
    # Add labels
    cv2.putText(frame_resized, "Original RGB", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_resized, "Thermal Simulation", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    comparison = np.hstack([frame_resized, display_resized])
    
    cv2.imshow('Comparison View', comparison)
    cv2.imshow('Thermal Detection', display_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()