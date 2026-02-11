import cv2
import time
from ultralytics import YOLO
import numpy as np

# Config according to the device as needed


class SimpleKartController:
    """Dummy controller for testing detection"""
    
    def __init__(self):
        self.current_steering = 0.0  # -1 to 1
        self.current_speed = 0.0     # 0 to 1
        
    def steer_left(self):
        self.current_steering = -0.5
        print("🔵 STEERING LEFT")
        
    def steer_right(self):
        self.current_steering = 0.5
        print("🔵 STEERING RIGHT")
        
    def go_straight(self):
        self.current_steering = 0.0
        print("🟢 GOING STRAIGHT")
        
    def stop(self):
        self.current_speed = 0.0
        print("🛑 STOPPING")
        
    def go_slow(self):
        self.current_speed = 0.3
        print("🟡 SLOW SPEED")
        
    def go_normal(self):
        self.current_speed = 0.6
        print("🟢 NORMAL SPEED")


def main():
    # SSH into the nvidia jetson
    # Load your trained model
    # Use .engine file if you converted to TensorRT, otherwise use .pt
    MODEL_PATH = 'best.pt'  # or 'best.engine'
    
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Initialize controller
    controller = SimpleKartController()
    
    # Initialize camera
    # For USB camera use 0, for CSI camera use something like:
    # 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    
    cap = cv2.VideoCapture(0)  # Change to 0, 1, 2 depending on your camera
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Camera opened successfully!")
    print("\nStarting detection... Press 'q' to quit")
    print("=" * 50)
    
    frame_count = 0
    fps_list = []
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=0.5, verbose=False)
            
            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': class_name,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                        'area': (x2 - x1) * (y2 - y1)
                    })
            
            # Make driving decisions based on detections
            frame_height, frame_width = frame.shape[:2]
            make_driving_decision(detections, controller, frame_width, frame_height)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # Add info overlay
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Steering: {controller.current_steering:.2f}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Speed: {controller.current_speed:.2f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLOv8 Detection Test', annotated_frame)
            
            # Print summary every 30 frames
            if frame_count % 30 == 0:
                print(f"\nFrame {frame_count} | FPS: {avg_fps:.1f} | Objects: {len(detections)}")
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                controller.stop()
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        controller.stop()
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTest completed. Average FPS: {np.mean(fps_list):.1f}")


def make_driving_decision(detections, controller, frame_width, frame_height):
    """
    Simple logic to control the kart based on detections
    Customize this based on your trained classes!
    """
    
    if len(detections) == 0:
        # No obstacles detected - go normal speed
        controller.go_straight()
        controller.go_normal()
        return
    
    # Divide frame into zones
    left_zone = frame_width * 0.33
    right_zone = frame_width * 0.67
    close_threshold = frame_height * 0.6  # Bottom 40% of frame
    
    # Categorize detections
    obstacles_left = []
    obstacles_center = []
    obstacles_right = []
    obstacles_close = []
    
    for det in detections:
        cx = det['center_x']
        cy = det['center_y']
        
        # Check if close (large in frame or in bottom portion)
        if cy > close_threshold or det['area'] > (frame_width * frame_height * 0.1):
            obstacles_close.append(det)
        
        # Categorize by position
        if cx < left_zone:
            obstacles_left.append(det)
        elif cx > right_zone:
            obstacles_right.append(det)
        else:
            obstacles_center.append(det)
    
    # Decision logic
    if obstacles_close:
        # Emergency stop for very close objects
        print(f"⚠️  CLOSE OBSTACLE DETECTED: {obstacles_close[0]['class']}")
        controller.stop()
        
    elif obstacles_center:
        # Object in center - need to steer around it
        print(f"⚠️  CENTER OBSTACLE: {obstacles_center[0]['class']}")
        controller.go_slow()
        
        # Steer toward side with fewer obstacles
        if len(obstacles_left) < len(obstacles_right):
            controller.steer_left()
        else:
            controller.steer_right()
            
    elif obstacles_left or obstacles_right:
        # Obstacles on sides - adjust steering slightly
        controller.go_slow()
        
        if obstacles_left and not obstacles_right:
            controller.steer_right()
        elif obstacles_right and not obstacles_left:
            controller.steer_left()
        else:
            controller.go_straight()
    
    else:
        # Detections but not in critical zones
        controller.go_straight()
        controller.go_normal()


if __name__ == "__main__":
    main()