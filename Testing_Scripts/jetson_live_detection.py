from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TestConfig:
    # Model settings
    model_path: Path = ROOT / "weights" / "best_s1.pt"
    device: str | None = None
    confidence: float = 0.31
    iou_threshold: float = 0.45
    image_size: int = 480
    max_detections: int = 100

    # Camera settings
    use_csi_camera: bool = False
    usb_camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    camera_fps: int = 30
    flip_method: int = 0

    # Recording settings
    show_preview: bool = True
    output_dir: Path = ROOT / "recordings"
    output_name: str = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


CONFIG = TestConfig()


CLASS_COLORS = {
    "person": (0, 255, 255),
    "two_wheeler": (0, 165, 255),
    "car": (0, 255, 0),
    "large_vehicle": (255, 0, 255),
    "light": (128, 0, 128),
    "sign": (0, 255, 128),
    "bike": (0, 165, 255),
    "motor": (0, 165, 255),
    "scooter": (0, 165, 255),
    "bus": (255, 0, 255),
    "truck": (255, 0, 255),
}


def pick_device(preferred_device):
    if preferred_device:
        return preferred_device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_csi_pipeline(config):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){config.frame_width}, "
        f"height=(int){config.frame_height}, format=(string)NV12, "
        f"framerate=(fraction){config.camera_fps}/1 ! "
        f"nvvidconv flip-method={config.flip_method} ! "
        f"video/x-raw, width=(int){config.frame_width}, height=(int){config.frame_height}, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def open_camera(config):
    if config.use_csi_camera:
        source = build_csi_pipeline(config)
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(config.usb_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open the camera.")

    return cap


def create_video_writer(config, cap):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / config.output_name

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.frame_width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.frame_height
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(config.camera_fps)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    return writer, output_path


def draw_detections(frame, result):
    names = result.names
    boxes = result.boxes
    object_count = 0

    if boxes is None or len(boxes) == 0:
        return frame, object_count

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, classes):
        object_count += 1
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        class_name = names[class_id]
        label = f"{class_name} {confidence:.2f}"
        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_bottom = max(text_height + 8, y1)
        cv2.rectangle(
            frame,
            (x1, label_bottom - text_height - 8),
            (x1 + text_width, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, label_bottom - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame, object_count


def draw_status_text(frame, fps, object_count):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (12, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )


def run_detection(config):
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model not found: {config.model_path}")

    device = pick_device(config.device)
    print(f"Loading model from: {config.model_path}")
    print(f"Running on device: {device}")

    model = YOLO(str(config.model_path))
    cap = open_camera(config)
    writer, output_path = create_video_writer(config, cap)

    print(f"Recording detection video to: {output_path}")
    if config.show_preview:
        print("Press 'q' to stop.")

    fps_history = deque(maxlen=30)

    try:
        while True:
            start_time = time.time()

            success, frame = cap.read()
            if not success:
                print("Could not read a frame from the camera. Stopping.")
                break

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            results = model.predict(
                source=frame,
                imgsz=config.image_size,
                conf=config.confidence,
                iou=config.iou_threshold,
                max_det=config.max_detections,
                device=device,
                verbose=False,
            )

            annotated_frame = frame.copy()
            annotated_frame, object_count = draw_detections(annotated_frame, results[0])

            frame_time = max(time.time() - start_time, 1e-6)
            fps_history.append(1.0 / frame_time)
            avg_fps = sum(fps_history) / len(fps_history)

            draw_status_text(annotated_frame, avg_fps, object_count)
            writer.write(annotated_frame)

            if config.show_preview:
                cv2.imshow("Jetson Object Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved video: {output_path}")


def main():
    run_detection(CONFIG)


if __name__ == "__main__":
    main()
