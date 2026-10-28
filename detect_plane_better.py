"""
Airplane detection in video using YOLO.

Run:
    ./venv/Scripts/python.exe detect_plane_better.py

Optional:
    --source VIDEO.mp4     video file to analyze
    --model yolov8x.pt     bigger model = more accurate, slower (n < s < m < l < x)
    --imgsz 1280           analysis size, bigger helps with small objects
    --conf 0.1             confidence threshold, lower = more detections
    --augment              multi-scale analysis (more accurate, slower)
    --all_classes          detect all object types, not only airplanes
    --no_show              do not open the live preview window
"""
import argparse
import time

import cv2
import torch
from ultralytics import YOLO

BOX_WIDTH = 54


def table_row(label, value):
    text = f"| {label:<22}: {value}"
    return text + " " * (BOX_WIDTH - len(text) - 1) + "|"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="Bush_Flyover.mp4")

    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    # Select device (GPU if available)
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"

    # Load model. Class 4 = airplane in the COCO dataset
    model = YOLO(args.model)
    classes = None if args.all_classes else [4]

    # Read the video frame rate (used to convert frames to seconds)
    cap = cv2.VideoCapture(args.source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    total_frames = 0
    detected_frames = 0
    total_boxes = 0
    conf_sum = 0.0
    max_conf = 0.0
    start_time = time.time()

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=classes,
        augment=args.augment,
        device=device,
        save=True,
        show=not args.no_show,
        stream=True,
        verbose=False,
    )

    for r in results:
        total_frames += 1
        if len(r.boxes):
            detected_frames += 1
            total_boxes += len(r.boxes)
            conf_sum += float(r.boxes.conf.sum())
            max_conf = max(max_conf, float(r.boxes.conf.max()))

    elapsed = time.time() - start_time
    detection_rate = detected_frames / total_frames * 100 if total_frames else 0
    avg_conf = conf_sum / total_boxes if total_boxes else 0
    processing_fps = total_frames / elapsed if elapsed else 0
    line = "+" + "-" * (BOX_WIDTH - 2) + "+"

    print()
    print(line)
    print("|" + "DETECTION RESULTS".center(BOX_WIDTH - 2) + "|")
    print(line)
    print(table_row("Video", args.source))
    print(table_row("Model", args.model))
    print(table_row("Image size", args.imgsz))
    print(table_row("Confidence threshold", args.conf))
    print(table_row("Augment", "Yes" if args.augment else "No"))
    print(table_row("Device", device_name))
    print(line)
    print(table_row("Total frames", total_frames))
    print(table_row("Detected frames", f"{detected_frames} ({detection_rate:.1f}%)"))
    print(table_row("Detected duration", f"{detected_frames / video_fps:.1f} s of {total_frames / video_fps:.1f} s"))
    print(table_row("Max confidence", f"{max_conf:.0%}"))
    print(table_row("Average confidence", f"{avg_conf:.0%}"))
    print(line)
    print(table_row("Processing time", f"{elapsed:.1f} s"))
    print(table_row("Processing speed", f"{processing_fps:.1f} FPS"))
    print(line)
    print("Output video saved in: runs\\detect\\")


if __name__ == "__main__":
    main()