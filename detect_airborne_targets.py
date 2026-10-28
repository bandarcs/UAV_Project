r"""Detect and track airborne objects in a video with a trained Ultralytics YOLO model.

IMPORTANT:
    This program can only find object types that are already included in the
    model weights. The standard yolov8*.pt COCO weights include airplane, but
    they do not include helicopter, drone, or missile classes. Those classes
    require a model trained on labeled examples of them.

Example (run from the UAV project folder):
    .\venv\Scripts\python.exe detect_airborne_targets.py

Choose another video or trained model:
    .\venv\Scripts\python.exe detect_airborne_targets.py --source test02.mp4 --weights weights\best.pt

The program saves an annotated video and a CSV file under runs/airborne/.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# These are the names this program recognizes in the model's class list.
# Add dataset-specific names here if your trained model uses different labels.
AIRBORNE_LABELS = {
    "airplane": ("airplane", "aeroplane", "aircraft", "plane", "jet"),
    "helicopter": ("helicopter", "chopper"),
    "drone": ("drone", "uav", "quadcopter", "quadrotor", "multirotor"),
    "missile": ("missile", "rocket"),
}


def normalize_label(label: str) -> str:
    """Make labels comparable when they contain spaces, dashes, or underscores."""
    return "".join(character for character in label.lower() if character.isalnum())


def find_airborne_classes(names):
    """Return model class IDs and the airborne groups they match."""
    if isinstance(names, dict):
        class_names = {int(class_id): str(name) for class_id, name in names.items()}
    else:
        class_names = {class_id: str(name) for class_id, name in enumerate(names)}

    class_ids = []
    found_groups = set()
    for class_id, class_name in class_names.items():
        normalized_name = normalize_label(class_name)
        for group, aliases in AIRBORNE_LABELS.items():
            if any(normalize_label(alias) in normalized_name for alias in aliases):
                class_ids.append(class_id)
                found_groups.add(group)
                break
    return class_names, sorted(set(class_ids)), found_groups


def parse_source(source: str):
    """Allow either a video filename or a camera number such as 0."""
    return int(source) if source.isdigit() else source


def choose_image_size(model, cap, source_value, class_ids, confidence, augment, requested_size):
    """Pick a useful analysis size by comparing a few frames from the same video."""
    if requested_size.lower() != "auto":
        size = int(requested_size)
        if size <= 0:
            raise SystemExit("--imgsz must be 'auto' or a positive number.")
        return size

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if isinstance(source_value, int) or frame_count <= 0:
        print("Automatic image-size check is unavailable for this camera/stream; using 1280.")
        return 1280

    sample_positions = sorted(
        {round((frame_count - 1) * fraction) for fraction in (0.10, 0.30, 0.50, 0.70, 0.90)}
    )
    sample_frames = []
    for position in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = cap.read()
        if ok:
            sample_frames.append(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not sample_frames:
        print("Could not sample the video; using image size 1280.")
        return 1280

    candidate_sizes = (640, 960, 1280, 1600)
    scores = {}
    print(f"Checking image sizes on {len(sample_frames)} sample frames...")
    for size in candidate_sizes:
        box_count = 0
        confidence_sum = 0.0
        for frame in sample_frames:
            result = model.predict(
                frame,
                classes=class_ids,
                conf=confidence,
                imgsz=size,
                augment=augment,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                box_count += len(boxes)
                confidence_sum += float(boxes.conf.sum())
        scores[size] = (box_count, confidence_sum)
        print(f"  {size}: {box_count} detections, confidence total {confidence_sum:.2f}")

    selected_size = max(scores, key=lambda size: (scores[size][0], scores[size][1], -size))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Automatically selected image size: {selected_size}")
    return selected_size


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track airborne classes supported by the chosen YOLO weights."
    )
    parser.add_argument("--source", default="Bush_Flyover.mp4", help="Video filename or camera number (for example 0).")
    parser.add_argument("--weights", default="yolov8m.pt", help="YOLO weights file; custom trained weights can add more classes.")
    parser.add_argument("--conf", type=float, default=0.10, help="Minimum confidence from 0 to 1 (default: 0.10).")
    parser.add_argument("--imgsz", default="auto", help="Analysis size in pixels, or auto to compare several sizes on this video.")
    parser.add_argument("--augment", action="store_true", help="Try multiple image scales; slower, may find more small objects.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument("--output-dir", default="runs/airborne", help="Folder for the result video and CSV report.")
    parser.add_argument("--no-show", action="store_true", help="Save results without opening a preview window.")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")

    model_path = Path(args.weights)
    if not model_path.is_file():
        parser.error(f"Weights file not found: {model_path}")

    model = YOLO(str(model_path))
    class_names, class_ids, found_groups = find_airborne_classes(model.names)

    print("Classes in the chosen model:", ", ".join(class_names.values()))
    if not class_ids:
        print("This model has no recognized airborne class names, so there is nothing to detect.")
        print("Train or choose weights whose class names include airplane, helicopter, drone, or missile.")
        return

    missing_groups = sorted(set(AIRBORNE_LABELS) - found_groups)
    print("Airborne types available in these weights:", ", ".join(sorted(found_groups)))
    if missing_groups:
        print("Not available in these weights:", ", ".join(missing_groups))
        print("Those types will not be detected until you use weights trained to recognize them.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "airborne_tracking.mp4"
    csv_path = output_dir / "airborne_detections.csv"

    source_value = parse_source(args.source)
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video or camera: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = choose_image_size(
        model, cap, source_value, class_ids, args.conf, args.augment, args.imgsz
    )
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create result video: {video_path}")

    frame_number = 0
    detection_count = 0
    start_time = time.time()

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["frame", "time_seconds", "class", "confidence", "tracking_id", "x1", "y1", "x2", "y2"]
            )

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1

                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    classes=class_ids,
                    conf=args.conf,
                    imgsz=image_size,
                    augment=args.augment,
                    verbose=False,
                )[0]

                boxes = result.boxes
                if boxes is not None and len(boxes):
                    class_values = boxes.cls.int().tolist()
                    confidence_values = boxes.conf.tolist()
                    coordinates = boxes.xyxy.int().tolist()
                    tracking_values = (
                        boxes.id.int().tolist()
                        if boxes.id is not None
                        else [""] * len(boxes)
                    )

                    for class_id, confidence, coords, tracking_id in zip(
                        class_values, confidence_values, coordinates, tracking_values
                    ):
                        x1, y1, x2, y2 = coords
                        csv_writer.writerow(
                            [
                                frame_number,
                                f"{(frame_number - 1) / fps:.3f}",
                                class_names[class_id],
                                f"{confidence:.4f}",
                                tracking_id,
                                x1,
                                y1,
                                x2,
                                y2,
                            ]
                        )
                        detection_count += 1

                annotated_frame = result.plot()
                writer.write(annotated_frame)
                if not args.no_show:
                    cv2.imshow("Airborne object detection (press Q to stop)", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("\nFinished.")
    print(f"Frames processed: {frame_number}")
    print(f"Airborne detections recorded: {detection_count}")
    print(f"Processing speed: {frame_number / elapsed:.1f} frames/second" if elapsed else "Processing speed: unavailable")
    print(f"Video with boxes and tracking IDs: {video_path}")
    print(f"Detection details table: {csv_path}")


if __name__ == "__main__":
    main()





