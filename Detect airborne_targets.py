"""Detect and track airborne objects in a video with a YOLO model.

IMPORTANT: the program can only find classes that exist in the model weights.
The standard yolov8*.pt (COCO) weights include airplane, but NOT helicopter,
drone or missile. Those need weights trained on labeled examples.

Example:
    python detect_airborne_targets.py --source Bush_Flyover.mp4 --weights yolov8m.pt --imgsz 1280 --conf 0.15 --no-show

Outputs (folder runs/airborne/): annotated video and a CSV of detections.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

AIRBORNE_LABELS = {
    "airplane": ("airplane", "aeroplane", "aircraft", "plane", "jet"),
    "helicopter": ("helicopter", "chopper"),
    "drone": ("drone", "uav", "quadcopter", "quadrotor", "multirotor"),
    "missile": ("missile", "rocket"),
}


def normalize_label(label):
    return "".join(c for c in label.lower() if c.isalnum())


def find_airborne_classes(names):
    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        class_names = {i: str(v) for i, v in enumerate(names)}
    ids, groups = [], set()
    for cid, cname in class_names.items():
        n = normalize_label(cname)
        for group, aliases in AIRBORNE_LABELS.items():
            if any(normalize_label(al) in n for al in aliases):
                ids.append(cid)
                groups.add(group)
                break
    return class_names, sorted(set(ids)), groups


def main():
    p = argparse.ArgumentParser(description="Detect and track airborne classes with YOLO.")
    p.add_argument("--source", default="Bush_Flyover.mp4", help="Video file or camera number.")
    p.add_argument("--weights", default="yolov8m.pt")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (bigger helps small objects).")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="runs/airborne")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    if not 0 <= a.conf <= 1:
        p.error("--conf must be between 0 and 1.")
    if not Path(a.weights).is_file() and not a.weights.startswith("yolo"):
        p.error(f"Weights file not found: {a.weights}")

    model = YOLO(a.weights)
    class_names, class_ids, groups = find_airborne_classes(model.names)
    if not class_ids:
        print("No airborne class names in this model. Use weights that include airplane, helicopter, drone or missile.")
        return
    print("Airborne types available:", ", ".join(sorted(groups)))
    missing = sorted(set(AIRBORNE_LABELS) - groups)
    if missing:
        print("NOT available in these weights (will not be detected):", ", ".join(missing))

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = out_dir / f"airborne_tracking_{stamp}.mp4"
    csv_path = out_dir / f"airborne_detections_{stamp}.csv"

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video or camera: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create result video: {video_path}")

    frames = 0
    frames_with_det = 0
    det_total = 0
    conf_sum = 0.0
    conf_max = 0.0
    ids_seen = set()
    start = time.time()

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_seconds", "class", "confidence", "tracking_id", "x1", "y1", "x2", "y2"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                res = model.track(frame, persist=True, tracker=a.tracker, classes=class_ids,
                                  conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
                boxes = res.boxes
                if boxes is not None and len(boxes):
                    frames_with_det += 1
                    cls = boxes.cls.int().tolist()
                    confs = boxes.conf.tolist()
                    xyxy = boxes.xyxy.int().tolist()
                    tids = boxes.id.int().tolist() if boxes.id is not None else [""] * len(boxes)
                    for c, cf, xy, tid in zip(cls, confs, xyxy, tids):
                        w.writerow([frames, f"{(frames - 1) / fps:.3f}", class_names[c],
                                    f"{cf:.4f}", tid, *xy])
                        det_total += 1
                        conf_sum += cf
                        conf_max = max(conf_max, cf)
                        if tid != "":
                            ids_seen.add(tid)
                annotated = res.plot()
                writer.write(annotated)
                if not a.no_show:
                    cv2.imshow("Airborne detection (press Q to stop)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-9)
    pct = 100.0 * frames_with_det / frames if frames else 0.0
    print("\n=============== AIRBORNE DETECTION RESULTS ===============")
    print(f"Source                  : {a.source}")
    print(f"Weights / image size    : {a.weights} / {a.imgsz}")
    print(f"Confidence threshold    : {a.conf}")
    print(f"Tracker                 : {a.tracker}")
    print(f"Total frames            : {frames}")
    print(f"Frames with detection   : {frames_with_det} ({pct:.1f}%)")
    print(f"Detected duration (sec) : {frames_with_det / fps:.1f}")
    print(f"Detections recorded     : {det_total}")
    print(f"Unique tracking IDs     : {len(ids_seen)}")
    print(f"Max confidence          : {conf_max * 100:.1f}%")
    print(f"Average confidence      : {(conf_sum / det_total * 100) if det_total else 0:.1f}%")
    print(f"Processing speed        : {frames / elapsed:.1f} FPS (includes video writing)")
    print(f"Result video            : {video_path}")
    print(f"Detections table (CSV)  : {csv_path}")
    print("Note: frames with detection is NOT accuracy. Check the video visually.")


if __name__ == "__main__":
    main()