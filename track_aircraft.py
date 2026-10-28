r"""
Small / distant airplane detection using SAHI (sliced inference).

Run:
    .\venv\Scripts\python.exe track_aircraft.py

Optional:
    --source VIDEO.mp4   video file to analyze
    --model yolov8m.pt   detection model
    --conf 0.15          confidence threshold
    --slice 640          slice size in pixels (smaller = better for tiny objects, slower)
    --overlap 0.2        overlap between slices
    --no_show            do not open the preview window (press Q to stop when shown)
"""
import argparse
import time

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

BOX_WIDTH = 54
AIRPLANE_CLASS_ID = 4  # COCO class id for airplane


def table_row(label, value):
    text = f"| {label:<22}: {value}"
    return text + " " * (BOX_WIDTH - len(text) - 1) + "|"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test02.mp4")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=device,
    )

    cap = cv2.VideoCapture(args.source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "output_sahi.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             video_fps, (width, height))

    total_frames = 0
    detected_frames = 0
    total_boxes = 0
    conf_sum = 0.0
    max_conf = 0.0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            rgb,
            detection_model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )

        found = False
        for obj in result.object_prediction_list:
            if obj.category.id != AIRPLANE_CLASS_ID:
                continue
            found = True
            score = obj.score.value
            total_boxes += 1
            conf_sum += score
            max_conf = max(max_conf, score)
            x1, y1, x2, y2 = (int(obj.bbox.minx), int(obj.bbox.miny),
                              int(obj.bbox.maxx), int(obj.bbox.maxy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"airplane {score:.0%}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if found:
            detected_frames += 1

        writer.write(frame)
        if not args.no_show:
            cv2.imshow("SAHI airplane detection (press Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    detection_rate = detected_frames / total_frames * 100 if total_frames else 0
    avg_conf = conf_sum / total_boxes if total_boxes else 0
    processing_fps = total_frames / elapsed if elapsed else 0
    line = "+" + "-" * (BOX_WIDTH - 2) + "+"

    print()
    print(line)
    print("|" + "SAHI DETECTION RESULTS".center(BOX_WIDTH - 2) + "|")
    print(line)
    print(table_row("Video", args.source))
    print(table_row("Model", args.model))
    print(table_row("Slice size", args.slice))
    print(table_row("Overlap", args.overlap))
    print(table_row("Confidence threshold", args.conf))
    print(table_row("Device", device))
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
    print(f"Output video saved as: {output_path}")


if __name__ == "__main__":
    main()