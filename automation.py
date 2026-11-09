"""Automatically configure a local recorded-video research experiment.

Only the video is required. Model discovery, class checks, frame-size trials,
device selection and analysis scheduling are performed by this program.
Selection is a documented resource heuristic, not a measured accuracy claim.
No training, network camera, flight control or threat assessment is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import statistics
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

TARGET_ALIASES = {
    "airplane": {"airplane", "aeroplane", "aircraft", "plane"},
    "helicopter": {"helicopter", "chopper"},
    "drone": {"drone", "uav", "quadcopter", "quadrotor"},
    "bird": {"bird"},
}
POLICY = {
    "version": "offline-resource-policy-v2",
    "max_analysis_fps": 30,
    "confidence": 0.25,
    "sample_fractions": [0.1, 0.5, 0.9],
    "sizes": [640, 960, 1280],
    "timing_margin": 1.25,
    "schedule_window": 20,
    "output_long_edge": 1280,
}


@dataclass
class RateController:
    """Increase stride promptly; recover after two consistently faster windows."""
    source_fps: float
    stride: int
    lower_windows: int = 0

    def update(self, median_ms):
        suggested = max(1, math.ceil(self.source_fps / POLICY["max_analysis_fps"]),
                        math.ceil(self.source_fps * median_ms * POLICY["timing_margin"] / 1000))
        if suggested > self.stride:
            self.stride, self.lower_windows = suggested, 0
            return self.stride
        if suggested < self.stride:
            self.lower_windows += 1
            if self.lower_windows >= 2:
                self.stride, self.lower_windows = suggested, 0
                return self.stride
        else:
            self.lower_windows = 0
        return None


def class_mapping(names):
    """Match explicit class names; never invent a class missing from weights."""
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for index, name in pairs:
        normal = str(name).strip().lower().replace("-", "").replace("_", "")
        for canonical, aliases in TARGET_ALIASES.items():
            if normal in aliases:
                mapping[int(index)] = canonical
                break
    return mapping


def select_profile(profiles, source_fps, difficult):
    """Select within measured time budget; confidence counts are not a score."""
    good = [p for p in profiles if p.get("ok") and p["latency_ms"] > 0
            and math.isfinite(p["latency_ms"]) and p["supported_classes"]]
    if not good:
        raise ValueError("No local detection model completed a valid calibration trial.")
    # Restrict comparisons to one supported class set. A faster model must not
    # silently remove a class that another compatible model supports.
    groups = {}
    for p in good:
        groups.setdefault(tuple(sorted(p["supported_classes"])), []).append(p)
    supported = max(sorted(groups), key=lambda key: (len(key), len(groups[key])))
    comparable = groups[supported]
    target_fps = min(source_fps, POLICY["max_analysis_fps"])
    budget_ms = 1000 / target_fps
    feasible = [p for p in comparable
                if p["latency_ms"] * POLICY["timing_margin"] <= budget_ms]
    if feasible:
        # Size/model preferences are an engineering heuristic. Without reviewed
        # ground truth, a larger model or input size is not proof of accuracy.
        if difficult:
            selected = max(feasible, key=lambda p: (p["parameters"], p["imgsz"], -p["latency_ms"]))
        else:
            selected = max(feasible, key=lambda p: (p["imgsz"], p["parameters"], -p["latency_ms"]))
        reason = "Measured timing fits the source-rate budget; detail/model preference is heuristic."
    else:
        selected = min(comparable, key=lambda p: p["latency_ms"])
        reason = "No profile meets the timing budget; selected the fastest compatible profile."
    stride = max(1, math.ceil(source_fps / POLICY["max_analysis_fps"]),
                 math.ceil(source_fps * selected["latency_ms"] * POLICY["timing_margin"] / 1000))
    return dict(selected), stride, reason


def checked_video(value):
    if "://" in str(value) or str(value).isdigit():
        raise ValueError("Use a local recorded video file, not a camera number or stream URL.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported recorded-video extension.")
    return path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def run(video: Path, output: Path, project_root: Path, *, frame_source=read_frames, calibration_end=None):
    import cv2
    import torch
    from ultralytics import YOLO

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "initialising", "source": str(video), "policy": POLICY,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "purpose": "Offline academic recorded-video experiment",
              "accuracy_evaluated": False}
    status_path = output / "summary.json"
    write_json(status_path, status)
    cap = cv2.VideoCapture(str(video))
    writer = None
    frame_iterator = None
    counts = {"decoded_frames": 0, "analysed_frames": 0, "skipped_frames": 0,
              "written_frames": 0, "detection_rows": 0}
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open the recorded video.")
        width, height = [int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width < 2 or height < 2 or not math.isfinite(fps) or fps <= 0 or total < 1:
            raise ValueError("Video dimensions, frame count and a valid frame rate are required.")
        # In a replay of a live session, startup can sample only the
        # frames available before the first simulated connection loss.
        sample_count = total if calibration_end is None else min(total, calibration_end)
        positions = sorted({round((sample_count - 1) * fraction) for fraction in POLICY["sample_fractions"]})
        status["calibration_sample_frames"] = positions
        samples, quality = [], []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                continue
            samples.append(frame)
            scale = min(1.0, 640 / max(frame.shape[:2]))
            probe = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
            quality.append({"brightness": float(gray.mean()), "contrast": float(gray.std()),
                            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
        if not samples:
            raise ValueError("No readable calibration frames were found.")
        means = {key: statistics.mean(q[key] for q in quality) for key in quality[0]}
        difficult = means["brightness"] < 45 or means["brightness"] > 210 or means["contrast"] < 25 or means["laplacian_variance"] < 70
        status.update(source_properties={"width": width, "height": height, "fps": fps, "reported_frames": total},
                      image_quality={**means, "low_visibility_heuristic": difficult,
                                     "note": "Quality thresholds are heuristic, not a ground-truth quality rating."})
        device = 0 if torch.cuda.is_available() else "cpu"
        status["device"] = str(device)
        def sync():
            if device != "cpu":
                torch.cuda.synchronize()
        def release():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Only local, trusted weights are loaded. No weights/data are downloaded.
        paths = sorted(project_root.glob("yolov8*.pt"))
        paths += sorted((project_root / "weights" / "academic").glob("*.pt"))
        if not paths:
            raise ValueError("No local YOLOv8 or weights/academic/*.pt detection weights were found.")
        sizes = [s for s in POLICY["sizes"] if s == 640 or max(width, height) >= s]
        profiles = []
        for path in paths:
            model = None
            try:
                model = YOLO(str(path))
                if model.task != "detect":
                    profiles.append({"weights": str(path), "ok": False, "error": "Not a detection model."})
                    continue
                mapping = class_mapping(model.names)
                if not mapping:
                    profiles.append({"weights": str(path), "ok": False, "error": "No requested classes in weights."})
                    continue
                parameters = sum(p.numel() for p in model.model.parameters())
                for size in sizes:
                    trial = {"weights": str(path), "imgsz": size, "parameters": parameters,
                             "supported_classes": sorted(set(mapping.values()))}
                    try:
                        kwargs = dict(imgsz=size, conf=POLICY["confidence"], classes=sorted(mapping), device=device, verbose=False)
                        model.predict(samples[0], **kwargs)  # Unmeasured warm-up.
                        times = []
                        for frame in samples:
                            sync()
                            tick = time.perf_counter()
                            result = model.predict(frame, **kwargs)[0]
                            sync()
                            times.append((time.perf_counter() - tick) * 1000)
                            del result
                        trial.update(ok=True, latency_ms=statistics.median(times))
                        print(f"Calibrated {path.name}, size {size}: {trial['latency_ms']:.1f} ms", flush=True)
                    except Exception as exc:
                        trial.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    profiles.append(trial)
                    write_json(output / "calibration.json", profiles)
            except Exception as exc:
                profiles.append({"weights": str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                del model
                release()
        write_json(output / "calibration.json", profiles)
        selected, stride, reason = select_profile(profiles, fps, difficult)
        model = YOLO(selected["weights"])
        mapping = class_mapping(model.names)
        missing = sorted(set(TARGET_ALIASES) - set(mapping.values()))
        status.update(selection=selected, selection_reason=reason, supported_classes=sorted(set(mapping.values())),
                      unsupported_classes=missing, initial_stride=stride,
                      class_coverage_complete=not missing,
                      calibration_seconds=time.perf_counter() - started,
                      note="No accuracy or superiority claim. Unsupported classes cannot be recognised with these weights.")
        write_json(output / "decision.json", status)
        print(f"Selected {Path(selected['weights']).name}, size {selected['imgsz']}, stride {stride}", flush=True)
        if missing:
            print("Weights do not support: " + ", ".join(missing), flush=True)
        # Reopen to ensure the entire original video, not a seek remainder, is read.
        cap.release()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError("Could not reopen source after calibration.")
        scale = min(1.0, POLICY["output_long_edge"] / max(width, height))
        output_size = tuple(max(2, int(dim * scale) // 2 * 2) for dim in (width, height))
        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
        if not writer.isOpened():
            raise ValueError("Could not create output video.")
        window, pending_stride = [], None
        rate = RateController(fps, stride)
        processing_started = time.perf_counter()
        with (output / "detections.csv").open("w", newline="", encoding="utf-8") as detections, (output / "frames.csv").open("w", newline="", encoding="utf-8") as frames, (output / "decisions.csv").open("w", newline="", encoding="utf-8") as decisions:
            dw, fw, sw = csv.writer(detections), csv.writer(frames), csv.writer(decisions)
            dw.writerow(["frame", "time_seconds", "class", "confidence", "track_id", "x1", "y1", "x2", "y2"])
            fw.writerow(["frame", "analysed", "stride", "detections", "analysis_ms"])
            sw.writerow(["effective_frame", "stride", "reason"])
            sw.writerow([0, stride, "startup calibration"])
            frame_iterator = iter(frame_source(cap))
            for frame in frame_iterator:
                if frame is None or not frame.size:
                    raise ValueError("An empty decoded frame was encountered.")
                index = counts["decoded_frames"]
                counts["decoded_frames"] += 1
                analysed = index % stride == 0
                count, analysis_ms = 0, None
                if analysed:
                    sync()
                    tick = time.perf_counter()
                    result = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=selected["imgsz"],
                                         conf=POLICY["confidence"], classes=sorted(mapping), device=device, verbose=False)[0]
                    boxes = result.boxes
                    if boxes is not None:
                        xyxy = boxes.xyxy.cpu().tolist()
                        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [""] * len(boxes)
                        for cls, confidence, coords, identity in zip(boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), xyxy, ids):
                            dw.writerow([index, index / fps, mapping[cls], confidence, identity, *coords])
                        count = len(boxes)
                    # Keep labels in a compact legend: large model-generated
                    # labels can cover nearby small objects and one another.
                    display = result.plot(labels=False, line_width=2)
                    legend = []
                    if boxes is not None:
                        for cls, confidence, identity in zip(boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
                            legend.append(f"ID {identity or '-'}  {mapping[cls]}  {confidence:.0%}")
                    sync()
                    analysis_ms = (time.perf_counter() - tick) * 1000
                    window.append(analysis_ms)
                    counts["analysed_frames"] += 1
                    counts["detection_rows"] += count
                    del result
                    if len(window) == POLICY["schedule_window"]:
                        pending_stride = rate.update(statistics.median(window))
                        window.clear()
                else:
                    display = frame.copy()
                    counts["skipped_frames"] += 1
                # Unanalysed frames stay in the output timeline and are marked;
                # stale boxes are never copied onto frames not actually analysed.
                display = cv2.resize(display, output_size, interpolation=cv2.INTER_AREA)
                if analysed and legend:
                    lines = legend[:8]
                    if len(legend) > 8:
                        lines.append(f"+ {len(legend) - 8} more detections")
                    for row, label in enumerate(lines):
                        y = 22 + row * 22
                        cv2.rectangle(display, (5, y - 17), (min(output_size[0]-5, 290), y + 4), (25, 35, 30), -1)
                        cv2.putText(display, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
                if not analysed:
                    cv2.putText(display, "Not analysed (automatic scheduling)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                if missing:
                    # A successful file-processing run must not look like a
                    # validated four-class classifier when weights lack classes.
                    text = "Limited classes: " + ", ".join(status["supported_classes"])
                    y = output_size[1] - 12
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                writer.write(display)
                counts["written_frames"] += 1
                fw.writerow([index, int(analysed), stride, count if analysed else "", analysis_ms if analysed else ""])
                if pending_stride is not None:
                    stride, pending_stride = pending_stride, None
                    sw.writerow([index + 1, stride, "rolling measured analysis latency"])
                if counts["decoded_frames"] % 100 == 0:
                    print(f"Decoded {counts['decoded_frames']}/{total} frames", flush=True)
        elapsed = time.perf_counter() - processing_started
        status.update(status="completed" if counts["decoded_frames"] >= total else "completed_with_decode_warning", **counts, processing_seconds=elapsed,
                      end_to_end_processing_fps=counts["decoded_frames"] / elapsed,
                      analysed_frames_per_second=counts["analysed_frames"] / elapsed,
                      decode_ended_before_reported_count=counts["decoded_frames"] < total,
                      output_video_size=list(output_size), final_stride=stride,
                      total_seconds=time.perf_counter() - started)
        write_json(status_path, status)
        return status
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}", **counts)
        write_json(status_path, status)
        raise
    finally:
        close_frames = getattr(frame_iterator, "close", None)
        if close_frames is not None:
            close_frames()
        cap.release()
        if writer is not None:
            writer.release()


def open_result(video: Path) -> bool:
    """Open a completed local result; playback failure does not erase results."""
    try:
        if sys.platform == "win32":
            os.startfile(str(video.resolve()))
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(video.resolve())])
        return True
    except OSError as exc:
        print(f"Could not open the video player: {exc}. Open this file manually: {video}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Local recorded video. No model or image-size choice is required.")
    parser.add_argument("--output-dir", type=Path, help="Optional new results directory.")
    parser.add_argument("--no-open", action="store_true", help="Save results without opening the video player (batch runs).")
    args = parser.parse_args()
    try:
        video = checked_video(args.video)
        root = Path(__file__).resolve().parent
        output = args.output_dir or root / "runs" / "automation" / f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        result = run(video, output, root)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Automation stopped: {exc}\n")
    print(json.dumps({"status": result["status"], "weights": result["selection"]["weights"],
                      "imgsz": result["selection"]["imgsz"], "analysed_frames": result["analysed_frames"],
                      "unsupported_classes": result["unsupported_classes"], "output": str(output)}, indent=2))
    print(f"Annotated video: {output / 'annotated.mp4'}")
    print(f"Detection table: {output / 'detections.csv'}")
    if not args.no_open and result["written_frames"]:
        open_result(output / "annotated.mp4")


if __name__ == "__main__":
    main()
