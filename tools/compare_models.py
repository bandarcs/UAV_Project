"""Train and compare Ultralytics YOLO models on an explicitly selected dataset.

A dataset YAML must be provided with --data. Review its class names before
running training; a dataset photographed from the air need not label airborne
objects. This script does not select a dataset automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install the project requirements first, for example: "
        "pip install ultralytics"
    ) from exc


CSV_FIELDS = [
    "model",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "inference_ms",
    "fps_inference",
    "params_M",
    "train_minutes",
    "epochs",
    "train_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and compare YOLO models on an explicitly selected labelled dataset."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="One or more pretrained YOLO weights/model names.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Explicit dataset YAML. Check that its annotated classes match the experiment.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to use (0 < fraction <= 1).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, such as 0, 0,1, cpu. Defaults to GPU 0 when available.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Dataset split for final evaluation.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold used during final detection evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project",
        default="runs_model_comparison",
        help="Folder where per-model training runs are saved.",
    )
    parser.add_argument(
        "--output",
        default="results/compare_results.csv",
        help="CSV file for the comparison results.",
    )
    return parser.parse_args()


def metric_value(metrics: Any, name: str) -> float:
    """Read an Ultralytics detection metric and return a plain float."""
    value = getattr(metrics.box, name, None)
    if value is None:
        return math.nan
    return float(value)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The file header is written once before training; use plain UTF-8 here so
    # appending rows does not insert another BOM before each model name.
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def save_training_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)  # يُستبدل السجل بعد اكتمال كتابته حتى يبقى الملف قابلًا للقراءة.
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.imgsz < 1:
        raise SystemExit("--imgsz must be greater than 0.")
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1.")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be greater than 0 and at most 1.")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")

    device: int | str = args.device if args.device is not None else (
        0 if torch.cuda.is_available() else "cpu"
    )
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit("Results CSV already exists; choose another --output.")
    if not Path(args.data).is_file():
        raise SystemExit("Dataset YAML must exist locally; review it before training.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Each invocation starts a fresh CSV. Results are written after every model,
    # so completed comparisons remain available if a later model fails.
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []

    print(f"Dataset: {args.data}")
    print(f"Final evaluation split: {args.split}")
    print(f"Device: {torch.cuda.get_device_name(0) if device == 0 else device}")
    print(f"Training fraction: {args.fraction:g}")
    print(f"Results: {output_path.resolve()}")
    print("The first run may download and prepare the dataset and pretrained weights.\n")

    # Prepare the selected dataset before timing model training so its setup
    # does not inflate the first model's time.
    print("Checking/preparing dataset...")
    try:
        check_det_dataset(args.data, autodownload=False)
    except Exception as exc:
        raise SystemExit(
            f"Could not load dataset '{args.data}': {type(exc).__name__}: {exc}"
        ) from exc

    # Create/truncate the CSV and write the header before the potentially long run.
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.DictWriter(file, fieldnames=CSV_FIELDS).writeheader()

    for model_name in args.models:
        model_slug = Path(model_name).stem.replace(" ", "_")
        run_name = f"{run_stamp}_{model_slug}"
        print(f"=== {model_name}: training ===")

        try:
            # Prefer the project's organised copy when a bare model name is supplied.
            requested = Path(model_name)
            saved_model = Path(__file__).resolve().parents[1] / "weights" / requested.name
            if requested.name == model_name and not requested.is_file() and saved_model.is_file():
                model_name = str(saved_model)
            model = YOLO(model_name)
            train_start = time.perf_counter()
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                fraction=args.fraction,
                workers=args.workers,
                device=device,
                seed=args.seed,
                deterministic=True,
                project=args.project,
                name=run_name,
                exist_ok=False,
                plots=False,
                verbose=False,
            )
            train_minutes = (time.perf_counter() - train_start) / 60.0

            trainer = model.trainer
            best_path = Path(trainer.best)
            if not best_path.is_file():
                last_path = Path(trainer.last)
                if not last_path.is_file():
                    raise FileNotFoundError(
                        f"Training finished but no best.pt or last.pt was found in {trainer.save_dir}."
                    )
                best_path = last_path

            epochs_completed = max(0, int(getattr(trainer, "epoch", -1)) + 1)
            record_path = Path(trainer.save_dir) / "training_run.json"
            record = {
                "stage": "trained",
                "dataset_yaml": str(Path(args.data).resolve()),
                "starting_model": model_name,
                "checkpoint": str(best_path.resolve()),
                "epochs_completed": epochs_completed,
                "input_size": args.imgsz,
                "seed": args.seed,
                "evaluation_split": args.split,
                "results_csv": str(output_path.resolve()),
            }
            save_training_record(record_path, record)  # يربط النموذج المحفوظ ببياناته حتى لو تعثّر التقييم.

            print(f"=== {model_name}: evaluating on {args.split} split ===")
            trained_model = YOLO(str(best_path))
            metrics = trained_model.val(
                data=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device,
                conf=args.conf,
                plots=False,
                verbose=False,
                project=str(trainer.save_dir),  # تبقى نتائج التقييم بجانب ملفات تدريب النموذج لتسهيل مراجعتها.
                name=f"evaluation_{args.split}",
            )

            inference_ms = float(metrics.speed.get("inference", math.nan))
            fps_inference = 1000.0 / inference_ms if inference_ms > 0 else math.nan
            params_m = sum(parameter.numel() for parameter in trained_model.model.parameters()) / 1_000_000

            row: dict[str, Any] = {
                "model": model_name,
                "mAP50": metric_value(metrics, "map50"),
                "mAP50_95": metric_value(metrics, "map"),
                "precision": metric_value(metrics, "mp"),
                "recall": metric_value(metrics, "mr"),
                "inference_ms": inference_ms,
                "fps_inference": fps_inference,
                "params_M": params_m,
                "train_minutes": train_minutes,
                "epochs": epochs_completed,
                "train_fraction": args.fraction,
            }
            append_csv(output_path, row)
            record["stage"] = "evaluated"
            save_training_record(record_path, record)  # يوضح السجل أن قياس الأداء حُفظ في الجدول.
            rows.append(row)

            print(
                f"{model_name}: mAP50={row['mAP50']:.4f}, "
                f"mAP50-95={row['mAP50_95']:.4f}, "
                f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
                f"inference={inference_ms:.2f} ms ({fps_inference:.2f} FPS), "
                f"parameters={params_m:.2f}M, training={train_minutes:.1f} min, "
                f"epochs={epochs_completed}"
            )
        except Exception as exc:
            print(
                f"ERROR while processing {model_name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(f"Completed results, if any, are in: {output_path.resolve()}", file=sys.stderr)
            return 1

    print("\n================ Model comparison ================")
    print(
        f"{'Model':<18} {'mAP50':>9} {'mAP50-95':>10} {'Precision':>10} "
        f"{'Recall':>9} {'Infer ms':>10} {'FPS':>9} {'Params M':>10}"
    )
    for row in rows:
        print(
            f"{row['model']:<18} {row['mAP50']:>9.4f} {row['mAP50_95']:>10.4f} "
            f"{row['precision']:>10.4f} {row['recall']:>9.4f} "
            f"{row['inference_ms']:>10.2f} {row['fps_inference']:>9.2f} "
            f"{row['params_M']:>10.2f}"
        )
    print(f"\nCSV saved to: {output_path.resolve()}")
    print("Inference FPS is computed from Ultralytics' validation inference time only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
