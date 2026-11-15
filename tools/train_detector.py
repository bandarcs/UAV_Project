"""Train one detector using the shared, validated training/evaluation pipeline.

Example: python tools/train_detector.py --data reviewed/data.yaml --model yolov8n.pt
No training is started by importing this module. Training requires reviewed labels.
"""
import sys
from compare_models import main

if __name__ == "__main__":
    if "--models" in sys.argv:
        raise SystemExit("Use compare_models.py for multiple models; use --model here.")
    if "--model" in sys.argv:
        sys.argv[sys.argv.index("--model")] = "--models"
    else:
        sys.argv.extend(["--models", "yolov8n.pt"])
    if "--split" not in sys.argv:
        sys.argv.extend(["--split", "val"])
    raise SystemExit(main())
