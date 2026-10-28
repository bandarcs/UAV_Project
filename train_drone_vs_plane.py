import argparse

import torch
import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="مسار ملف data.yaml حق الداتاسيت")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    print("الكلاسات في الداتاسيت:", cfg.get("names"))

    device = 0 if torch.cuda.is_available() else "cpu"
    print("الجهاز:", "GPU" if device == 0 else "CPU (بطيء)")

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=2, project="runs_drone_plane", name="train", exist_ok=True,
    )

    split = "test" if cfg.get("test") else "val"
    best_path = model.trainer.best
    best = YOLO(best_path)
    m = best.val(data=args.data, split=split, imgsz=args.imgsz, device=device,
                 project="runs_drone_plane", name="evaluation", exist_ok=True)

    print("\n================ نتائج التقييم ================")
    print(f"البيانات المستخدمة للتقييم: {split}")
    print(f"Precision : {m.box.mp:.3f}")
    print(f"Recall    : {m.box.mr:.3f}")
    print(f"mAP50     : {m.box.map50:.3f}")
    print(f"mAP50-95  : {m.box.map:.3f}")
    print("\nأفضل أوزان محفوظة في:", best_path)
    print("مصفوفة الالتباس في: runs_drone_plane/evaluation/confusion_matrix.png")


if __name__ == "__main__":
    main()