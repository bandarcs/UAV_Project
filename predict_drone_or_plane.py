import argparse
from collections import Counter

from ultralytics import YOLO

PLANE_WORDS = ("plane", "airplane", "aircraft", "jet", "fixed")
DRONE_WORDS = ("drone", "uav", "quad", "multirotor")


def to_group(name: str) -> str:
    n = name.lower()
    if any(w in n for w in PLANE_WORDS):
        return "طيارة"
    if any(w in n for w in DRONE_WORDS):
        return "درون"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.4, help="حد الثقة (0 إلى 1)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = int(args.source) if args.source.isdigit() else args.source

    counts = Counter()
    frames = 0
    results = model.predict(source=src, conf=args.conf, save=True, stream=True, verbose=False)
    for r in results:
        frames += 1
        for c, p in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            counts[to_group(r.names[c])] += 1
            if frames == 1:
                print(f"  اكتشاف: {r.names[c]}  (ثقة {p:.0%})")

    print("\n========== الحكم النهائي ==========")
    print(f"عدد الفريمات/الصور: {frames}")
    if not counts:
        print("ما تم اكتشاف شي. جرب تنزل --conf مثلاً 0.25")
    else:
        for k, v in counts.most_common():
            print(f"  {k}: {v} اكتشاف")
        print(f"\nالحكم: {counts.most_common(1)[0][0]}")
    print("الناتج المرسوم محفوظ في مجلد runs/detect/")


if __name__ == "__main__":
    main()