from pathlib import Path
import time
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL = PROJECT_DIR / "models" / "yolov8s-world.pt"

IMAGE = (
    PROJECT_DIR
    / "dataset_split"
    / "train"
    / "Acrochordus_granulatus"
    / "1338671ae468256f.jpg"
)

PROMPTS = [
    ["snake"],
    ["snake animal"],
    ["reptile"],
    ["a snake"],
    ["animal"],
    ["snake body"],
    ["snake reptile animal"]
]

THRESHOLDS = [0.03, 0.05, 0.08, 0.10]

print("=" * 70)
print("YOLO-WORLD PROMPT + THRESHOLD TEST")
print("=" * 70)

print("Model :", MODEL)
print("Image :", IMAGE)

if not MODEL.exists():
    raise RuntimeError(f"Model not found: {MODEL}")

if not IMAGE.exists():
    raise RuntimeError(f"Image not found: {IMAGE}")

print()
print("Loading model...")

model = YOLOWorld(str(MODEL))

print("Model loaded")
print()

for prompts in PROMPTS:

    print("-" * 70)
    print("PROMPT:", prompts)
    print("-" * 70)

    model.set_classes(prompts)

    for threshold in THRESHOLDS:

        start = time.time()

        result = model.predict(
            source=str(IMAGE),
            conf=threshold,
            imgsz=640,
            device="cpu",
            verbose=False
        )[0]

        elapsed = time.time() - start

        if result.boxes is None or len(result.boxes) == 0:
            print(
                f"conf={threshold:.2f} | "
                f"NONE | "
                f"{elapsed:.2f}s"
            )
        else:
            confs = result.boxes.conf.cpu().numpy()

            print(
                f"conf={threshold:.2f} | "
                f"detections={len(confs)} | "
                f"best={float(confs.max()):.4f} | "
                f"{elapsed:.2f}s"
            )

print()
print("=" * 70)
print("TEST FINISHED")
print("=" * 70)