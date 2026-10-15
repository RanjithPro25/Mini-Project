from pathlib import Path
import csv
import time
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MANIFEST = PROJECT_DIR / "dataset_split" / "phase7_split_manifest.csv"
MODEL = PROJECT_DIR / "models" / "yolov8s-world.pt"

print("=" * 70)
print("PHASE 8 CLEAN PIPELINE TEST")
print("=" * 70)

print("\nSTEP 1: Checking files")

if not MANIFEST.exists():
    raise RuntimeError(f"Manifest not found:\n{MANIFEST}")

if not MODEL.exists():
    raise RuntimeError(f"Model not found:\n{MODEL}")

print("Manifest :", MANIFEST)
print("Model    :", MODEL)

print("\nSTEP 2: Reading Phase 7 manifest")

with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Manifest records:", len(rows))

if len(rows) != 36966:
    raise RuntimeError(
        f"Expected exactly 36,966 records, found {len(rows)}"
    )

print("Dataset count: 36,966 VERIFIED")

print("\nSTEP 3: Checking first image")

row = rows[0]

source = Path(row["source"])
species = row["species"]
split = row["split"]

print("Species :", species)
print("Split   :", split)
print("Source  :", source)

if not source.exists():
    raise RuntimeError(f"Source image does not exist:\n{source}")

print("Image exists: YES")

print("\nSTEP 4: Loading YOLO-World")

start = time.time()

model = YOLOWorld(str(MODEL))

print("Model loaded in:", round(time.time() - start, 2), "seconds")

print("\nSTEP 5: Setting prompt")

model.set_classes(["snake"])

print("Prompt set")

print("\nSTEP 6: Running ONE inference")

start = time.time()

result = model.predict(
    source=str(source),
    conf=0.03,
    imgsz=640,
    device="cpu",
    verbose=False
)[0]

elapsed = time.time() - start

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Inference time:", round(elapsed, 3), "seconds")

if result.boxes is None or len(result.boxes) == 0:
    print("Detection: NONE")
else:
    print("Detections:", len(result.boxes))

    confs = result.boxes.conf.cpu().numpy()

    print(
        "Best confidence:",
        round(float(confs.max()), 4)
    )

    for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box

        print(
            f"Box {i + 1}: "
            f"x1={x1:.1f}, "
            f"y1={y1:.1f}, "
            f"x2={x2:.1f}, "
            f"y2={y2:.1f}"
        )

print("\n" + "=" * 70)
print("CLEAN PIPELINE TEST FINISHED")
print("=" * 70)