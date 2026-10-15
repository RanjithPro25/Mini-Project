from pathlib import Path
import csv
import json
import shutil
import time
import cv2
import torch
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_FILE = PROJECT_DIR / "models" / "yolov8s-world.pt"
SPLIT_MANIFEST = PROJECT_DIR / "dataset_split" / "phase7_split_manifest.csv"

YOLO_DIR = PROJECT_DIR / "yolo_dataset"
IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

MANIFEST_FILE = YOLO_DIR / "phase8_fast_manifest.csv"
PROGRESS_FILE = YOLO_DIR / "phase8_fast_progress.json"

PROMPTS = ["snake animal", "snake", "reptile", "a snake"]

CONF = 0.08
IMGSZ = 640
NMS_IOU = 0.50

MIN_CONF = 0.20
MIN_AREA = 0.003
MAX_AREA = 0.98

for split in ["train", "val", "test"]:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("PHASE 8 - FAST AUTOMATIC BOUNDING BOX GENERATION")
print("=" * 70)
print(f"Device : {device}")
print(f"Model  : {MODEL_FILE}")
print(f"Size   : {IMGSZ}")
print(f"Conf   : {CONF}")
print()

if not SPLIT_MANIFEST.exists():
    raise FileNotFoundError(
        f"Missing Phase 7 manifest:\n{SPLIT_MANIFEST}"
    )

model = YOLOWorld(str(MODEL_FILE))
model.set_classes(PROMPTS)

print("YOLO-World loaded")
print("Prompts:", ", ".join(PROMPTS))
print()

manifest = {}

if MANIFEST_FILE.exists():
    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            key = f"{row['split']}|{row['destination']}"
            manifest[key] = row

def save_manifest():

    fields = [
        "split",
        "destination",
        "species",
        "label",
        "status",
        "confidence",
        "source"
    ]

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(manifest.values())

def save_progress():

    counts = {
        "existing": 0,
        "accepted": 0,
        "review": 0,
        "no_detection": 0,
        "failed": 0
    }

    for row in manifest.values():

        status = row.get("status", "")

        if status in counts:
            counts[status] += 1

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "total": len(manifest),
                "counts": counts,
                "updated": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            },
            f,
            indent=2
        )

def valid_label(label_file):

    if not label_file.exists():
        return False

    try:

        text = label_file.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return False

        for line in text.splitlines():

            p = line.split()

            if len(p) != 5:
                return False

            v = [float(x) for x in p]

            if int(v[0]) != 0:
                return False

            if not all(
                0 <= x <= 1
                for x in v[1:]
            ):
                return False

        return True

    except:
        return False

def save_label(
    label_file,
    box,
    width,
    height
):

    x1, y1, x2, y2 = box

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))

    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return False

    area = (
        (x2 - x1) *
        (y2 - y1)
    )

    ratio = area / (width * height)

    if ratio < MIN_AREA:
        return False

    if ratio > MAX_AREA:
        return False

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height

    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    label_file.write_text(
        f"0 {xc:.6f} {yc:.6f} "
        f"{bw:.6f} {bh:.6f}\n",
        encoding="utf-8"
    )

    return True

tasks = []

with open(
    SPLIT_MANIFEST,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        split = row["split"].strip()

        if split not in {
            "train",
            "val",
            "test"
        }:
            continue

        source = Path(
            row["source"].strip()
        )

        destination = Path(
            row["destination"].strip()
        )

        if not source.exists():
            continue

        tasks.append(
            {
                "source": source,
                "destination": destination,
                "species": row["species"].strip(),
                "split": split
            }
        )

if len(tasks) != 36966:
    raise RuntimeError(
        f"Expected 36,966 images, "
        f"found {len(tasks):,}. "
        f"Stopping for safety."
    )

print(
    "Exact Phase 7 dataset: 36,966 images"
)
print()

existing_images = 0
copied_images = 0

for task in tasks:

    split = task["split"]
    source = task["source"]

    destination = (
        IMAGES_DIR /
        split /
        source.name
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        existing_images += 1

    else:

        try:

            temp = destination.with_suffix(
                destination.suffix + ".part"
            )

            shutil.copy2(
                source,
                temp
            )

            temp.replace(destination)

            copied_images += 1

        except:
            pass

print(
    f"Existing YOLO images : {existing_images:,}"
)

print(
    f"New images copied     : {copied_images:,}"
)

print()

total = len(tasks)

existing = 0
accepted = 0
review = 0
no_detection = 0
failed = 0

start_time = time.time()

try:

    for index, task in enumerate(
        tasks,
        1
    ):

        split = task["split"]
        source = task["source"]
        species = task["species"]

        image_name = source.name

        image_file = (
            IMAGES_DIR /
            split /
            image_name
        )

        label_file = (
            LABELS_DIR /
            split /
            f"{source.stem}.txt"
        )

        key = (
            f"{split}|"
            f"{task['destination']}"
        )

        if valid_label(label_file):

            existing += 1

            manifest[key] = {
                "split": split,
                "destination": str(
                    task["destination"]
                ),
                "species": species,
                "label": str(
                    label_file.relative_to(
                        YOLO_DIR
                    )
                ),
                "status": "existing",
                "confidence": "existing",
                "source": str(source)
            }

            print(
                f"[{index:05d}/{total}] "
                f"KEEP | {species} | "
                f"{image_name}",
                flush=True
            )

            continue

        print(
            f"[{index:05d}/{total}] "
            f"PROCESSING | {species} | "
            f"{image_name}",
            flush=True
        )

        image_start = time.time()

        try:

            image = cv2.imread(
                str(image_file)
            )

            if image is None:
                raise RuntimeError(
                    "Image read failed"
                )

            height, width = image.shape[:2]

            result = model.predict(
                source=str(image_file),
                conf=CONF,
                iou=NMS_IOU,
                imgsz=IMGSZ,
                device=device,
                verbose=False
            )[0]

            if (
                result.boxes is None
                or
                len(result.boxes) == 0
            ):

                manifest[key] = {
                    "split": split,
                    "destination": str(
                        task["destination"]
                    ),
                    "species": species,
                    "label": "",
                    "status": "no_detection",
                    "confidence": "0",
                    "source": str(source)
                }

                no_detection += 1

                print(
                    "    NO DETECTION",
                    flush=True
                )

            else:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confs = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                best = int(
                    confs.argmax()
                )

                box = boxes[best]
                confidence = float(
                    confs[best]
                )

                if confidence < MIN_CONF:

                    status = "review"

                else:

                    saved = save_label(
                        label_file,
                        box,
                        width,
                        height
                    )

                    if saved:
                        status = "accepted"
                    else:
                        status = "review"

                if status == "accepted":

                    accepted += 1

                    print(
                        f"    ACCEPTED | "
                        f"conf={confidence:.3f}",
                        flush=True
                    )

                else:

                    review += 1

                    print(
                        f"    REVIEW | "
                        f"conf={confidence:.3f}",
                        flush=True
                    )

                manifest[key] = {
                    "split": split,
                    "destination": str(
                        task["destination"]
                    ),
                    "species": species,
                    "label": (
                        str(
                            label_file.relative_to(
                                YOLO_DIR
                            )
                        )
                        if status == "accepted"
                        else ""
                    ),
                    "status": status,
                    "confidence": f"{confidence:.6f}",
                    "source": str(source)
                }

        except Exception as e:

            failed += 1

            manifest[key] = {
                "split": split,
                "destination": str(
                    task["destination"]
                ),
                "species": species,
                "label": "",
                "status": "failed",
                "confidence": "error",
                "source": str(source)
            }

            print(
                f"    FAILED | {e}",
                flush=True
            )

        save_manifest()
        save_progress()

        completed = (
            existing +
            accepted +
            review +
            no_detection +
            failed
        )

        elapsed = time.time() - start_time

        if completed > 0:

            average = (
                elapsed / completed
            )

            remaining = (
                total - completed
            )

            eta = (
                average *
                remaining
            )

            print(
                f"    {completed:,}/{total:,} "
                f"({completed / total * 100:.2f}%) "
                f"| {time.time() - image_start:.1f}s "
                f"| ETA {eta / 60:.1f} min",
                flush=True
            )

except KeyboardInterrupt:

    save_manifest()
    save_progress()

    print()
    print("=" * 70)
    print("INTERRUPTED SAFELY")
    print("=" * 70)
    print("Progress saved.")
    print(
        "Run the same command again to continue."
    )

    raise SystemExit(0)

save_manifest()
save_progress()

print()
print("=" * 70)
print("PHASE 8 FINAL SUMMARY")
print("=" * 70)

print(
    f"Total images       : {total:,}"
)

print(
    f"Existing kept      : {existing:,}"
)

print(
    f"Accepted           : {accepted:,}"
)

print(
    f"Review              : {review:,}"
)

print(
    f"No detection        : {no_detection:,}"
)

print(
    f"Failed              : {failed:,}"
)

print()
print(
    f"Manifest: {MANIFEST_FILE}"
)

print(
    f"Progress: {PROGRESS_FILE}"
)

print()
print(
    "PHASE 8 FINISHED SUCCESSFULLY"
)