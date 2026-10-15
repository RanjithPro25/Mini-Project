from pathlib import Path
import csv
import json
import shutil
import time
import cv2
from ultralytics import YOLOE

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MANIFEST = PROJECT_DIR / "dataset_split" / "phase7_split_manifest.csv"
MODEL_DIR = PROJECT_DIR / "models"
YOLO_DATASET = PROJECT_DIR / "yolo_dataset"

IMAGE_OUT = YOLO_DATASET / "images"
LABEL_OUT = YOLO_DATASET / "labels"

PROGRESS_FILE = PROJECT_DIR / "phase8" / "phase8_progress.csv"
REVIEW_FILE = PROJECT_DIR / "phase8" / "phase8_review.csv"

MODEL = MODEL_DIR / "yoloe-11s-seg.pt"

TOTAL_EXPECTED = 36966

CONF = 0.05
IOU = 0.50
IMGSZ = 640

MIN_AREA = 0.003
MAX_AREA = 0.98

PROMPTS = [
    "snake",
    "snake animal",
    "reptile"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def write_progress(rows):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        PROGRESS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "destination",
                "species",
                "split",
                "status",
                "confidence",
                "box"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def load_progress():
    if not PROGRESS_FILE.exists():
        return {}

    data = {}

    with open(
        PROGRESS_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        for row in csv.DictReader(f):
            data[row["source"]] = row

    return data


def save_review(row):
    exists = REVIEW_FILE.exists()

    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        REVIEW_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "species",
                "split",
                "reason"
            ]
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def valid_label(path):
    if not path.exists():
        return False

    try:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            return False

        for line in text.splitlines():

            parts = line.split()

            if len(parts) != 5:
                return False

            values = [float(x) for x in parts]

            if int(values[0]) != 0:
                return False

            for value in values[1:]:
                if value < 0 or value > 1:
                    return False

        return True

    except Exception:
        return False


def mask_to_box(mask, width, height):

    ys, xs = mask.nonzero()

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    bw = x2 - x1
    bh = y2 - y1

    area = (bw * bh) / float(width * height)

    if area < MIN_AREA:
        return None

    if area > MAX_AREA:
        return None

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height

    wn = bw / width
    hn = bh / height

    return xc, yc, wn, hn, area


print("=" * 75)
print("PHASE 8 - YOLOE AUTOMATIC SNAKE ANNOTATION")
print("=" * 75)

print()
print("STEP 1: Checking files")

if not MANIFEST.exists():
    raise RuntimeError(
        f"Phase 7 manifest not found:\n{MANIFEST}"
    )

if not MODEL.exists():
    print()
    print("YOLOE model not found locally.")
    print("Ultralytics will download it automatically on first use.")
    print()
    print("Expected model:")
    print(MODEL)

print("Manifest:", MANIFEST)
print("Model:", MODEL)

print()
print("STEP 2: Reading Phase 7 manifest")

with open(
    MANIFEST,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    manifest_rows = list(csv.DictReader(f))

print(
    "Manifest records:",
    len(manifest_rows)
)

if len(manifest_rows) != TOTAL_EXPECTED:
    raise RuntimeError(
        f"Expected {TOTAL_EXPECTED} records, "
        f"found {len(manifest_rows)}"
    )

required_columns = {
    "source",
    "destination",
    "species",
    "split"
}

missing_columns = required_columns - set(manifest_rows[0].keys())

if missing_columns:
    raise RuntimeError(
        f"Missing manifest columns: {missing_columns}"
    )

print("Dataset count: 36,966 VERIFIED")

print()
print("STEP 3: Preparing output folders")

for split in ["train", "val", "test"]:
    (IMAGE_OUT / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (LABEL_OUT / split).mkdir(
        parents=True,
        exist_ok=True
    )

print("Output folders ready")

print()
print("STEP 4: Loading existing progress")

progress = load_progress()

print(
    "Previous progress records:",
    len(progress)
)

print()
print("STEP 5: Loading YOLOE")

start_model = time.time()

model = YOLOE(
    "yoloe-11s-seg.pt"
)

print(
    "YOLOE loaded in:",
    round(time.time() - start_model, 2),
    "seconds"
)

print()
print("STEP 6: Setting snake prompts")

model.set_classes(PROMPTS)

print("Prompts:", ", ".join(PROMPTS))

print()
print("STEP 7: Starting annotation")
print("CPU mode")
print("Image size:", IMGSZ)
print("Confidence:", CONF)
print()

results = []

accepted = 0
review = 0
failed = 0
skipped = 0
processed = 0

start_all = time.time()

for index, row in enumerate(
    manifest_rows,
    start=1
):

    source = Path(row["source"])

    split = row["split"]
    species = row["species"]

    source_key = str(source)

    destination_name = source.name

    image_destination = (
        IMAGE_OUT
        / split
        / species
        / destination_name
    )

    label_destination = (
        LABEL_OUT
        / split
        / species
        / f"{Path(destination_name).stem}.txt"
    )

    image_destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    label_destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if valid_label(label_destination):

        skipped += 1

        results.append({
            "source": source_key,
            "destination": str(image_destination),
            "species": species,
            "split": split,
            "status": "existing",
            "confidence": "",
            "box": ""
        })

        if index % 100 == 0:

            print(
                f"[{index:05d}/{TOTAL_EXPECTED}] "
                f"SKIP | existing label | "
                f"accepted={accepted} "
                f"review={review} "
                f"failed={failed}"
            )

        continue

    if not source.exists():

        failed += 1

        results.append({
            "source": source_key,
            "destination": str(image_destination),
            "species": species,
            "split": split,
            "status": "missing_source",
            "confidence": "",
            "box": ""
        })

        save_review({
            "source": source_key,
            "species": species,
            "split": split,
            "reason": "missing_source"
        })

        continue

    try:

        image = cv2.imread(
            str(source),
            cv2.IMREAD_COLOR
        )

        if image is None:

            failed += 1

            results.append({
                "source": source_key,
                "destination": str(image_destination),
                "species": species,
                "split": split,
                "status": "unreadable",
                "confidence": "",
                "box": ""
            })

            save_review({
                "source": source_key,
                "species": species,
                "split": split,
                "reason": "unreadable"
            })

            continue

        height, width = image.shape[:2]

        prediction = model.predict(
            source=image,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            device="cpu",
            verbose=False
        )[0]

        best_box = None
        best_conf = 0.0

        if (
            prediction.boxes is not None
            and len(prediction.boxes) > 0
            and prediction.masks is not None
        ):

            boxes = prediction.boxes
            masks = prediction.masks.data.cpu().numpy()

            confidences = (
                boxes.conf.cpu().numpy()
            )

            for mask, confidence in zip(
                masks,
                confidences
            ):

                box = mask_to_box(
                    mask > 0.5,
                    width,
                    height
                )

                if box is None:
                    continue

                if float(confidence) > best_conf:

                    best_conf = float(confidence)
                    best_box = box

        if best_box is None:

            review += 1

            results.append({
                "source": source_key,
                "destination": str(image_destination),
                "species": species,
                "split": split,
                "status": "review",
                "confidence": "",
                "box": ""
            })

            save_review({
                "source": source_key,
                "species": species,
                "split": split,
                "reason": "no_valid_snake_box"
            })

            status = "REVIEW"

        else:

            xc, yc, wn, hn, area = best_box

            label_destination.write_text(
                f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n",
                encoding="utf-8"
            )

            if not image_destination.exists():

                shutil.copy2(
                    source,
                    image_destination
                )

            accepted += 1

            results.append({
                "source": source_key,
                "destination": str(image_destination),
                "species": species,
                "split": split,
                "status": "accepted",
                "confidence": f"{best_conf:.6f}",
                "box": (
                    f"{xc:.6f},"
                    f"{yc:.6f},"
                    f"{wn:.6f},"
                    f"{hn:.6f}"
                )
            })

            status = "ACCEPTED"

        processed += 1

        if processed % 10 == 0 or index == 1:

            elapsed = time.time() - start_all

            speed = processed / elapsed if elapsed > 0 else 0

            remaining = TOTAL_EXPECTED - index

            eta = (
                remaining / speed
                if speed > 0
                else 0
            )

            print(
                f"[{index:05d}/{TOTAL_EXPECTED}] "
                f"{status} | "
                f"{processed} new | "
                f"accepted={accepted} "
                f"review={review} "
                f"failed={failed} | "
                f"{speed:.2f} img/s | "
                f"ETA {eta / 60:.1f} min"
            )

        if processed % 10 == 0:

            write_progress(results)

    except KeyboardInterrupt:

        print()
        print("STOPPED BY USER")
        print("Saving progress...")

        write_progress(results)

        print(
            "Progress saved:",
            len(results)
        )

        raise

    except Exception as e:

        failed += 1

        results.append({
            "source": source_key,
            "destination": str(image_destination),
            "species": species,
            "split": split,
            "status": "failed",
            "confidence": "",
            "box": ""
        })

        save_review({
            "source": source_key,
            "species": species,
            "split": split,
            "reason": f"exception: {type(e).__name__}: {e}"
        })

        print(
            f"[{index:05d}/{TOTAL_EXPECTED}] "
            f"FAILED | {type(e).__name__}"
        )

write_progress(results)

elapsed_total = time.time() - start_all

print()
print("=" * 75)
print("PHASE 8 COMPLETE")
print("=" * 75)

print("Manifest images :", TOTAL_EXPECTED)
print("Existing labels :", skipped)
print("New processed   :", processed)
print("Accepted        :", accepted)
print("Review          :", review)
print("Failed          :", failed)

print(
    "Total time      :",
    round(elapsed_total / 60, 2),
    "minutes"
)

print()
print("Progress file:")
print(PROGRESS_FILE)

print()
print("Review file:")
print(REVIEW_FILE)

print()
print("PHASE 8 FINISHED SUCCESSFULLY")