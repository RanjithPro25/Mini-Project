from pathlib import Path
import csv
import cv2
import torch
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "yolo_dataset"
MANIFEST_FILE = DATASET_DIR / "phase8_species_manifest.csv"

LABEL_DIR = DATASET_DIR / "labels"
PREVIEW_DIR = DATASET_DIR / "auto_annotation_previews"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "yolov8s-world.pt"

CONFIDENCE = 0.20
IOU = 0.50
IMAGE_SIZE = 640

SAVE_PREVIEWS = True
PREVIEW_LIMIT = 20

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def load_manifest():
    rows = []

    with open(MANIFEST_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def save_manifest(rows):
    temp = MANIFEST_FILE.with_suffix(".tmp")

    fields = [
        "split",
        "species_name",
        "original_image",
        "destination",
        "label_file",
        "annotation_status"
    ]

    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(MANIFEST_FILE)


def write_label(label_path, boxes):
    temp = label_path.with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        for box in boxes:
            x1, y1, x2, y2, confidence = box

            width = x2 - x1
            height = y2 - y1

            center_x = x1 + width / 2
            center_y = y1 + height / 2

            f.write(
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

    temp.replace(label_path)


def create_preview(
    image,
    boxes,
    scientific_name,
    common_name,
    output_path
):

    preview = image.copy()

    for x1, y1, x2, y2, confidence in boxes:

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f"{scientific_name} | {common_name} | {confidence:.2f}"

        text_y = max(30, y1 - 10)

        cv2.putText(
            preview,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = output_path.with_suffix(".tmp.jpg")

    cv2.imwrite(
        str(temp),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    temp.replace(output_path)


def extract_names(species_name):

    scientific_name = species_name.replace("_", " ")

    common_name = ""

    return scientific_name, common_name


if not MANIFEST_FILE.exists():

    print("ERROR: Phase 8A manifest not found.")
    print(MANIFEST_FILE)
    raise SystemExit(1)


rows = load_manifest()

pending = []

for row in rows:

    if row["annotation_status"] == "completed":
        continue

    if row["annotation_status"] == "no_detection":
        continue

    if row["annotation_status"] == "low_confidence":
        continue

    pending.append(row)


print()
print("=" * 70)
print("VENOMGUARD PHASE 8B")
print("AUTOMATIC YOLO-WORLD SNAKE ANNOTATION")
print("=" * 70)

print(f"Total images      : {len(rows)}")
print(f"Already completed : {len(rows) - len(pending)}")
print(f"Remaining         : {len(pending)}")
print(f"Device            : {DEVICE}")
print(f"Confidence        : {CONFIDENCE}")
print("=" * 70)


if not MODEL_FILE.exists():

    print()
    print("YOLO-World model not found.")
    print("Downloading yolov8s-world.pt...")
    print()

    model = YOLOWorld("yolov8s-world.pt")

    model.save(str(MODEL_FILE))

else:

    print()
    print("Loading existing YOLO-World model:")
    print(MODEL_FILE)

    model = YOLOWorld(str(MODEL_FILE))


model.set_classes(["snake"])

print()
print("YOLO-World prompt: snake")
print("Model ready.")
print()


completed = 0
no_detection = 0
low_confidence = 0
failed = 0

preview_count = 0

review_file = DATASET_DIR / "phase8_auto_annotation_review.csv"

review_exists = review_file.exists()

review_temp = review_file.with_suffix(".tmp")

review_fields = [
    "split",
    "species_name",
    "image",
    "status",
    "detection_count",
    "max_confidence"
]

if review_exists:

    with open(
        review_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        existing_reviews = list(
            csv.DictReader(f)
        )

else:

    existing_reviews = []


review_rows = existing_reviews.copy()


for index, row in enumerate(pending, 1):

    image_path = PROJECT_DIR / row["destination"]
    label_path = PROJECT_DIR / row["label_file"]

    species_name = row["species_name"]

    scientific_name, common_name = extract_names(
        species_name
    )

    print(
        f"[{index}/{len(pending)}] "
        f"{species_name}"
    )

    if not image_path.exists():

        print("  IMAGE MISSING")

        failed += 1

        review_rows.append({
            "split": row["split"],
            "species_name": species_name,
            "image": row["destination"],
            "status": "missing_image",
            "detection_count": 0,
            "max_confidence": 0
        })

        continue


    if label_path.exists() and label_path.stat().st_size > 0:

        row["annotation_status"] = "completed"

        completed += 1

        print("  Existing label found - skipped")

        save_manifest(rows)

        continue


    image = cv2.imread(str(image_path))

    if image is None:

        print("  IMAGE READ FAILED")

        failed += 1

        review_rows.append({
            "split": row["split"],
            "species_name": species_name,
            "image": row["destination"],
            "status": "image_read_failed",
            "detection_count": 0,
            "max_confidence": 0
        })

        continue


    try:

        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

    except Exception as e:

        print(f"  PREDICTION FAILED: {e}")

        failed += 1

        review_rows.append({
            "split": row["split"],
            "species_name": species_name,
            "image": row["destination"],
            "status": "prediction_failed",
            "detection_count": 0,
            "max_confidence": 0
        })

        continue


    result = results[0]

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("  NO SNAKE DETECTED")

        row["annotation_status"] = "no_detection"

        no_detection += 1

        review_rows.append({
            "split": row["split"],
            "species_name": species_name,
            "image": row["destination"],
            "status": "no_detection",
            "detection_count": 0,
            "max_confidence": 0
        })

        save_manifest(rows)

        continue


    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()


    valid_boxes = []

    for box, confidence in zip(xyxy, confs):

        x1, y1, x2, y2 = box

        valid_boxes.append(
            (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(confidence)
            )
        )


    valid_boxes.sort(
        key=lambda x: x[4],
        reverse=True
    )


    max_confidence = valid_boxes[0][4]


    if max_confidence < 0.40:

        print(
            f"  LOW CONFIDENCE: "
            f"{max_confidence:.3f}"
        )

        row["annotation_status"] = "low_confidence"

        low_confidence += 1

        review_rows.append({
            "split": row["split"],
            "species_name": species_name,
            "image": row["destination"],
            "status": "low_confidence",
            "detection_count": len(valid_boxes),
            "max_confidence": f"{max_confidence:.6f}"
        })

        save_manifest(rows)

        continue


    write_label(
        label_path,
        valid_boxes
    )


    if SAVE_PREVIEWS and preview_count < PREVIEW_LIMIT:

        preview_path = (
            PREVIEW_DIR /
            row["split"] /
            image_path.name
        )

        create_preview(
            image,
            valid_boxes,
            scientific_name,
            common_name,
            preview_path
        )

        preview_count += 1


    row["annotation_status"] = "completed"

    completed += 1

    review_rows.append({
        "split": row["split"],
        "species_name": species_name,
        "image": row["destination"],
        "status": "completed",
        "detection_count": len(valid_boxes),
        "max_confidence": f"{max_confidence:.6f}"
    })


    save_manifest(rows)


    with open(
        review_temp,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=review_fields
        )

        writer.writeheader()
        writer.writerows(review_rows)


    review_temp.replace(review_file)


    print(
        f"  DETECTED: "
        f"{len(valid_boxes)} box(es) "
        f"max_conf={max_confidence:.3f}"
    )


print()
print("=" * 70)
print("PHASE 8B SUMMARY")
print("=" * 70)

print(f"Total images       : {len(rows)}")
print(f"Completed          : {completed}")
print(f"No detection      : {no_detection}")
print(f"Low confidence    : {low_confidence}")
print(f"Failed            : {failed}")

remaining = sum(
    1
    for row in rows
    if row["annotation_status"] == "pending"
)

print(f"Still pending      : {remaining}")

print()
print(f"Review report      : {review_file}")
print(f"Preview directory  : {PREVIEW_DIR}")
print(f"Label directory    : {LABEL_DIR}")

print("=" * 70)

if remaining == 0:

    print("ALL IMAGES HAVE BEEN PROCESSED.")

else:

    print("Some images remain.")
    print("Run the same command again to continue.")

print()
print("PHASE 8B FINISHED SUCCESSFULLY")