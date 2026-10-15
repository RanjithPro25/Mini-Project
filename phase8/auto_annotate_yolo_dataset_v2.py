from pathlib import Path
import csv
import cv2
import torch
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "yolo_dataset"
LABEL_DIR = DATASET_DIR / "labels"

MANIFEST_FILE = DATASET_DIR / "phase8_species_manifest.csv"
SOURCE_CSV = PROJECT_DIR / "data" / "text based new.csv"
MODEL_PATH = PROJECT_DIR / "models" / "yolov8s-world.pt"

REVIEW_DIR = DATASET_DIR / "auto_annotation_previews"
REVIEW_FILE = DATASET_DIR / "phase8_auto_annotation_review.csv"

IMAGE_SIZE = 640
CONFIDENCE = 0.01
IOU = 0.50

ACCEPT_THRESHOLD = 0.20
REVIEW_THRESHOLD = 0.05

PROMPTS = [
    "snake animal",
    "snake",
    "reptile",
    "a snake"
]

MAX_PREVIEWS = 100

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"

print("Loading source CSV...")

source_rows = None

for encoding in ["cp1252", "latin1", "utf-8"]:
    try:
        with open(
            SOURCE_CSV,
            "r",
            encoding=encoding,
            newline=""
        ) as f:
            source_rows = list(csv.DictReader(f))
        print("Source CSV encoding:", encoding)
        break
    except Exception:
        pass

if source_rows is None:
    raise RuntimeError("Could not read source CSV")


def clean(value):
    if not value:
        return ""
    return " ".join(
        str(value).replace("_", " ").strip().split()
    )


species_map = {}

for row in source_rows:

    scientific = clean(
        row.get("scientific_name", "")
    )

    common = clean(
        row.get("common_name", "")
    )

    taxon_species = clean(
        row.get("taxon_species_name", "")
    )

    for key in [
        scientific.lower(),
        taxon_species.lower()
    ]:

        if not key:
            continue

        if key not in species_map:

            species_map[key] = {
                "scientific": scientific,
                "common": common
            }

        elif not species_map[key]["common"] and common:

            species_map[key]["common"] = common


def get_names(species):

    species = clean(species)

    key = species.lower()

    if key in species_map:
        return species_map[key]

    parts = species.split()

    if len(parts) >= 2:

        key = " ".join(parts[:2]).lower()

        if key in species_map:
            return species_map[key]

    return {
        "scientific": species,
        "common": ""
    }


print("Species mappings:", len(species_map))
print()
print("Loading YOLO-World...")
print("Device:", device)

model = YOLOWorld(str(MODEL_PATH))

model.set_classes(PROMPTS)

print("YOLO prompts:", ", ".join(PROMPTS))


with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = list(csv.DictReader(f))


fields = list(rows[0].keys())

for field in [
    "annotation_status",
    "best_confidence",
    "detection_count",
    "annotation_note",
    "scientific_name",
    "common_name"
]:

    if field not in fields:
        fields.append(field)


def save_manifest():

    temp = MANIFEST_FILE.with_suffix(".tmp")

    with open(
        temp,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    temp.replace(MANIFEST_FILE)


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    intersection = w * h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / union


def nms(boxes):

    boxes = sorted(
        boxes,
        key=lambda x: x[4],
        reverse=True
    )

    selected = []

    while boxes:

        best = boxes.pop(0)

        selected.append(best)

        remaining = []

        for box in boxes:

            if box_iou(
                best[:4],
                box[:4]
            ) < IOU:

                remaining.append(box)

        boxes = remaining

    return selected


def normalize(box, width, height):

    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height

    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    return xc, yc, bw, bh


def draw_preview(
    image,
    boxes,
    scientific,
    common,
    status,
    output
):

    result = image.copy()

    for box in boxes:

        x1, y1, x2, y2, confidence = box

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            scientific,
            (x1, max(25, y1 - 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        if common:

            cv2.putText(
                result,
                common,
                (x1, max(45, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    cv2.putText(
        result,
        status.upper(),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imwrite(
        str(output),
        result
    )


review_rows = []

if REVIEW_FILE.exists():

    with open(
        REVIEW_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        review_rows = list(
            csv.DictReader(f)
        )


review_fields = [
    "split",
    "species_name",
    "scientific_name",
    "common_name",
    "image",
    "status",
    "best_confidence",
    "detection_count",
    "note"
]


def save_reviews():

    temp = REVIEW_FILE.with_suffix(".tmp")

    with open(
        temp,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=review_fields
        )

        writer.writeheader()
        writer.writerows(review_rows)

    temp.replace(REVIEW_FILE)


total = len(rows)

accepted = 0
review = 0
no_detection = 0
failed = 0
skipped = 0

preview_count = 0

print()
print("Total images:", total)
print("Starting Phase 8B...")
print()


for index, row in enumerate(rows, 1):

    status = row.get(
        "annotation_status",
        ""
    ).strip().lower()

    if status in [
        "completed",
        "review",
        "no_detection"
    ]:

        skipped += 1
        continue

    split = row["split"]
    species = row["species_name"]

    names = get_names(species)

    scientific = names["scientific"]
    common = names["common"]

    row["scientific_name"] = scientific
    row["common_name"] = common

    image_path = Path(
        row["destination"]
    )

    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    label_path = Path(
        row["label_file"]
    )

    if not label_path.is_absolute():
        label_path = PROJECT_DIR / label_path

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        row["annotation_status"] = "failed"
        row["best_confidence"] = "0"
        row["detection_count"] = "0"
        row["annotation_note"] = "image_read_failed"

        failed += 1

        save_manifest()

        continue

    height, width = image.shape[:2]

    print(
        f"[{index}/{total}] "
        f"{scientific}"
        + (
            f" | {common}"
            if common
            else ""
        )
    )

    try:

        results = model.predict(
            source=image,
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )

        boxes = []

        if results:

            result = results[0]

            if result.boxes is not None:

                xyxy = (
                    result.boxes
                    .xyxy
                    .cpu()
                    .tolist()
                )

                confs = (
                    result.boxes
                    .conf
                    .cpu()
                    .tolist()
                )

                classes = (
                    result.boxes
                    .cls
                    .cpu()
                    .tolist()
                )

                for box, confidence, cls in zip(
                    xyxy,
                    confs,
                    classes
                ):

                    if confidence < CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = box

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence
                    ])

        boxes = nms(boxes)

        if not boxes:

            row["annotation_status"] = (
                "no_detection"
            )

            row["best_confidence"] = "0"
            row["detection_count"] = "0"

            row["annotation_note"] = (
                "no_candidate_box"
            )

            no_detection += 1

            if preview_count < MAX_PREVIEWS:

                preview = (
                    REVIEW_DIR /
                    f"{index}_no_detection.jpg"
                )

                draw_preview(
                    image,
                    [],
                    scientific,
                    common,
                    "no detection",
                    preview
                )

                preview_count += 1

            review_rows.append({
                "split": split,
                "species_name": species,
                "scientific_name": scientific,
                "common_name": common,
                "image": str(image_path),
                "status": "no_detection",
                "best_confidence": "0",
                "detection_count": "0",
                "note": "No candidate box"
            })

            save_reviews()
            save_manifest()

            continue

        best_confidence = max(
            box[4]
            for box in boxes
        )

        valid_boxes = []

        for box in boxes:

            normalized = normalize(
                box[:4],
                width,
                height
            )

            if normalized is not None:

                valid_boxes.append(
                    (box, normalized)
                )

        if not valid_boxes:

            row["annotation_status"] = (
                "failed"
            )

            row["best_confidence"] = (
                f"{best_confidence:.4f}"
            )

            row["detection_count"] = "0"
            row["annotation_note"] = (
                "invalid_box"
            )

            failed += 1

            save_manifest()

            continue

        label_lines = []

        for box, normalized in valid_boxes:

            xc, yc, bw, bh = normalized

            label_lines.append(
                f"0 {xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_label = label_path.with_suffix(
            ".tmp"
        )

        with open(
            temp_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(label_lines)
            )

        temp_label.replace(
            label_path
        )

        if best_confidence >= ACCEPT_THRESHOLD:

            annotation_status = "completed"
            annotation_note = "accepted"

            accepted += 1

        elif best_confidence >= REVIEW_THRESHOLD:

            annotation_status = "review"
            annotation_note = (
                "low_confidence_requires_review"
            )

            review += 1

        else:

            annotation_status = "no_detection"
            annotation_note = (
                "below_review_threshold"
            )

            no_detection += 1

        row["annotation_status"] = (
            annotation_status
        )

        row["best_confidence"] = (
            f"{best_confidence:.4f}"
        )

        row["detection_count"] = str(
            len(valid_boxes)
        )

        row["annotation_note"] = (
            annotation_note
        )

        if annotation_status != "completed":

            review_rows.append({
                "split": split,
                "species_name": species,
                "scientific_name": scientific,
                "common_name": common,
                "image": str(image_path),
                "status": annotation_status,
                "best_confidence": (
                    f"{best_confidence:.4f}"
                ),
                "detection_count": str(
                    len(valid_boxes)
                ),
                "note": annotation_note
            })

            save_reviews()

        if (
            preview_count < MAX_PREVIEWS
            or annotation_status != "completed"
        ):

            preview = (
                REVIEW_DIR /
                f"{index}_{annotation_status}.jpg"
            )

            draw_preview(
                image,
                boxes,
                scientific,
                common,
                annotation_status,
                preview
            )

            preview_count += 1

        save_manifest()

    except KeyboardInterrupt:

        save_manifest()
        save_reviews()

        print()
        print("INTERRUPTED SAFELY")
        print("Progress has been saved.")
        print("Run the same command again to resume.")

        raise

    except Exception as e:

        row["annotation_status"] = "failed"
        row["best_confidence"] = "0"
        row["detection_count"] = "0"
        row["annotation_note"] = str(e)

        failed += 1

        save_manifest()

    if index % 10 == 0:

        print(
            f"Progress: {index}/{total} | "
            f"Accepted: {accepted} | "
            f"Review: {review} | "
            f"No detection: {no_detection} | "
            f"Failed: {failed} | "
            f"Skipped: {skipped}"
        )


save_manifest()
save_reviews()

print()
print("=" * 65)
print("PHASE 8B SUMMARY")
print("=" * 65)
print("Total images      :", total)
print("Skipped           :", skipped)
print("Accepted          :", accepted)
print("Review required   :", review)
print("No detection      :", no_detection)
print("Failed            :", failed)
print()
print("Labels            :", LABEL_DIR)
print("Previews          :", REVIEW_DIR)
print("Review CSV        :", REVIEW_FILE)
print("Manifest          :", MANIFEST_FILE)
print()
print("PHASE 8B FINISHED SUCCESSFULLY")