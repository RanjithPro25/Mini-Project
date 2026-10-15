from pathlib import Path
import csv
import hashlib
import cv2

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

YOLO_DATASET = PROJECT_DIR / "yolo_dataset"

IMAGE_DIR = YOLO_DATASET / "images"
LABEL_DIR = YOLO_DATASET / "labels"

REPORT_DIR = PROJECT_DIR / "phase8"

VALID_REPORT = REPORT_DIR / "phase8_valid_annotations.csv"
INVALID_REPORT = REPORT_DIR / "phase8_invalid_annotations.csv"
SUMMARY_REPORT = REPORT_DIR / "phase8_validation_summary.csv"

SPLITS = ["train", "val", "test"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

EXPECTED_TOTAL = 4254

MIN_BOX_AREA = 0.0001
MAX_BOX_AREA = 0.98

print("=" * 75)
print("PHASE 8 - YOLO ANNOTATION VALIDATION")
print("=" * 75)

print()
print("STEP 1: Checking dataset folders")

if not IMAGE_DIR.exists():
    raise RuntimeError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not LABEL_DIR.exists():
    raise RuntimeError(
        f"Label directory not found:\n{LABEL_DIR}"
    )

for split in SPLITS:

    image_split = IMAGE_DIR / split
    label_split = LABEL_DIR / split

    if not image_split.exists():
        raise RuntimeError(
            f"Missing image split:\n{image_split}"
        )

    if not label_split.exists():
        raise RuntimeError(
            f"Missing label split:\n{label_split}"
        )

print("Dataset folders verified")

print()
print("STEP 2: Collecting image files")

images = {}

for split in SPLITS:

    images[split] = []

    for path in (
        IMAGE_DIR / split
    ).rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        ):

            images[split].append(path)

    images[split].sort()

print(
    "Train images      :",
    len(images["train"])
)

print(
    "Validation images :",
    len(images["val"])
)

print(
    "Test images       :",
    len(images["test"])
)

total_images = sum(
    len(images[split])
    for split in SPLITS
)

print(
    "Total images      :",
    total_images
)

print()
print("STEP 3: Collecting label files")

labels = {}

for split in SPLITS:

    labels[split] = []

    for path in (
        LABEL_DIR / split
    ).rglob("*.txt"):

        if path.is_file():
            labels[split].append(path)

    labels[split].sort()

print(
    "Train labels      :",
    len(labels["train"])
)

print(
    "Validation labels :",
    len(labels["val"])
)

print(
    "Test labels       :",
    len(labels["test"])
)

total_labels = sum(
    len(labels[split])
    for split in SPLITS
)

print(
    "Total labels      :",
    total_labels
)

print()
print("STEP 4: Building image-label maps")

image_maps = {}
label_maps = {}

for split in SPLITS:

    image_maps[split] = {}
    label_maps[split] = {}

    for image in images[split]:

        key = str(
            image.relative_to(
                IMAGE_DIR / split
            ).with_suffix("")
        ).lower()

        if key in image_maps[split]:

            print(
                "DUPLICATE IMAGE:",
                split,
                key
            )

        image_maps[split][key] = image

    for label in labels[split]:

        key = str(
            label.relative_to(
                LABEL_DIR / split
            ).with_suffix("")
        ).lower()

        if key in label_maps[split]:

            print(
                "DUPLICATE LABEL:",
                split,
                key
            )

        label_maps[split][key] = label

print("Maps created")

print()
print("STEP 5: Validating annotations")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

valid_rows = []
invalid_rows = []

valid_count = 0
invalid_count = 0
missing_label_count = 0
missing_image_count = 0
empty_label_count = 0
invalid_coordinate_count = 0
invalid_class_count = 0
invalid_box_count = 0
image_read_error_count = 0

checked = 0

for split in SPLITS:

    image_map = image_maps[split]
    label_map = label_maps[split]

    all_keys = sorted(
        set(image_map.keys())
        | set(label_map.keys())
    )

    for key in all_keys:

        checked += 1

        image = image_map.get(key)
        label = label_map.get(key)

        relative_name = key

        if image is None:

            missing_image_count += 1
            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "missing_image",
                "reason": "Label exists but image is missing"
            })

            continue

        if label is None:

            missing_label_count += 1
            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "missing_label",
                "reason": "Image exists but label is missing"
            })

            continue

        image_data = cv2.imread(
            str(image),
            cv2.IMREAD_COLOR
        )

        if image_data is None:

            image_read_error_count += 1
            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "image_read_error",
                "reason": "OpenCV could not read image"
            })

            continue

        height, width = image_data.shape[:2]

        try:

            text = label.read_text(
                encoding="utf-8"
            ).strip()

        except Exception as e:

            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "label_read_error",
                "reason": str(e)
            })

            continue

        if not text:

            empty_label_count += 1
            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "empty_label",
                "reason": "Label file is empty"
            })

            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        file_valid = True
        reasons = []

        boxes = []

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()

            if len(parts) != 5:

                file_valid = False
                reasons.append(
                    f"line {line_number}: expected 5 values"
                )

                continue

            try:

                class_id = int(float(parts[0]))

                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

            except Exception:

                file_valid = False

                reasons.append(
                    f"line {line_number}: non-numeric values"
                )

                continue

            if class_id != 0:

                invalid_class_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: class={class_id}, expected 0"
                )

            values = [
                xc,
                yc,
                bw,
                bh
            ]

            if any(
                value < 0 or value > 1
                for value in values
            ):

                invalid_coordinate_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"coordinates outside 0-1"
                )

                continue

            if bw <= 0 or bh <= 0:

                invalid_box_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"width/height <= 0"
                )

                continue

            area = bw * bh

            if area < MIN_BOX_AREA:

                invalid_box_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"box too small"
                )

            if area > MAX_BOX_AREA:

                invalid_box_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"box too large"
                )

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            if (
                x1 < 0
                or y1 < 0
                or x2 > 1
                or y2 > 1
            ):

                invalid_coordinate_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"box extends outside image"
                )

            pixel_x1 = int(
                max(0, x1 * width)
            )

            pixel_y1 = int(
                max(0, y1 * height)
            )

            pixel_x2 = int(
                min(width, x2 * width)
            )

            pixel_y2 = int(
                min(height, y2 * height)
            )

            if pixel_x2 <= pixel_x1:
                invalid_box_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"invalid pixel width"
                )

            if pixel_y2 <= pixel_y1:
                invalid_box_count += 1
                file_valid = False

                reasons.append(
                    f"line {line_number}: "
                    f"invalid pixel height"
                )

            boxes.append(
                (
                    xc,
                    yc,
                    bw,
                    bh
                )
            )

        if not boxes:

            file_valid = False

            if "empty label" not in reasons:
                reasons.append(
                    "No valid bounding boxes"
                )

        if file_valid:

            valid_count += 1

            valid_rows.append({
                "split": split,
                "file": relative_name,
                "image": str(image),
                "label": str(label),
                "width": width,
                "height": height,
                "boxes": len(boxes),
                "status": "valid"
            })

        else:

            invalid_count += 1

            invalid_rows.append({
                "split": split,
                "file": relative_name,
                "status": "invalid",
                "reason": " | ".join(reasons)
            })

        if checked % 250 == 0:

            print(
                f"Checked {checked}/{total_images} | "
                f"valid={valid_count} | "
                f"invalid={invalid_count}"
            )

print()
print("STEP 6: Saving validation reports")

with open(
    VALID_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "file",
            "image",
            "label",
            "width",
            "height",
            "boxes",
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(valid_rows)

with open(
    INVALID_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "file",
            "status",
            "reason"
        ]
    )

    writer.writeheader()
    writer.writerows(invalid_rows)

summary = [
    ["Metric", "Value"],
    ["Expected annotated images", EXPECTED_TOTAL],
    ["Images found", total_images],
    ["Labels found", total_labels],
    ["Checked", checked],
    ["Valid annotations", valid_count],
    ["Invalid annotations", invalid_count],
    ["Missing labels", missing_label_count],
    ["Missing images", missing_image_count],
    ["Empty labels", empty_label_count],
    ["Invalid classes", invalid_class_count],
    ["Invalid coordinates", invalid_coordinate_count],
    ["Invalid boxes", invalid_box_count],
    ["Image read errors", image_read_error_count]
]

with open(
    SUMMARY_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerows(summary)

print()
print("=" * 75)
print("PHASE 8 VALIDATION RESULT")
print("=" * 75)

print(
    f"Expected annotated images : {EXPECTED_TOTAL}"
)

print(
    f"Images found              : {total_images}"
)

print(
    f"Labels found              : {total_labels}"
)

print(
    f"Checked                   : {checked}"
)

print(
    f"Valid annotations         : {valid_count}"
)

print(
    f"Invalid annotations       : {invalid_count}"
)

print(
    f"Missing labels            : {missing_label_count}"
)

print(
    f"Missing images            : {missing_image_count}"
)

print(
    f"Empty labels              : {empty_label_count}"
)

print(
    f"Invalid classes           : {invalid_class_count}"
)

print(
    f"Invalid coordinates       : {invalid_coordinate_count}"
)

print(
    f"Invalid boxes             : {invalid_box_count}"
)

print(
    f"Image read errors         : {image_read_error_count}"
)

print()
print("Valid report:")
print(VALID_REPORT)

print()
print("Invalid report:")
print(INVALID_REPORT)

print()
print("Summary report:")
print(SUMMARY_REPORT)

print()

if (
    total_images == EXPECTED_TOTAL
    and total_labels == EXPECTED_TOTAL
    and checked == EXPECTED_TOTAL
    and invalid_count == 0
):

    print("=" * 75)
    print("PHASE 8 FINISHED SUCCESSFULLY")
    print("=" * 75)

else:

    print("=" * 75)
    print("PHASE 8 VALIDATION FOUND ISSUES")
    print("=" * 75)

    print()
    print(
        "Do NOT start YOLOv8 training yet."
    )

    print(
        "Review:"
    )

    print(
        INVALID_REPORT
    )