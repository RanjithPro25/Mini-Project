from pathlib import Path
import csv
import shutil
import cv2

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

YOLO_DATASET = PROJECT_DIR / "yolo_dataset"

IMAGE_ROOT = YOLO_DATASET / "images"
LABEL_ROOT = YOLO_DATASET / "labels"

PHASE8_MANIFEST = (
    PROJECT_DIR
    / "yolo_dataset"
    / "phase8_species_manifest.csv"
)

CLEAN_ROOT = PROJECT_DIR / "yolo_dataset_clean"

CLEAN_IMAGES = CLEAN_ROOT / "images"
CLEAN_LABELS = CLEAN_ROOT / "labels"

REPORT_DIR = PROJECT_DIR / "phase8"

VALID_REPORT = REPORT_DIR / "phase8_final_valid.csv"
INVALID_REPORT = REPORT_DIR / "phase8_final_invalid.csv"
SUMMARY_REPORT = REPORT_DIR / "phase8_final_summary.csv"

EXPECTED_TOTAL = 4254

EXPECTED_SPLITS = {
    "train": 2710,
    "val": 738,
    "test": 806
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

MIN_AREA = 0.0001
MAX_AREA = 0.98


def find_column(columns, candidates):

    lower = {
        c.lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def read_manifest():

    if not PHASE8_MANIFEST.exists():

        raise RuntimeError(
            "Original Phase 8A manifest was not found:\n"
            f"{PHASE8_MANIFEST}"
        )

    with open(
        PHASE8_MANIFEST,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} Phase 8A records, "
            f"found {len(rows)}"
        )

    return rows


def resolve_image(row, split, species):

    candidates = []

    for key in [
        "destination",
        "image",
        "image_path",
        "path",
        "source"
    ]:

        value = row.get(key)

        if value:
            candidates.append(Path(value))

    for candidate in candidates:

        if candidate.exists():
            return candidate

    filename = None

    for candidate in candidates:

        if candidate.name:
            filename = candidate.name
            break

    if filename:

        direct = (
            IMAGE_ROOT
            / split
            / species
            / filename
        )

        if direct.exists():
            return direct

        matches = list(
            (IMAGE_ROOT / split).rglob(filename)
        )

        if matches:
            return matches[0]

    return None


def label_path(image, split, species):

    return (
        LABEL_ROOT
        / split
        / species
        / f"{image.stem}.txt"
    )


def validate_and_repair_label(
    label,
    width,
    height
):

    try:

        text = label.read_text(
            encoding="utf-8"
        ).strip()

    except Exception as e:

        return False, None, f"read_error: {e}"

    if not text:

        return False, None, "empty_label"

    output_lines = []
    repaired = False

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        parts = line.split()

        if len(parts) != 5:

            return (
                False,
                None,
                f"line {line_number}: expected 5 values"
            )

        try:

            class_id = int(float(parts[0]))

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except Exception:

            return (
                False,
                None,
                f"line {line_number}: non-numeric"
            )

        if class_id != 0:

            return (
                False,
                None,
                f"line {line_number}: class {class_id}"
            )

        if bw <= 0 or bh <= 0:

            return (
                False,
                None,
                f"line {line_number}: invalid size"
            )

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        original = (
            x1,
            y1,
            x2,
            y2
        )

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        if original != (
            x1,
            y1,
            x2,
            y2
        ):
            repaired = True

        if x2 <= x1 or y2 <= y1:

            return (
                False,
                None,
                f"line {line_number}: "
                f"box invalid after clipping"
            )

        new_xc = (x1 + x2) / 2
        new_yc = (y1 + y2) / 2
        new_bw = x2 - x1
        new_bh = y2 - y1

        area = new_bw * new_bh

        if area < MIN_AREA:

            return (
                False,
                None,
                f"line {line_number}: box too small"
            )

        if area > MAX_AREA:

            return (
                False,
                None,
                f"line {line_number}: box too large"
            )

        output_lines.append(
            f"0 "
            f"{new_xc:.6f} "
            f"{new_yc:.6f} "
            f"{new_bw:.6f} "
            f"{new_bh:.6f}"
        )

    if not output_lines:

        return False, None, "no_boxes"

    return (
        True,
        "\n".join(output_lines) + "\n",
        "repaired" if repaired else "valid"
    )


print("=" * 75)
print("PHASE 8 - FINAL DETECTION DATASET COMPLETION")
print("=" * 75)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("STEP 1: Reading original Phase 8A manifest")

rows = read_manifest()

print(
    f"Phase 8A records: {len(rows)}"
)

print("Expected: 4,254")

print()
print("STEP 2: Checking Phase 8 split counts")

split_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

for row in rows:

    split = (
        row.get("split")
        or row.get("set")
        or row.get("subset")
    )

    if split not in split_counts:

        raise RuntimeError(
            f"Unknown split in manifest: {split}"
        )

    split_counts[split] += 1

print(
    "Train:",
    split_counts["train"]
)

print(
    "Val:",
    split_counts["val"]
)

print(
    "Test:",
    split_counts["test"]
)

for split in EXPECTED_SPLITS:

    if split_counts[split] != EXPECTED_SPLITS[split]:

        raise RuntimeError(
            f"{split} expected "
            f"{EXPECTED_SPLITS[split]}, "
            f"found {split_counts[split]}"
        )

print("Split counts VERIFIED")

print()
print("STEP 3: Creating clean YOLO dataset")

for split in [
    "train",
    "val",
    "test"
]:

    (CLEAN_IMAGES / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (CLEAN_LABELS / split).mkdir(
        parents=True,
        exist_ok=True
    )

print("Clean dataset folders ready")

print()
print("STEP 4: Validating exactly 4,254 records")

valid_rows = []
invalid_rows = []

valid_count = 0
invalid_count = 0
repaired_count = 0
missing_image_count = 0
missing_label_count = 0
read_error_count = 0

processed = 0

for row in rows:

    processed += 1

    split = (
        row.get("split")
        or row.get("set")
        or row.get("subset")
    )

    species = (
        row.get("species")
        or row.get("scientific_name")
        or row.get("class")
        or ""
    )

    if not species:

        invalid_count += 1

        invalid_rows.append({
            "split": split,
            "species": "",
            "file": "",
            "status": "invalid",
            "reason": "missing_species"
        })

        continue

    image = resolve_image(
        row,
        split,
        species
    )

    if image is None:

        missing_image_count += 1
        invalid_count += 1

        invalid_rows.append({
            "split": split,
            "species": species,
            "file": "",
            "status": "missing_image",
            "reason": "Phase 8 image not found"
        })

        continue

    label = label_path(
        image,
        split,
        species
    )

    if not label.exists():

        missing_label_count += 1
        invalid_count += 1

        invalid_rows.append({
            "split": split,
            "species": species,
            "file": image.name,
            "status": "missing_label",
            "reason": str(label)
        })

        continue

    image_data = cv2.imread(
        str(image),
        cv2.IMREAD_COLOR
    )

    if image_data is None:

        read_error_count += 1
        invalid_count += 1

        invalid_rows.append({
            "split": split,
            "species": species,
            "file": image.name,
            "status": "image_read_error",
            "reason": "OpenCV could not read image"
        })

        continue

    height, width = image_data.shape[:2]

    ok, clean_label, status = (
        validate_and_repair_label(
            label,
            width,
            height
        )
    )

    if not ok:

        invalid_count += 1

        invalid_rows.append({
            "split": split,
            "species": species,
            "file": image.name,
            "status": "invalid_label",
            "reason": status
        })

        continue

    destination_image = (
        CLEAN_IMAGES
        / split
        / species
        / image.name
    )

    destination_label = (
        CLEAN_LABELS
        / split
        / species
        / f"{image.stem}.txt"
    )

    destination_image.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_label.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not destination_image.exists():

        shutil.copy2(
            image,
            destination_image
        )

    if not destination_label.exists():

        destination_label.write_text(
            clean_label,
            encoding="utf-8"
        )

    if status == "repaired":

        repaired_count += 1

    valid_count += 1

    valid_rows.append({
        "split": split,
        "species": species,
        "file": image.name,
        "image": str(destination_image),
        "label": str(destination_label),
        "status": status
    })

    if processed % 250 == 0:

        print(
            f"[{processed:04d}/{EXPECTED_TOTAL}] "
            f"valid={valid_count} "
            f"invalid={invalid_count} "
            f"repaired={repaired_count}"
        )

print()
print("STEP 5: Saving reports")

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
            "species",
            "file",
            "image",
            "label",
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
            "species",
            "file",
            "status",
            "reason"
        ]
    )

    writer.writeheader()
    writer.writerows(invalid_rows)

summary = [
    ["Metric", "Value"],
    ["Expected Phase 8 records", EXPECTED_TOTAL],
    ["Phase 8 train", split_counts["train"]],
    ["Phase 8 val", split_counts["val"]],
    ["Phase 8 test", split_counts["test"]],
    ["Processed", processed],
    ["Valid", valid_count],
    ["Repaired", repaired_count],
    ["Invalid", invalid_count],
    ["Missing images", missing_image_count],
    ["Missing labels", missing_label_count],
    ["Image read errors", read_error_count]
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
print("PHASE 8 FINAL RESULT")
print("=" * 75)

print(
    "Expected records :",
    EXPECTED_TOTAL
)

print(
    "Processed        :",
    processed
)

print(
    "Valid            :",
    valid_count
)

print(
    "Repaired         :",
    repaired_count
)

print(
    "Invalid          :",
    invalid_count
)

print(
    "Missing images   :",
    missing_image_count
)

print(
    "Missing labels   :",
    missing_label_count
)

print(
    "Read errors      :",
    read_error_count
)

print()
print("Clean dataset:")
print(CLEAN_ROOT)

print()
print("Valid report:")
print(VALID_REPORT)

print()
print("Invalid report:")
print(INVALID_REPORT)

print()
print("Summary:")
print(SUMMARY_REPORT)

print()

if (
    processed == EXPECTED_TOTAL
    and valid_count == EXPECTED_TOTAL
    and invalid_count == 0
    and missing_image_count == 0
    and missing_label_count == 0
    and read_error_count == 0
):

    print("=" * 75)
    print("PHASE 8 FINISHED SUCCESSFULLY")
    print("=" * 75)

else:

    print("=" * 75)
    print("PHASE 8 NOT YET COMPLETE")
    print("=" * 75)

    print()
    print(
        "The invalid records were NOT deleted."
    )

    print(
        "Review:",
        INVALID_REPORT
    )