from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

ROOT = PROJECT_DIR / "yolo_dataset_final"
IMAGE_ROOT = ROOT / "images"
LABEL_ROOT = ROOT / "labels"

REPORT = PROJECT_DIR / "phase8" / "phase8_final_dataset_validation.csv"

SPLITS = ["train", "val", "test"]

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

valid = 0
invalid = 0
missing_label = 0
missing_image = 0

rows = []

print("=" * 70)
print("PHASE 8 - FINAL YOLO DATASET VALIDATION")
print("=" * 70)

for split in SPLITS:

    image_dir = IMAGE_ROOT / split
    label_dir = LABEL_ROOT / split

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    labels = list(label_dir.rglob("*.txt"))

    print()
    print(split)
    print("Images:", len(images))
    print("Labels:", len(labels))

    label_map = {
        p.stem.lower(): p
        for p in labels
    }

    for image in images:

        label = label_map.get(
            image.stem.lower()
        )

        if label is None:

            missing_label += 1
            rows.append({
                "split": split,
                "file": image.name,
                "status": "missing_label"
            })

            continue

        try:

            text = label.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                raise ValueError("empty label")

            for line in text.splitlines():

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        "wrong number of values"
                    )

                cls = int(float(parts[0]))

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                if cls != 0:
                    raise ValueError(
                        "class must be 0"
                    )

                if not (
                    0 <= xc <= 1
                    and 0 <= yc <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    raise ValueError(
                        "invalid coordinates"
                    )

                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2

                if (
                    x1 < 0
                    or y1 < 0
                    or x2 > 1
                    or y2 > 1
                ):
                    raise ValueError(
                        "box outside image"
                    )

            valid += 1

            rows.append({
                "split": split,
                "file": image.name,
                "status": "valid"
            })

        except Exception as e:

            invalid += 1

            rows.append({
                "split": split,
                "file": image.name,
                "status": f"invalid: {e}"
            })

for split in SPLITS:

    label_dir = LABEL_ROOT / split
    image_stems = {
        p.stem.lower()
        for p in (IMAGE_ROOT / split).rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    }

    for label in label_dir.rglob("*.txt"):

        if label.stem.lower() not in image_stems:

            missing_image += 1

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("Valid images/labels :", valid)
print("Invalid labels     :", invalid)
print("Missing labels     :", missing_label)
print("Missing images     :", missing_image)

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "file",
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("Report:")
print(REPORT)

if (
    valid > 0
    and invalid == 0
    and missing_label == 0
    and missing_image == 0
):

    print()
    print("=" * 70)
    print("PHASE 8 FINISHED SUCCESSFULLY")
    print("=" * 70)

else:

    print()
    print("PHASE 8 STILL REQUIRES VALIDATION/REPAIR")