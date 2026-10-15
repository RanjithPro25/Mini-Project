from pathlib import Path
import csv
import shutil

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET = PROJECT_DIR / "yolo_dataset"
MANIFEST = DATASET / "phase8_species_manifest.csv"

IMAGE_ROOT = DATASET / "images"
LABEL_ROOT = DATASET / "labels"

FINAL_ROOT = PROJECT_DIR / "yolo_dataset_final"

FINAL_IMAGES = FINAL_ROOT / "images"
FINAL_LABELS = FINAL_ROOT / "labels"

REPORT = PROJECT_DIR / "phase8" / "phase8_recovery_report.csv"

MIN_AREA = 0.0001
MAX_AREA = 0.98

print("=" * 75)
print("PHASE 8 - RECOVER EXISTING VALID ANNOTATIONS")
print("=" * 75)

if not MANIFEST.exists():
    raise RuntimeError(
        f"Manifest not found:\n{MANIFEST}"
    )

with open(
    MANIFEST,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

print("Phase 8A records:", len(rows))

if len(rows) != 4254:
    raise RuntimeError(
        f"Expected 4254 records, found {len(rows)}"
    )

all_labels = list(
    LABEL_ROOT.rglob("*.txt")
)

print("Existing label files:", len(all_labels))

label_by_name = {}

for label in all_labels:

    key = label.name.lower()

    label_by_name.setdefault(
        key,
        []
    ).append(label)


def validate_label(label):

    try:
        text = label.read_text(
            encoding="utf-8"
        ).strip()

    except Exception:
        return False, "read_error"

    if not text:
        return False, "empty"

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            return False, "format"

        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except Exception:
            return False, "numeric"

        if cls != 0:
            return False, "class"

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            return False, "coordinate"

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
            return False, "outside"

        area = w * h

        if area < MIN_AREA:
            return False, "too_small"

        if area > MAX_AREA:
            return False, "too_large"

    return True, "valid"


print()
print("STEP 1: Recovering labels")

recovered = 0
already_correct = 0
invalid = 0
not_found = 0
ambiguous = 0
image_missing = 0

report_rows = []

for index, row in enumerate(
    rows,
    start=1
):

    split = row["split"]

    species = row["species_name"]

    destination = Path(
        row["destination"]
    )

    image_name = destination.name

    image_stem = destination.stem

    intended_image = (
        IMAGE_ROOT
        / split
        / image_name
    )

    final_image = (
        FINAL_IMAGES
        / split
        / image_name
    )

    final_label = (
        FINAL_LABELS
        / split
        / f"{image_stem}.txt"
    )

    label_name = (
        f"{image_stem}.txt"
    )

    matches = label_by_name.get(
        label_name.lower(),
        []
    )

    if len(matches) == 0:

        not_found += 1

        report_rows.append({
            "split": split,
            "species": species,
            "image": image_name,
            "status": "no_label_found",
            "label_source": ""
        })

        continue

    if len(matches) > 1:

        exact = [
            p
            for p in matches
            if p.parent.name.lower()
            == split.lower()
        ]

        if len(exact) == 1:

            matches = exact

        else:

            ambiguous += 1

            report_rows.append({
                "split": split,
                "species": species,
                "image": image_name,
                "status": "ambiguous_label",
                "label_source": ";".join(
                    str(p)
                    for p in matches
                )
            })

            continue

    source_label = matches[0]

    ok, reason = validate_label(
        source_label
    )

    if not ok:

        invalid += 1

        report_rows.append({
            "split": split,
            "species": species,
            "image": image_name,
            "status": f"invalid_{reason}",
            "label_source": str(source_label)
        })

        continue

    if not intended_image.exists():

        candidates = list(
            (IMAGE_ROOT / split).rglob(
                image_name
            )
        )

        if candidates:
            intended_image = candidates[0]

    if not intended_image.exists():

        image_missing += 1

        report_rows.append({
            "split": split,
            "species": species,
            "image": image_name,
            "status": "image_not_found",
            "label_source": str(source_label)
        })

        continue

    final_image.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_label.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not final_image.exists():

        shutil.copy2(
            intended_image,
            final_image
        )

    if not final_label.exists():

        shutil.copy2(
            source_label,
            final_label
        )

        recovered += 1

        status = "recovered"

    else:

        already_correct += 1

        status = "already_present"

    report_rows.append({
        "split": split,
        "species": species,
        "image": image_name,
        "status": status,
        "label_source": str(source_label)
    })

    if index % 250 == 0:

        print(
            f"[{index:04d}/4254] "
            f"recovered={recovered} | "
            f"existing={already_correct} | "
            f"invalid={invalid} | "
            f"missing={not_found} | "
            f"ambiguous={ambiguous}"
        )


print()
print("STEP 2: Saving report")

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
            "species",
            "image",
            "status",
            "label_source"
        ]
    )

    writer.writeheader()
    writer.writerows(report_rows)

print()
print("=" * 75)
print("PHASE 8 RECOVERY RESULT")
print("=" * 75)

print("Phase 8A records :", len(rows))
print("Recovered        :", recovered)
print("Already present  :", already_correct)
print("Invalid labels   :", invalid)
print("Not found        :", not_found)
print("Ambiguous        :", ambiguous)
print("Missing images   :", image_missing)

print()
print("Final dataset:")
print(FINAL_ROOT)

print()
print("Report:")
print(REPORT)

print()
print("=" * 75)
print("RECOVERY FINISHED")
print("=" * 75)