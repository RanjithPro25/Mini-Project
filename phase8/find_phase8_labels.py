from pathlib import Path
import csv
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

YOLO_DATASET = PROJECT_DIR / "yolo_dataset"
MANIFEST = YOLO_DATASET / "phase8_species_manifest.csv"

IMAGE_ROOT = YOLO_DATASET / "images"
LABEL_ROOT = YOLO_DATASET / "labels"

print("=" * 75)
print("PHASE 8 - LOCATE EXISTING 4,254 LABELS")
print("=" * 75)

print("\nSTEP 1: Reading Phase 8A manifest")

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

print("Manifest records:", len(rows))

if len(rows) != 4254:
    raise RuntimeError(
        f"Expected 4254 records, found {len(rows)}"
    )

print("\nManifest columns:")
print(list(rows[0].keys()))

print("\nFirst 5 records:")

for row in rows[:5]:
    print(row)

print("\nSTEP 2: Indexing ALL existing labels")

label_files = list(
    LABEL_ROOT.rglob("*.txt")
)

print(
    "Existing label files:",
    len(label_files)
)

label_by_name = defaultdict(list)
label_by_stem = defaultdict(list)

for label in label_files:

    label_by_name[
        label.name.lower()
    ].append(label)

    label_by_stem[
        label.stem.lower()
    ].append(label)

print("Label filename index created")

print("\nSTEP 3: Matching Phase 8A images to labels")

matched = []
missing = []
ambiguous = []

for index, row in enumerate(
    rows,
    start=1
):

    source = row.get("source", "")
    destination = row.get("destination", "")

    candidates = []

    for value in [
        destination,
        source
    ]:

        if value:

            p = Path(value)

            candidates.append(p.name)

    found = []

    for filename in candidates:

        found.extend(
            label_by_name.get(
                Path(filename).stem.lower() + ".txt",
                []
            )
        )

    unique = []

    seen = set()

    for item in found:

        key = str(item).lower()

        if key not in seen:

            seen.add(key)
            unique.append(item)

    if len(unique) == 1:

        matched.append(
            (
                row,
                unique[0]
            )
        )

    elif len(unique) > 1:

        ambiguous.append(
            (
                row,
                unique
            )
        )

    else:

        stem = None

        if candidates:

            stem = Path(
                candidates[0]
            ).stem.lower()

        if stem:

            stem_matches = label_by_stem.get(
                stem,
                []
            )

            if len(stem_matches) == 1:

                matched.append(
                    (
                        row,
                        stem_matches[0]
                    )
                )

            elif len(stem_matches) > 1:

                ambiguous.append(
                    (
                        row,
                        stem_matches
                    )
                )

            else:

                missing.append(row)

        else:

            missing.append(row)

    if index % 500 == 0:

        print(
            f"{index}/4254 | "
            f"matched={len(matched)} | "
            f"missing={len(missing)} | "
            f"ambiguous={len(ambiguous)}"
        )

print("\n" + "=" * 75)
print("MATCHING RESULT")
print("=" * 75)

print("Phase 8A images :", len(rows))
print("Matched labels  :", len(matched))
print("Missing labels  :", len(missing))
print("Ambiguous       :", len(ambiguous))

print("\nSTEP 4: Matched label locations")

for row, label in matched[:20]:

    print()
    print("Species :", row.get("species", ""))
    print("Split   :", row.get("split", ""))
    print("Image   :", row.get("destination", ""))
    print("Label   :", label)

print("\nSTEP 5: Split statistics")

split_total = defaultdict(int)
split_matched = defaultdict(int)
split_missing = defaultdict(int)

for row in rows:

    split = row.get("split", "")

    split_total[split] += 1

for row, label in matched:

    split = row.get("split", "")

    split_matched[split] += 1

for row in missing:

    split = row.get("split", "")

    split_missing[split] += 1

for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"{split:5s} | "
        f"total={split_total[split]} | "
        f"matched={split_matched[split]} | "
        f"missing={split_missing[split]}"
    )

print("\nSTEP 6: Saving diagnostic report")

REPORT = (
    PROJECT_DIR
    / "phase8"
    / "phase8_label_location_report.csv"
)

with open(
    REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "source",
        "destination",
        "species",
        "split",
        "label_status",
        "label_path"
    ])

    matched_set = {
        str(label).lower(): label
        for row, label in matched
    }

    for row in rows:

        source = row.get("source", "")
        destination = row.get("destination", "")
        species = row.get("species", "")
        split = row.get("split", "")

        candidates = []

        for value in [
            destination,
            source
        ]:

            if value:
                candidates.append(
                    Path(value).stem.lower()
                )

        label = None

        for stem in candidates:

            matches = label_by_stem.get(
                stem,
                []
            )

            if len(matches) == 1:

                label = matches[0]
                break

        if label:

            status = "matched"
            label_path = str(label)

        else:

            status = "missing_or_ambiguous"
            label_path = ""

        writer.writerow([
            source,
            destination,
            species,
            split,
            status,
            label_path
        ])

print("Report:")
print(REPORT)

print("\n" + "=" * 75)
print("DIAGNOSTIC FINISHED")
print("=" * 75)