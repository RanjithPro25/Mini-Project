from pathlib import Path
import json
import csv

BASE = Path(__file__).resolve().parent

KB_DIR = BASE / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FILE = KB_DIR / "class_to_idx.json"

CSV_FILE = KB_DIR / "snake_knowledge.csv"
JSON_FILE = KB_DIR / "snake_knowledge.json"
SOURCE_FILE = KB_DIR / "sources.csv"

# --------------------------------------------------
# FIND CLASS MAPPING
# --------------------------------------------------

if not CLASS_FILE.exists():

    search_paths = [
        BASE / "class_to_idx.json",
        BASE / "VenomGuard_EfficientNet_B3" / "class_to_idx.json",
        BASE / "models" / "class_to_idx.json",
        BASE / "models" / "efficientnet" / "class_to_idx.json"
    ]

    found = None

    for p in search_paths:

        if p.exists():
            found = p
            break

    if found:

        CLASS_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        CLASS_FILE.write_bytes(
            found.read_bytes()
        )

        print(
            "class_to_idx.json found:"
        )

        print(found)

        print(
            "Copied to:"
        )

        print(CLASS_FILE)

    else:

        print()
        print(
            "ERROR: class_to_idx.json was not found."
        )

        print()
        print(
            "Place your class_to_idx.json here:"
        )

        print(CLASS_FILE)

        raise FileNotFoundError(
            "class_to_idx.json not found"
        )


# --------------------------------------------------
# LOAD CLASS MAPPING
# --------------------------------------------------

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:

    class_to_idx = json.load(f)


class_to_idx = {
    str(k): int(v)
    for k, v in class_to_idx.items()
}

classes = sorted(
    class_to_idx,
    key=lambda x: class_to_idx[x]
)

print()
print(
    "Classes found:",
    len(classes)
)

if len(classes) != 313:

    raise ValueError(
        f"Expected 313 classes, found {len(classes)}"
    )


# --------------------------------------------------
# DATABASE FIELDS
# --------------------------------------------------

fields = [
    "class_index",
    "scientific_name",
    "common_name",
    "family",
    "venomous",
    "venomous_scale",
    "identification_features",
    "habitat",
    "distribution",
    "behavior",
    "specific_precautions",
    "what_not_to_do",
    "snakebite_first_aid",
    "emergency_level",
    "medical_attention",
    "source"
]


# --------------------------------------------------
# LOAD EXISTING DATABASE
# --------------------------------------------------

existing = {}

if CSV_FILE.exists():

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            name = row.get(
                "scientific_name",
                ""
            ).strip()

            if name:

                existing[name] = row


# --------------------------------------------------
# CREATE / PRESERVE RECORDS
# --------------------------------------------------

def create_record(
    name,
    index
):

    if name in existing:

        row = existing[name]

        row["class_index"] = str(index)

        for field in fields:

            if field not in row:

                row[field] = ""

        return row


    return {
        "class_index":
            str(index),

        "scientific_name":
            name,

        "common_name":
            "",

        "family":
            "",

        "venomous":
            "unknown",

        "venomous_scale":
            "",

        "identification_features":
            "",

        "habitat":
            "",

        "distribution":
            "",

        "behavior":
            "",

        "specific_precautions":
            "Do not approach, handle, capture, or provoke the snake. Maintain a safe distance and keep people and pets away.",

        "what_not_to_do":
            "Do not attempt to catch, kill, touch, or provoke the snake.",

        "snakebite_first_aid":
            "Move away from the snake, keep the person calm and still, and seek urgent medical care if a snakebite is suspected.",

        "emergency_level":
            "unknown",

        "medical_attention":
            "Seek urgent medical evaluation if a snakebite is suspected.",

        "source":
            ""
    }


records = []

for name in classes:

    index = class_to_idx[name]

    record = create_record(
        name,
        index
    )

    records.append(record)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

with open(
    CSV_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(records)


# --------------------------------------------------
# SAVE JSON
# --------------------------------------------------

with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        indent=2,
        ensure_ascii=False
    )


# --------------------------------------------------
# SOURCES DATABASE
# --------------------------------------------------

source_fields = [
    "scientific_name",
    "source",
    "source_url",
    "verified",
    "verification_date"
]

source_existing = {}

if SOURCE_FILE.exists():

    with open(
        SOURCE_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            name = row.get(
                "scientific_name",
                ""
            ).strip()

            if name:

                source_existing[name] = row


with open(
    SOURCE_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=source_fields
    )

    writer.writeheader()

    for name in classes:

        row = source_existing.get(
            name,
            {
                "scientific_name":
                    name,

                "source":
                    "",

                "source_url":
                    "",

                "verified":
                    "no",

                "verification_date":
                    ""
            }
        )

        writer.writerow(row)


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

existing_count = len(existing)

source_count = sum(
    1
    for r in records
    if r["source"].strip()
)

unknown_venom = sum(
    1
    for r in records
    if r["venomous"].strip().lower()
    == "unknown"
)


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print()
print("=" * 65)
print("VENOMGUARD PHASE 14")
print("KNOWLEDGE BASE")
print("=" * 65)

print()

print(
    "Class mapping:",
    CLASS_FILE
)

print(
    "Total model classes:",
    len(classes)
)

print(
    "Knowledge records:",
    len(records)
)

print(
    "Previously existing:",
    existing_count
)

print(
    "Records with sources:",
    source_count
)

print(
    "Unknown venom status:",
    unknown_venom
)

print()

print(
    "Created:"
)

print(
    CSV_FILE
)

print(
    JSON_FILE
)

print(
    SOURCE_FILE
)

print()

print("=" * 65)
print("PHASE 14 DATABASE STRUCTURE CREATED SUCCESSFULLY")
print("=" * 65)

print()

print(
    "IMPORTANT:"
)

print(
    "Species-specific venom status, precautions, and"
)

print(
    "medical information must be verified from"
)

print(
    "authoritative sources before production use."
)