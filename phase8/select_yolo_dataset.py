from pathlib import Path
import shutil
import csv
import random

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

SOURCE_DIR = PROJECT_DIR / "dataset_split"
OUTPUT_DIR = PROJECT_DIR / "yolo_dataset"

TRAIN_COUNT = 15
VAL_COUNT = 5
TEST_COUNT = 5

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(SEED)

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

manifest_file = OUTPUT_DIR / "phase8_species_manifest.csv"

existing = {}

if manifest_file.exists():
    with open(manifest_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["destination"]] = row

def save_manifest():
    rows = list(existing.values())

    rows.sort(
        key=lambda x: (
            x["split"],
            x["species_name"],
            x["destination"]
        )
    )

    temp = manifest_file.with_suffix(".tmp")

    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "species_name",
                "original_image",
                "destination",
                "label_file",
                "annotation_status"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    temp.replace(manifest_file)

def select_images(source_split, count):
    selected = []

    species_dirs = sorted(
        p for p in source_split.iterdir()
        if p.is_dir()
    )

    for species_dir in species_dirs:

        images = sorted(
            p for p in species_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            continue

        local_rng = random.Random(
            SEED + hash(species_dir.name) % 100000
        )

        local_rng.shuffle(images)

        selected.extend(
            (species_dir.name, image)
            for image in images[:min(count, len(images))]
        )

    return selected

total_new = 0
total_existing = 0

for split, count in [
    ("train", TRAIN_COUNT),
    ("val", VAL_COUNT),
    ("test", TEST_COUNT)
]:

    source_split = SOURCE_DIR / split

    if not source_split.exists():
        print(f"Missing source folder: {source_split}")
        continue

    selected = select_images(source_split, count)

    print()
    print("=" * 60)
    print(f"{split.upper()}")
    print(f"Selected candidates: {len(selected)}")
    print("=" * 60)

    for species, image_path in selected:

        new_name = f"{species}__{image_path.name}"

        destination = (
            OUTPUT_DIR /
            "images" /
            split /
            new_name
        )

        label_file = (
            OUTPUT_DIR /
            "labels" /
            split /
            f"{destination.stem}.txt"
        )

        key = str(destination.relative_to(PROJECT_DIR))

        if destination.exists() and destination.stat().st_size > 0:
            total_existing += 1
        else:
            temp = destination.with_suffix(destination.suffix + ".part")

            shutil.copy2(image_path, temp)
            temp.replace(destination)

            total_new += 1

        if key not in existing:

            existing[key] = {
                "split": split,
                "species_name": species,
                "original_image": str(
                    image_path.relative_to(PROJECT_DIR)
                ),
                "destination": key,
                "label_file": str(
                    label_file.relative_to(PROJECT_DIR)
                ),
                "annotation_status": "pending"
            }

    save_manifest()

data_yaml = OUTPUT_DIR / "data.yaml"

yaml_text = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: snake
"""

if not data_yaml.exists():
    temp = data_yaml.with_suffix(".tmp")
    temp.write_text(yaml_text, encoding="utf-8")
    temp.replace(data_yaml)

print()
print("=" * 60)
print("PHASE 8A SUMMARY")
print("=" * 60)
print(f"Already existing : {total_existing}")
print(f"Newly copied     : {total_new}")
print(f"Total manifest   : {len(existing)}")
print(f"Manifest         : {manifest_file}")
print(f"YAML             : {data_yaml}")
print()
print("PHASE 8A FINISHED SUCCESSFULLY")