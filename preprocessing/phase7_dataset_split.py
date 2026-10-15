from pathlib import Path
import shutil
import random
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_DIR = PROJECT_DIR / "processed_dataset" / "species"
OUTPUT_DIR = PROJECT_DIR / "dataset_split"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

MANIFEST_FILE = OUTPUT_DIR / "phase7_split_manifest.csv"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42
MAX_WORKERS = 16

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 0.0001:
    raise ValueError("Split ratios must add up to 1.0")


def get_split(images):

    images = sorted(images, key=lambda x: str(x))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)

    n = len(images)

    if n == 1:
        return images, [], []

    if n == 2:
        return images[:1], images[1:], []

    if n == 3:
        return images[:1], images[1:2], images[2:]

    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)

    train_n = max(train_n, 1)
    val_n = max(val_n, 1)

    test_n = n - train_n - val_n

    if test_n < 1:

        test_n = 1

        if train_n > val_n:
            train_n -= 1
        else:
            val_n -= 1

    train = images[:train_n]
    val = images[train_n:train_n + val_n]
    test = images[train_n + val_n:]

    return train, val, test


def destination_for(source, species, split):

    if split == "train":
        folder = TRAIN_DIR
    elif split == "val":
        folder = VAL_DIR
    else:
        folder = TEST_DIR

    return folder / species / source.name


def copy_image(record):

    source, species, split = record

    destination = destination_for(
        source,
        species,
        split
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists() and destination.stat().st_size > 0:

        return {
            "status": "already_exists",
            "source": str(source),
            "destination": str(destination),
            "species": species,
            "split": split
        }

    temp = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:

        shutil.copy2(
            source,
            temp
        )

        if not temp.exists() or temp.stat().st_size == 0:
            raise RuntimeError("Copied file is empty")

        temp.replace(destination)

        return {
            "status": "copied",
            "source": str(source),
            "destination": str(destination),
            "species": species,
            "split": split
        }

    except Exception as e:

        if temp.exists():
            try:
                temp.unlink()
            except:
                pass

        return {
            "status": "failed",
            "source": str(source),
            "destination": str(destination),
            "species": species,
            "split": split,
            "error": str(e)
        }


print("=" * 70)
print("VENOMGUARD - PHASE 7 : RESUME DATASET SPLITTING")
print("=" * 70)

if not INPUT_DIR.exists():

    print("\nERROR: Phase 6 dataset not found.")
    print(INPUT_DIR)
    raise SystemExit(1)

print("\nChecking input dataset...")

species_dirs = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.is_dir()
])

print(f"Species folders found : {len(species_dirs):,}")

species_images = defaultdict(list)

print("\nCollecting images by species...")

for species_dir in tqdm(
    species_dirs,
    desc="Scanning species"
):

    species = species_dir.name

    images = [
        p for p in species_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if images:
        species_images[species].extend(images)

total_images = sum(
    len(images)
    for images in species_images.values()
)

print(f"Total images : {total_images:,}")

if total_images == 0:
    print("\nERROR: No images found.")
    raise SystemExit(1)

print("\nRecreating the SAME deterministic split...")

split_records = []

expected_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

for species in sorted(species_images):

    train, val, test = get_split(
        species_images[species]
    )

    for image in train:
        split_records.append(
            (image, species, "train")
        )

    for image in val:
        split_records.append(
            (image, species, "val")
        )

    for image in test:
        split_records.append(
            (image, species, "test")
        )

    expected_counts["train"] += len(train)
    expected_counts["val"] += len(val)
    expected_counts["test"] += len(test)

print("\n" + "=" * 70)
print("EXPECTED SPLIT")
print("=" * 70)

print(
    f"Train        : {expected_counts['train']:,} "
    f"({expected_counts['train'] / total_images * 100:.2f}%)"
)

print(
    f"Validation   : {expected_counts['val']:,} "
    f"({expected_counts['val'] / total_images * 100:.2f}%)"
)

print(
    f"Test         : {expected_counts['test']:,} "
    f"({expected_counts['test'] / total_images * 100:.2f}%)"
)

if (
    expected_counts["train"]
    + expected_counts["val"]
    + expected_counts["test"]
    != total_images
):
    print("\nERROR: Split count mismatch.")
    raise SystemExit(1)

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

print("\nChecking existing files...")

pending = []
already_exists = 0

for record in tqdm(
    split_records,
    desc="Checking progress"
):

    source, species, split = record

    destination = destination_for(
        source,
        species,
        split
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        already_exists += 1
    else:
        pending.append(record)

print("\n" + "=" * 70)
print("RESUME STATUS")
print("=" * 70)

print(f"Total assigned       : {len(split_records):,}")
print(f"Already copied       : {already_exists:,}")
print(f"Remaining to copy    : {len(pending):,}")

if len(pending) == 0:

    print("\nAll files are already copied.")
    
else:

    print("\nContinuing from previous execution...")

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        for result in tqdm(
            executor.map(copy_image, pending),
            total=len(pending),
            desc="Resuming"
        ):

            results.append(result)

    copied = sum(
        1 for r in results
        if r["status"] == "copied"
    )

    existing = sum(
        1 for r in results
        if r["status"] == "already_exists"
    )

    failed = [
        r for r in results
        if r["status"] == "failed"
    ]

    print("\nCopy operation completed.")

    print(f"Newly copied     : {copied:,}")
    print(f"Already existed  : {existing:,}")
    print(f"Failed           : {len(failed):,}")

    if failed:

        failed_file = (
            OUTPUT_DIR
            / "phase7_failed_images.csv"
        )

        with open(
            failed_file,
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
                    "error"
                ]
            )

            writer.writeheader()

            for item in failed:
                writer.writerow(item)

        print("\nFailed file report:")
        print(failed_file)

print("\nBuilding final manifest...")

manifest_records = []

for source, species, split in split_records:

    destination = destination_for(
        source,
        species,
        split
    )

    status = (
        "copied"
        if destination.exists()
        and destination.stat().st_size > 0
        else "missing"
    )

    manifest_records.append({
        "source": str(source),
        "destination": str(destination),
        "species": species,
        "split": split,
        "status": status
    })

with open(
    MANIFEST_FILE,
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
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(manifest_records)

final_train = sum(
    1 for r in manifest_records
    if r["split"] == "train"
    and r["status"] == "copied"
)

final_val = sum(
    1 for r in manifest_records
    if r["split"] == "val"
    and r["status"] == "copied"
)

final_test = sum(
    1 for r in manifest_records
    if r["split"] == "test"
    and r["status"] == "copied"
)

missing = [
    r for r in manifest_records
    if r["status"] == "missing"
]

print("\n" + "=" * 70)
print("FINAL PHASE 7 VERIFICATION")
print("=" * 70)

print(f"Expected total     : {total_images:,}")
print(f"Train               : {final_train:,}")
print(f"Validation          : {final_val:,}")
print(f"Test                : {final_test:,}")
print(f"Missing             : {len(missing):,}")

print(
    f"Train percentage    : "
    f"{final_train / total_images * 100:.2f}%"
)

print(
    f"Validation percentage: "
    f"{final_val / total_images * 100:.2f}%"
)

print(
    f"Test percentage     : "
    f"{final_test / total_images * 100:.2f}%"
)

print("\nManifest:")
print(MANIFEST_FILE)

if len(missing) == 0:

    print("\nAll 36,966 images are present in their assigned split.")

    print("\n" + "=" * 70)
    print("PHASE 7 FINISHED SUCCESSFULLY")
    print("=" * 70)

else:

    print("\nWARNING: Some assigned images are still missing.")
    print("Run this same script again to resume.")

    print("\n" + "=" * 70)
    print("PHASE 7 INCOMPLETE")
    print("=" * 70)