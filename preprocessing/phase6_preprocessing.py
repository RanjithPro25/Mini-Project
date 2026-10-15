from pathlib import Path
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_DIR = PROJECT_DIR / "image_dataset" / "species"
OUTPUT_DIR = PROJECT_DIR / "processed_dataset" / "species"
CONFIG_FILE = PROJECT_DIR / "preprocessing" / "phase6_augmentation_config.json"

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

OUTPUT_FORMAT = "JPEG"
JPEG_QUALITY = 95


def process_image(source):

    relative_path = source.relative_to(INPUT_DIR)
    destination = OUTPUT_DIR / relative_path

    destination = destination.with_suffix(".jpg")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        with Image.open(source) as img:

            img = img.convert("RGB")

            img.save(
                destination,
                format=OUTPUT_FORMAT,
                quality=JPEG_QUALITY,
                optimize=True
            )

        return {
            "status": "success",
            "source": str(source),
            "output": str(destination)
        }

    except Exception as e:

        return {
            "status": "failed",
            "source": str(source),
            "output": "",
            "error": str(e)
        }


print("=" * 70)
print("VENOMGUARD - PHASE 6 : IMAGE PREPROCESSING & AUGMENTATION")
print("=" * 70)

print("\nScanning source dataset...")

files = [
    p
    for p in INPUT_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Source images : {len(files):,}")

if len(files) == 0:
    print("\nERROR: No images found.")
    raise SystemExit(1)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nConverting images to RGB JPEG...")

results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    for result in tqdm(
        executor.map(process_image, files),
        total=len(files),
        desc="Processing"
    ):
        results.append(result)

successful = [
    r for r in results
    if r["status"] == "success"
]

failed = [
    r for r in results
    if r["status"] == "failed"
]

print("\n" + "=" * 70)
print("PREPROCESSING RESULTS")
print("=" * 70)

print(f"Source images       : {len(files):,}")
print(f"Processed images    : {len(successful):,}")
print(f"Failed images       : {len(failed):,}")

if failed:

    failed_file = (
        PROJECT_DIR
        / "image_quality"
        / "phase6_failed_images.txt"
    )

    failed_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        failed_file,
        "w",
        encoding="utf-8"
    ) as f:

        for item in failed:
            f.write(
                item["source"]
                + " | "
                + item["error"]
                + "\n"
            )

    print(f"\nFailed image report:")
    print(failed_file)

augmentation_config = {
    "phase": 6,
    "augmentation_enabled": True,
    "apply_during_training": True,
    "horizontal_flip": {
        "enabled": True,
        "probability": 0.5
    },
    "small_rotation": {
        "enabled": True,
        "degrees": 10
    },
    "scaling": {
        "enabled": True,
        "scale_min": 0.9,
        "scale_max": 1.1
    },
    "translation": {
        "enabled": True,
        "translate_x": 0.1,
        "translate_y": 0.1
    },
    "brightness": {
        "enabled": True,
        "brightness_min": 0.85,
        "brightness_max": 1.15
    },
    "normalization": {
        "enabled": True,
        "method": "min_max",
        "formula": "(x - xmin) / (xmax - xmin)"
    },
    "output": {
        "format": "JPEG",
        "quality": JPEG_QUALITY,
        "color_mode": "RGB"
    },
    "note": "Augmentation will be applied dynamically after dataset splitting."
}

CONFIG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        augmentation_config,
        f,
        indent=4
    )

print("\nAugmentation configuration saved:")
print(CONFIG_FILE)

print("\n" + "=" * 70)
print("PHASE 6 SUMMARY")
print("=" * 70)

print("Preprocessing:")
print("  RGB conversion       : YES")
print("  JPEG standardization : YES")
print("  Original images      : PRESERVED")
print("  Dataset splitting    : NOT YET")

print("\nAugmentation:")
print("  Horizontal flip      : YES")
print("  Small rotation       : YES")
print("  Scaling              : YES")
print("  Translation           : YES")
print("  Brightness adjustment: YES")

print("\nAugmentation mode:")
print("  Dynamic during training")

print("\n" + "=" * 70)
print("PHASE 6 FINISHED SUCCESSFULLY")
print("=" * 70)