
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ============================================================
# 1. PERFORMANCE SETTINGS
# ============================================================

CPU_CORES = os.cpu_count() or 4
TORCH_THREADS = max(1, min(8, CPU_CORES))

try:
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 2. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

YOLO_MODEL = (
    ROOT / "models" / "yolov8_snake" /
    "training" / "weights" / "best.pt"
)

EFFICIENTNET_MODEL = (
    ROOT / "models" / "efficientnet_b3" / "best.pt"
)

CLASS_FILE = ROOT / "knowledge_base" / "class_to_idx.json"

KNOWLEDGE_FILE = (
    ROOT / "knowledge_base" / "snake_knowledge.json"
)

OUT = ROOT / "phase17_outputs"
OUT.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUT / "phase17_image_results.csv"
REPORT_JSON = OUT / "phase17_unified_completion_report.json"
REPORT_CSV = OUT / "phase17_unified_completion_report.csv"


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

YOLO_CONF = 0.70
YOLO_IMGSZ_FAST = 512
YOLO_IMGSZ_RETRY = 640

HIGH = 0.70
UNCERTAIN = 0.50

# Minimum accepted bounding-box dimensions.
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20

# Maximum detections examined before selecting a snake.
YOLO_MAX_DETECTIONS = 30

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. GLOBAL VARIABLES
# ============================================================

YOLO = None
EFF = None

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

KB = {}
P15 = None

SNAKE_CLASS_IDS = set()

CACHE_MAX = 32
RESULT_CACHE = OrderedDict()


# ============================================================
# 5. CSV FIELDS
# ============================================================

FIELDS = [
    "timestamp",
    "image",
    "image_sha256",
    "width",
    "height",
    "yolo_status",
    "yolo_confidence",
    "yolo_box",
    "classification_source",
    "scientific_name",
    "common_name",
    "classification_confidence",
    "confidence_percent",
    "confidence_category",
    "venom_status",
    "risk",
    "decision",
    "top5",
    "error",
    "cache_hit",
    "message",
]


# ============================================================
# 6. IMAGE TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def norm(x: Any) -> str:
    x = clean(x).lower()
    x = x.replace("_", " ").replace("-", " ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# 8. LOAD CLASS MAPPING
# ============================================================

def load_classes():
    global CLASS_TO_IDX, IDX_TO_CLASS

    data = json.loads(
        CLASS_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict) or len(data) != 313:
        raise ValueError(
            "Expected exactly 313 EfficientNet classes."
        )

    CLASS_TO_IDX = {
        str(k): int(v)
        for k, v in data.items()
    }

    IDX_TO_CLASS = {
        v: k
        for k, v in CLASS_TO_IDX.items()
    }

    if len(IDX_TO_CLASS) != 313:
        raise ValueError(
            "Invalid class mapping: duplicate class indexes."
        )

    print("[PASS] 313 EfficientNet classes loaded")


# ============================================================
# 9. LOAD KNOWLEDGE BASE
# ============================================================

def load_kb():
    global KB

    candidates = [
        KNOWLEDGE_FILE,
        ROOT / "knowledge_base" / "snake_knowledge_final.json",
        ROOT / "snake_knowledge.json",
    ]

    p = next(
        (x for x in candidates if x.exists()),
        None
    )

    if not p:
        raise FileNotFoundError(
            "Phase 14 snake knowledge JSON not found."
        )

    data = json.loads(
        p.read_text(encoding="utf-8")
    )

    if isinstance(data, dict):
        for k in (
            "species",
            "snakes",
            "records",
            "data",
            "knowledge",
        ):
            if isinstance(data.get(k), list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise ValueError(
            "Unsupported Phase 14 JSON structure."
        )

    KB = {}

    for r in data:
        if isinstance(r, dict):
            species = clean(
                r.get("scientific_name")
                or r.get("scientific")
                or r.get("species")
            )

            if species:
                KB[norm(species)] = r

    print(f"[PASS] Knowledge base loaded: {len(KB)} records")

    return p


# ============================================================
# 10. YOLO SNAKE CLASS IDENTIFICATION
# ============================================================

def get_snake_class_ids():
    """
    Identify the snake class IDs from YOLO model names.

    Single-class detector:
        Assumes class 0 is snake.

    Multi-class detector:
        Accepts class names containing 'snake'.

    If no snake class can be identified, the system fails
    safely rather than accepting arbitrary object detections.
    """

    global SNAKE_CLASS_IDS

    names = YOLO.names

    if isinstance(names, dict):
        class_names = names
    else:
        class_names = {
            i: name
            for i, name in enumerate(names)
        }

    print("[YOLO CLASSES]", class_names)

    if len(class_names) == 1:
        SNAKE_CLASS_IDS = {
            int(next(iter(class_names.keys())))
        }

        print(
            "[PASS] Single-class YOLO detector. "
            "Assuming its only class is snake."
        )

        return

    detected_ids = set()

    for class_id, class_name in class_names.items():
        name = norm(class_name)

        if "snake" in name:
            detected_ids.add(int(class_id))

    if not detected_ids:
        raise ValueError(
            "Could not identify snake class in the "
            "multi-class YOLO model. Check YOLO.names."
        )

    SNAKE_CLASS_IDS = detected_ids

    print(
        "[PASS] Snake class IDs:",
        sorted(SNAKE_CLASS_IDS)
    )


# ============================================================
# 11. LOAD YOLO MODEL
# ============================================================

def load_yolo():
    global YOLO

    if YOLO is not None:
        return

    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            f"YOLOv8 model not found: {YOLO_MODEL}"
        )

    from ultralytics import YOLO as UYOLO

    YOLO = UYOLO(str(YOLO_MODEL))

    get_snake_class_ids()

    print("[PASS] YOLOv8 loaded")


# ============================================================
# 12. EFFICIENTNET CHECKPOINT HELPERS
# ============================================================

def state_dict_from_checkpoint(c):

    if hasattr(c, "state_dict"):
        return c.state_dict()

    if isinstance(c, dict):

        for k in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            v = c.get(k)

            if isinstance(v, dict):
                return v

            if hasattr(v, "state_dict"):
                return v.state_dict()

        if c and all(
            isinstance(k, str) for k in c
        ) and any(
            torch.is_tensor(v) for v in c.values()
        ):
            return c

    raise ValueError(
        "Could not find EfficientNet state_dict."
    )


# ============================================================
# 13. LOAD EFFICIENTNET
# ============================================================

def load_eff():
    global EFF

    if EFF is not None:
        return

    if not EFFICIENTNET_MODEL.exists():
        raise FileNotFoundError(
            f"EfficientNet model not found: "
            f"{EFFICIENTNET_MODEL}"
        )

    ckpt = torch.load(
        str(EFFICIENTNET_MODEL),
        map_location="cpu",
        weights_only=False
    )

    state = state_dict_from_checkpoint(ckpt)

    state = {
        (
            k[7:] if k.startswith("module.")
            else k[6:] if k.startswith("model.")
            else k
        ): v
        for k, v in state.items()
    }

    model = models.efficientnet_b3(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        313
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False
    )

    if missing:
        raise ValueError(
            f"EfficientNet checkpoint incompatible: "
            f"{len(missing)} missing keys."
        )

    if unexpected:
        raise ValueError(
            f"EfficientNet checkpoint contains "
            f"{len(unexpected)} unexpected keys."
        )

    cmap = (
        ckpt.get("class_to_idx")
        if isinstance(ckpt, dict)
        else None
    )

    if isinstance(cmap, dict):
        if cmap != CLASS_TO_IDX:
            raise ValueError(
                "Checkpoint class mapping does not "
                "match project mapping."
            )

    EFF = model.to(DEVICE).eval()

    if DEVICE.type == "cpu":
        try:
            EFF = EFF.to(
                memory_format=torch.channels_last
            )
        except Exception:
            pass

    print(
        f"[PASS] EfficientNet-B3 loaded | "
        f"device={DEVICE} | threads={TORCH_THREADS}"
    )


# ============================================================
# 14. LOAD PHASE 15
# ============================================================

def load_phase15():
    global P15

    if P15 is not None:
        return

    p = ROOT / "phase15_complete.py"

    if not p.exists():
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "vg_p15",
            str(p)
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "rank_description"):
            P15 = module

    except Exception as e:
        print(
            "[WARNING] Phase 15 unavailable:",
            e
        )


# ============================================================
# 15. EFFICIENTNET CLASSIFICATION
# ============================================================

def classify(img):

    if EFF is None:
        raise RuntimeError(
            "EfficientNet model is not loaded."
        )

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cpu":
        try:
            x = x.contiguous(
                memory_format=torch.channels_last
            )
        except Exception:
            pass

    with torch.inference_mode():
        logits = EFF(x)

        probabilities = torch.softmax(
            logits, dim=1
        )[0]

        values, ids = torch.topk(
            probabilities, 5
        )

    return [
        {
            "scientific_name": IDX_TO_CLASS[int(i)],
            "confidence": float(v),
        }
        for v, i in zip(values, ids)
    ]


# ============================================================
# 16. YOLO DETECTION
# ============================================================

def _yolo_once(img, imgsz):

    # Fail safely instead of producing a species label without a validated
    # snake detector.
    if YOLO is None or not SNAKE_CLASS_IDS:
        raise RuntimeError(
            "Snake detection is unavailable: YOLOv8 is not loaded or has no validated snake class."
        )

    rgb = np.asarray(img)
    bgr = rgb[:, :, ::-1].copy()

    results = YOLO.predict(
        source=bgr,
        conf=YOLO_CONF,
        imgsz=imgsz,
        max_det=YOLO_MAX_DETECTIONS,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes

    candidates = []

    for i in range(len(boxes)):

        confidence = float(
            boxes.conf[i].item()
        )

        class_id = int(
            boxes.cls[i].item()
        )

        # CRITICAL: Reject non-snake classes.
        if class_id not in SNAKE_CLASS_IDS:
            continue

        coordinates = [
            float(x)
            for x in boxes.xyxy[i].tolist()
        ]

        x1, y1, x2, y2 = coordinates

        # Reject invalid coordinates.
        if (
            not np.all(np.isfinite(coordinates))
            or x2 <= x1
            or y2 <= y1
        ):
            continue

        candidates.append({
            "confidence": confidence,
            "class_id": class_id,
            "box": coordinates,
        })

    if not candidates:
        return None

    # Select the highest-confidence snake detection.
    return max(
        candidates,
        key=lambda x: x["confidence"]
    )


def yolo_best(img):

    detection = _yolo_once(
        img,
        YOLO_IMGSZ_FAST
    )

    if detection is None:
        detection = _yolo_once(
            img,
            YOLO_IMGSZ_RETRY
        )

    return detection


# ============================================================
# 17. VALIDATE AND CROP SNAKE
# ============================================================

def crop(img, detection):

    if not detection or "box" not in detection:
        return None

    width, height = img.size

    x1, y1, x2, y2 = detection["box"]

    if not np.all(
        np.isfinite([x1, y1, x2, y2])
    ):
        return None

    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))

    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < MIN_BOX_WIDTH:
        return None

    if crop_height < MIN_BOX_HEIGHT:
        return None

    cropped = img.crop(
        (x1, y1, x2, y2)
    )

    if cropped.width == 0 or cropped.height == 0:
        return None

    return cropped


# ============================================================
# 18. VENOM STATUS
# ============================================================

def venom(record):

    if not record:
        return "UNRESOLVED"

    value = norm(
        record.get("venomous")
        or record.get("venom_status")
        or record.get("venomous_status")
    )

    if value in {
        "yes",
        "true",
        "1",
        "venomous",
        "poisonous",
    }:
        return "VENOMOUS"

    if value in {
        "no",
        "false",
        "0",
        "non venomous",
        "nonvenomous",
    }:
        return "NON-VENOMOUS"

    return "UNRESOLVED"


# ============================================================
# 19. SAFETY RECOMMENDATIONS
# ============================================================

def recommendation(sc, conf):

    record = KB.get(norm(sc))

    if conf < UNCERTAIN:
        return {
            "risk": "UNKNOWN / LOW CONFIDENCE",
            "decision": "DO NOT RELY ON SPECIES PREDICTION",
            "actions": [
                "Keep a safe distance.",
                "Do not handle the snake.",
                "If a bite occurred, seek urgent "
                "professional medical care.",
            ],
        }

    if not record:
        return {
            "risk": "UNKNOWN",
            "decision": "DO NOT HANDLE",
            "actions": [
                "Keep a safe distance.",
                "Do not handle the snake.",
                "If a bite occurred, seek urgent "
                "professional medical care.",
            ],
        }

    status = venom(record)

    if status == "VENOMOUS":

        actions = [
            "Do not approach or handle the snake.",
            "Maintain a safe distance.",
            "If a bite occurred, seek urgent "
            "professional medical care immediately.",
        ]

        if clean(record.get("snakebite_first_aid")):
            actions.append(
                clean(record["snakebite_first_aid"])
            )

        elif clean(record.get("medical_attention")):
            actions.append(
                clean(record["medical_attention"])
            )

        return {
            "risk": "HIGH",
            "decision": "URGENT CAUTION",
            "actions": actions,
        }

    if status == "NON-VENOMOUS":
        return {
            "risk": "LOWER IDENTIFIED RISK",
            "decision": "KEEP DISTANCE",
            "actions": [
                "Do not approach or handle the snake.",
                "Allow the snake to move away.",
            ],
        }

    return {
        "risk": "UNRESOLVED",
        "decision": "CAUTION",
        "actions": [
            "Do not approach or handle the snake.",
            "Treat venom status as unresolved.",
            "If a bite occurred, seek urgent "
            "professional medical care.",
        ],
    }


# ============================================================
# 20. NO SNAKE RESULT
# ============================================================

def no_snake_result(
    path,
    image,
    image_hash,
    error="no snake found"
):

    return {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "image": str(path.resolve()),
        "image_sha256": image_hash,

        "width": image.width,
        "height": image.height,

        "yolo_status": "NO_SNAKE_FOUND",
        "yolo_confidence": "",
        "yolo_box": "",

        "classification_source": "NONE",

        "scientific_name": "",
        "common_name": "",

        "classification_confidence": 0,
        "confidence_percent": 0,
        "confidence_category": "NOT APPLICABLE",

        "venom_status": "UNRESOLVED",
        "risk": "UNRESOLVED",
        "decision": "NO SNAKE DETECTED",

        "top5": "[]",

        "error": error,
        "cache_hit": False,
        "message": "No snake detected",
    }


# ============================================================
# 21. CACHE MANAGEMENT
# ============================================================

def cache_result(key, result):

    RESULT_CACHE[key] = result
    RESULT_CACHE.move_to_end(key)

    while len(RESULT_CACHE) > CACHE_MAX:
        RESULT_CACHE.popitem(last=False)


# ============================================================
# 22. MAIN IMAGE INFERENCE
# ============================================================

def image_infer(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image_hash = sha256(path)

    # Check repeated-image cache.
    if image_hash in RESULT_CACHE:

        RESULT_CACHE.move_to_end(image_hash)

        return dict(
            RESULT_CACHE[image_hash],
            cache_hit=True
        )

    # Load and correct image orientation.
    try:
        with Image.open(path) as original:
            img = ImageOps.exif_transpose(
                original
            ).convert("RGB")

    except Exception as e:
        raise ValueError(
            f"Could not read image: {e}"
        )

    # STEP 1: YOLO snake detection.
    detection = yolo_best(img)

    # STEP 2: No snake detection.
    if detection is None:

        result = no_snake_result(
            path,
            img,
            image_hash
        )

        cache_result(image_hash, result)

        return dict(result)

    # STEP 3: Crop only the detected snake.
    snake_crop = crop(
        img,
        detection
    )

    if snake_crop is None:

        result = no_snake_result(
            path,
            img,
            image_hash,
            "invalid snake detection box"
        )

        cache_result(image_hash, result)

        return dict(result)

    # STEP 4: Classify only the snake crop.
    top = classify(snake_crop)

    if not top:
        raise RuntimeError(
            "EfficientNet returned no results."
        )

    prediction = top[0]

    scientific_name = prediction[
        "scientific_name"
    ]

    confidence = prediction["confidence"]

    # EfficientNet always ranks one of the 313 classes. Do not present its
    # top class as an identification when the model confidence is weak.
    # YOLO did detect an object, so report uncertainty rather than inventing
    # a species name.
    if confidence < UNCERTAIN:
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image": str(path.resolve()),
            "image_sha256": image_hash,
            "width": img.width,
            "height": img.height,
            "yolo_status": "SPECIES_UNCERTAIN",
            "yolo_confidence": detection["confidence"],
            "yolo_box": json.dumps(detection["box"]),
            "classification_source": "YOLOv8 crop (unconfirmed)",
            "scientific_name": "",
            "common_name": "",
            "classification_confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "confidence_category": "LOW CONFIDENCE",
            "venom_status": "UNRESOLVED",
            "risk": "UNKNOWN",
            "decision": "SPECIES UNCERTAIN",
            "top5": "[]",
            "error": "",
            "cache_hit": False,
            "message": "A possible snake-like object was detected, but the species could not be identified reliably.",
        }
        cache_result(image_hash, result)
        return dict(result)

    record = KB.get(
        norm(scientific_name)
    )

    rec = recommendation(
        scientific_name,
        confidence
    )

    # STEP 5: Build final result.
    result = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "image": str(path.resolve()),
        "image_sha256": image_hash,

        "width": img.width,
        "height": img.height,

        "yolo_status": "DETECTED",
        "yolo_confidence": detection["confidence"],
        "yolo_box": json.dumps(
            detection["box"]
        ),

        "classification_source": "YOLOv8 crop",

        "scientific_name": scientific_name,

        "common_name": (
            clean(record.get("common_name"))
            if record else ""
        ),

        "classification_confidence": confidence,

        "confidence_percent": round(
            confidence * 100, 2
        ),

        "confidence_category": (
            "HIGH CONFIDENCE"
            if confidence >= HIGH
            else "UNCERTAIN"
            if confidence >= UNCERTAIN
            else "LOW CONFIDENCE"
        ),

        "venom_status": venom(record),

        "risk": rec["risk"],
        "decision": rec["decision"],

        "top5": json.dumps(top),

        "error": "",
        "cache_hit": False,
        "message": "",
    }

    cache_result(image_hash, result)

    return dict(result)


# ============================================================
# 23. CSV READING
# ============================================================

def read_rows():

    rows = {}

    if RESULT_CSV.exists():

        try:
            with RESULT_CSV.open(
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as f:

                for row in csv.DictReader(f):

                    if row.get("image"):
                        rows[row["image"]] = row

        except Exception as e:
            print(
                "[WARNING] Could not read previous CSV:",
                e
            )

    return rows


# ============================================================
# 24. CSV WRITING
# ============================================================

def write_rows(rows):

    with RESULT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for row in rows.values():

            writer.writerow({
                key: row.get(key, "")
                for key in FIELDS
            })


# ============================================================
# 25. ANALYZE ONE IMAGE
# ============================================================

def one_image():

    p = Path(
        input("\nEnter image path: ")
        .strip()
        .strip('"')
    )

    if not p.exists():
        print("[ERROR] Image not found")
        return

    try:

        result = image_infer(p)

        rows = read_rows()

        rows[str(p.resolve())] = result

        write_rows(rows)

        if result["yolo_status"] == "NO_SNAKE_FOUND":

            print("\nNO SNAKE FOUND")

        else:

            print("\nSPECIES:",
                  result["scientific_name"])

            print(
                "CONFIDENCE:",
                result["confidence_percent"],
                "%"
            )

            print(
                "VENOM:",
                result["venom_status"]
            )

            print(
                "RISK:",
                result["risk"]
            )

            print(
                "YOLO CONFIDENCE:",
                round(
                    result["yolo_confidence"] * 100,
                    2
                ),
                "%"
            )

    except Exception as e:

        print("[FAILED]", e)


# ============================================================
# 26. PROCESS IMAGE FOLDER
# ============================================================

def folder():

    folder_path = Path(
        input("\nEnter image folder path: ")
        .strip()
        .strip('"')
    )

    if not folder_path.is_dir():
        print("[ERROR] Folder not found")
        return

    images = sorted(
        p for p in folder_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
    )

    rows = read_rows()

    new = 0
    skipped = 0
    failed = 0

    for i, p in enumerate(images, 1):

        key = str(p.resolve())

        try:
            current_hash = sha256(p)

            old = rows.get(key)

            if (
                old
                and old.get("image_sha256")
                == current_hash
            ):
                skipped += 1

                print(
                    f"[{i}/{len(images)}] SKIPPED"
                )

                continue

            rows[key] = image_infer(p)

            new += 1

            write_rows(rows)

        except Exception as e:

            rows[key] = {
                "image": key,
                "image_sha256": "",
                "error": str(e),
            }

            failed += 1

            write_rows(rows)

        print(
            f"[{i}/{len(images)}] "
            f"processed={new} "
            f"skipped={skipped} "
            f"failed={failed}"
        )

    report = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "total": len(images),
        "processed_new": new,
        "skipped_existing": skipped,
        "failed": failed,

        "results": str(RESULT_CSV),

        "status": (
            "COMPLETE"
            if failed == 0
            else "COMPLETE_WITH_FAILURES"
        ),
    }

    REPORT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nFOLDER COMPLETE")
    print("Processed:", new)
    print("Skipped:", skipped)
    print("Failed:", failed)


# ============================================================
# 27. TEXT DESCRIPTION
# ============================================================

def description():

    load_phase15()

    if P15 is None:
        print("Phase 15 unavailable")
        return

    query = input(
        "\nDescribe the unknown snake: "
    ).strip()

    if not query:
        return

    try:

        candidates, features, confident = (
            P15.rank_description(query)
        )

        print(
            "\nSTATUS:",
            "LEADING CANDIDATE"
            if confident
            else "UNCERTAIN - RANKED CANDIDATES"
        )

        for i, candidate in enumerate(
            candidates, 1
        ):

            print(
                f"{i}. "
                f"{candidate['scientific_name']} "
                f"({candidate['ranking_confidence'] * 100:.2f}%)"
            )

        print(
            "Do not approach or handle the snake."
        )

    except Exception as e:
        print("[FAILED]", e)


# ============================================================
# 28. MODEL HASH VERIFICATION
# ============================================================

def verify_hashes():

    print(
        "YOLO SHA256:",
        sha256(YOLO_MODEL)
    )

    print(
        "EfficientNet SHA256:",
        sha256(EFFICIENTNET_MODEL)
    )

    print("Device:", DEVICE)


# ============================================================
# 29. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("VENOMGUARD FAST PHASE 17")
    print("=" * 70)

    print("\nLoading models...")

    load_classes()
    load_kb()
    load_yolo()
    load_eff()

    print("\n" + "=" * 70)

    print(
        f"READY | device={DEVICE} | "
        f"threads={TORCH_THREADS} | "
        f"YOLO={YOLO_IMGSZ_FAST}->{YOLO_IMGSZ_RETRY}"
    )

    print("=" * 70)

    while True:

        print(
            "\n1. Analyze one image"
            "\n2. Process image folder"
            "\n3. Unknown snake text description"
            "\n4. Verify model hashes"
            "\n5. Exit"
        )

        choice = input(
            "\nSelect: "
        ).strip()

        if choice == "1":
            one_image()

        elif choice == "2":
            folder()

        elif choice == "3":
            description()

        elif choice == "4":
            verify_hashes()

        elif choice == "5":
            print("Exiting Phase 17...")
            break

        else:
            print("Select 1-5")


if __name__ == "__main__":
    main()
