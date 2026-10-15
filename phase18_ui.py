
# VenomGuard Phase 18 - Localhost UI
# Streamlit UI helpers for the main application.
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def svg_data_uri(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(raw.encode()).decode()


def confidence_color(conf: float) -> str:
    if conf >= 0.70:
        return "#39e36b"
    if conf >= 0.50:
        return "#f5c542"
    return "#ff6262"


def confidence_label(conf: float) -> str:
    if conf >= 0.70:
        return "HIGH CONFIDENCE"
    if conf >= 0.50:
        return "UNCERTAIN"
    return "LOW CONFIDENCE"


def normalize_venom(value: Any) -> str:
    s = "" if value is None else str(value).strip().lower()
    if s in {"yes", "true", "1", "venomous", "poisonous"}:
        return "VENOMOUS"
    if s in {"no", "false", "0", "non-venomous", "nonvenomous", "non venomous"}:
        return "NON-VENOMOUS"
    return "UNRESOLVED"


def field(record: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def make_xai(record: Dict[str, Any], confidence: float, source: str) -> List[str]:
    evidence: List[str] = []
    features = field(record, "identification_features", "visual_features", "features")
    family = field(record, "family", "family_name")
    genus = field(record, "genus")
    distribution = field(record, "distribution", "origin_country")

    if features:
        evidence.append(f"Recorded visual features: {features}.")
    else:
        evidence.append("Phase 14 has no populated visual-feature field for this species.")
    if family:
        evidence.append(f"Taxonomic family in the knowledge base: {family}.")
    if genus:
        evidence.append(f"Genus in the knowledge base: {genus}.")
    if distribution:
        evidence.append(f"Recorded distribution/origin: {distribution}.")
    evidence.append(f"Inference source: {source}.")
    evidence.append(f"Model confidence: {confidence * 100:.2f}%.")
    return evidence


def recommendation(record: Dict[str, Any] | None, confidence: float) -> Dict[str, Any]:
    if record is None or confidence < 0.50:
        return {
            "risk": "UNKNOWN",
            "decision": "DO NOT RELY ON SPECIES PREDICTION",
            "items": [
                "Keep a safe distance.",
                "Do not touch, catch, or kill the snake.",
                "Do not use this low-confidence result for species-specific safety decisions.",
                "If a bite occurred, seek urgent professional medical care.",
            ],
        }

    venom = normalize_venom(field(
        record, "venomous", "venom_status", "venomous_status"
    ))

    if venom == "VENOMOUS":
        items = [
            "Do not approach or handle the snake.",
            "Maintain a safe distance.",
            "If a bite occurred, seek urgent professional medical care immediately.",
        ]
        first_aid = field(record, "snakebite_first_aid", "first_aid", "medical_attention")
        dont = field(record, "what_not_to_do", "dont", "do_not")
        if first_aid:
            items.append(first_aid)
        if dont:
            items.append("Avoid: " + dont)
        return {
            "risk": "HIGH",
            "decision": "URGENT CAUTION",
            "items": items,
        }

    if venom == "NON-VENOMOUS":
        return {
            "risk": "LOWER IDENTIFIED RISK",
            "decision": "KEEP DISTANCE",
            "items": [
                "Do not approach or handle the snake.",
                "Allow the snake to move away.",
            ],
        }

    return {
        "risk": "UNRESOLVED",
        "decision": "CAUTION",
        "items": [
            "Do not approach or handle the snake.",
            "Treat venom status as unresolved.",
            "If a bite occurred, seek urgent professional medical care.",
        ],
    }
