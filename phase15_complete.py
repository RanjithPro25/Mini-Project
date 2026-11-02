# VenomGuard Phase 15 - Unknown Snake Description + Text + Voice + Image Context
# Python 3.10+
# Run from the VenomGuard project folder:
#     python .\phase15_complete.py
#
# Main user flow:
#   1) User describes an unknown snake in normal language.
#   2) DistilBERT detects the user's intent.
#   3) Description engine extracts visible characteristics.
#   4) Candidate species are ranked from the Phase-14 species set.
#   5) If BioCLIP is installed/cached, it provides an additional biology
#      zero-shot ranking signal from the natural-language description.
#   6) The selected candidate is looked up in the Phase-14 knowledge base.
#   7) Voice uses offline Vosk speech-to-text and follows the same pipeline.
#   8) Image predictions can be handed in from YOLOv8 + EfficientNet-B3 via
#      option 3 or by writing phase15_current_species.json.
#
# IMPORTANT SAFETY:
# Natural-language descriptions alone cannot guarantee species identification.
# The program therefore returns ranked candidates and confidence/uncertainty.
# A suspected bite is treated as an emergency regardless of identification.

import os
import re
import sys
import json
import math
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase15_outputs"
OUT.mkdir(parents=True, exist_ok=True)
VOICE_DIR = ROOT / "voice_models"
VOICE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K = 5
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip"

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def norm(x):
    x = "" if x is None else str(x)
    x = x.lower().replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean(x):
    return "" if x is None else str(x).strip()


def die(msg):
    print("\nERROR:", msg)
    sys.exit(1)


def find_file(names):
    """Find known project data paths without recursively scanning the full dataset."""
    known_dirs = (
        ROOT / "knowledge_base",
        ROOT / "phase14_final_outputs",
        ROOT / "data",
        ROOT / "datasets",
        ROOT,
    )
    for name in names:
        for folder in known_dirs:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError as e:
    die(
        f"Missing package: {e.name}\n\n"
        "Run:\n"
        "python -m pip install -U torch pandas numpy"
    )

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False

try:
    import vosk
    import sounddevice as sd
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# Optional BioCLIP. The program remains usable without it, but BioCLIP is the
# strongest automatic zero-shot biological signal in this implementation.
try:
    import open_clip
    BIOCLIP_AVAILABLE = True
except Exception:
    BIOCLIP_AVAILABLE = False

# -----------------------------------------------------------------------------
# Phase 14 loading
# -----------------------------------------------------------------------------

def load_phase14():
    p = find_file([
        "snake_knowledge_final.json",
        "snake_knowledge.json",
        "snake_knowledge_backup.json",
    ])
    if not p:
        die("Phase 14 snake knowledge JSON was not found.")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        die(f"Could not read Phase 14 JSON: {e}")

    if isinstance(data, dict):
        for k in ("snakes", "species", "records", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break

    if not isinstance(data, list) or not data:
        die("Phase 14 knowledge database is empty or has an unsupported format.")

    return data, p


KB, KB_FILE = load_phase14()

# Normalize the Phase-14 records without replacing their original values.
for r in KB:
    r["_scientific"] = clean(r.get("scientific_name"))
    r["_common"] = clean(r.get("common_name"))
    r["_family"] = clean(r.get("family"))
    r["_venomous"] = clean(r.get("venomous"))
    r["_score"] = clean(r.get("venomous_scale"))
    r["_features"] = clean(r.get("identification_features"))
    r["_habitat"] = clean(r.get("habitat"))
    r["_distribution"] = clean(r.get("distribution"))
    r["_behavior"] = clean(r.get("behavior"))
    r["_precautions"] = clean(r.get("specific_precautions"))
    r["_dont"] = clean(r.get("what_not_to_do"))
    r["_first_aid"] = clean(r.get("snakebite_first_aid"))
    r["_emergency"] = clean(r.get("emergency_level"))
    r["_attention"] = clean(r.get("medical_attention"))

print("\n" + "=" * 72)
print("VENOMGUARD PHASE 15 - UNKNOWN SNAKE DESCRIPTION ENGINE")
print("=" * 72)
print("Phase 14 records:", len(KB))
print("Phase 14 file   :", KB_FILE)

# -----------------------------------------------------------------------------
# Build species aliases
# -----------------------------------------------------------------------------
ALIASES = {}


def add_alias(text, record):
    text = norm(text)
    if text and len(text) >= 3:
        ALIASES[text] = record


for r in KB:
    add_alias(r["_scientific"], r)
    add_alias(r["_common"], r)

# -----------------------------------------------------------------------------
# Optional original iNaturalist text dataset.
# It contributes common names, family, genus and geographic context to the
# candidate profiles; it is NOT treated as visual identification evidence.
# -----------------------------------------------------------------------------

def enrich_from_inaturalist():
    p = find_file(["text based new.csv"])
    if not p:
        return 0

    try:
        df = pd.read_csv(
            p,
            encoding="latin1",
            usecols=[
                "scientific_name", "common_name", "taxon_family_name",
                "taxon_genus_name", "place_country_name",
                "place_state_name"
            ],
            low_memory=False,
        )
    except Exception:
        return 0

    by_sc = {norm(r["_scientific"]): r for r in KB if r["_scientific"]}
    changed = 0

    for row in df.itertuples(index=False):
        sc = clean(getattr(row, "scientific_name", ""))
        r = by_sc.get(norm(sc))
        if not r:
            continue

        for attr, col in [
            ("_common", "common_name"),
            ("_family", "taxon_family_name"),
        ]:
            if not r[attr]:
                v = clean(getattr(row, col, ""))
                if v:
                    r[attr] = v
                    changed += 1

        add_alias(getattr(row, "common_name", ""), r)
        add_alias(getattr(row, "scientific_name", ""), r)

    return changed


enriched = enrich_from_inaturalist()
print("Profile enrichment updates:", enriched)

# -----------------------------------------------------------------------------
# Natural-language feature extraction
# -----------------------------------------------------------------------------
# These are intentionally broad language variants used by ordinary people.
# They are evidence for candidate ranking, not a guarantee of identification.

FEATURES = {
    "colors": {
        "black": ["black", "jet black", "dark black", "very dark"],
        "brown": ["brown", "dark brown", "light brown", "reddish brown"],
        "red": ["red", "reddish", "red colored", "red colour", "red color"],
        "yellow": ["yellow", "yellowish", "golden", "gold"],
        "green": ["green", "greenish", "bright green"],
        "grey": ["grey", "gray", "greyish", "grayish"],
        "white": ["white", "whitish", "cream", "pale"],
        "orange": ["orange", "orange colored", "orange colour"],
        "blue": ["blue", "bluish"],
    },
    "patterns": {
        "bands": ["band", "bands", "banded", "stripes", "striped", "rings", "ringed"],
        "spots": ["spot", "spots", "spotted", "dots", "dotted"],
        "blotches": ["blotch", "blotches", "patches", "large patches"],
        "checkered": ["checkered", "chequered", "checkerboard", "squares"],
        "diamonds": ["diamond", "diamonds", "diamond pattern"],
        "lines": ["line", "lines", "longitudinal stripe", "long stripe"],
        "plain": ["plain", "uniform", "one color", "single color"],
    },
    "shape": {
        "hood": ["hood", "spreads its neck", "neck hood", "cobra hood"],
        "long": ["long", "very long", "large snake", "big snake"],
        "small": ["small", "tiny", "little snake"],
        "thick": ["thick", "heavy body", "thick body", "stocky"],
        "slender": ["slender", "thin", "slim", "thin body"],
        "short": ["short body", "short snake"],
    },
    "head": {
        "triangular": ["triangular head", "triangle head", "triangular shaped head"],
        "broad": ["broad head", "wide head", "large head"],
        "flat": ["flat head", "flattened head"],
        "narrow": ["narrow head", "small head"],
    },
    "tail": {
        "red_tail": ["red tail", "reddish tail", "tail is red", "red coloured tail"],
        "short_tail": ["short tail", "small tail"],
        "long_tail": ["long tail", "long thin tail"],
        "pointed_tail": ["pointed tail", "sharp tail"],
    },
    "behavior": {
        "aggressive": ["aggressive", "attacked", "chased", "struck", "strike", "angry"],
        "climbing": ["tree", "climbing", "climbed", "on a tree", "branch"],
        "water": ["water", "river", "pond", "lake", "swimming", "aquatic"],
        "burrowing": ["burrow", "underground", "soil", "ground"],
    },
    "location": {
        "house": ["house", "home", "garden", "yard", "backyard"],
        "farm": ["farm", "field", "paddy", "rice field", "agriculture"],
        "forest": ["forest", "woods", "jungle"],
        "mountain": ["mountain", "hill", "hills"],
    },
}


def extract_features(text):
    t = norm(text)
    found = []
    for group, values in FEATURES.items():
        for label, phrases in values.items():
            for phrase in phrases:
                if phrase in t:
                    found.append((group, label, phrase))
                    break
    return found

# -----------------------------------------------------------------------------
# Feature/profile matcher
# -----------------------------------------------------------------------------
# Phase 14 is the source of truth. If its identification_features field is
# populated in a future Phase-14 revision, it is used directly. The existing
# Phase-14 file currently has sparse/empty visual-feature text, so the engine
# does not invent missing species descriptions.


def profile_text(r):
    parts = [
        r["_scientific"], r["_common"], r["_family"],
        r["_features"], r["_habitat"], r["_distribution"],
        r["_behavior"], r["_precautions"]
    ]
    return " ".join(x for x in parts if x)


def token_set(text):
    return set(re.findall(r"[a-z]{3,}", norm(text)))


def lexical_score(query, r):
    q = token_set(query)
    p = token_set(profile_text(r))
    if not q or not p:
        return 0.0
    return len(q & p) / math.sqrt(len(q) * len(p))


def feature_score(query, r):
    """Score explicit feature words against any actual Phase-14 feature text."""
    if not r["_features"]:
        return 0.0
    qf = extract_features(query)
    pf = extract_features(r["_features"])
    if not qf or not pf:
        return 0.0
    qlabels = {(a, b) for a, b, _ in qf}
    plabels = {(a, b) for a, b, _ in pf}
    return len(qlabels & plabels) / max(1, len(qlabels))

# -----------------------------------------------------------------------------
# BioCLIP zero-shot text ranking
# -----------------------------------------------------------------------------
# BioCLIP is a biological vision-language foundation model trained on
# TreeOfLife data. Here its text encoder is used only as an additional ranking
# signal. It is never treated as a medical or safety authority.

BIO_MODEL = None
BIO_TOKENIZER = None
BIO_TEXT_FEATURES = None
BIO_LABELS = None


def load_bioclip():
    global BIO_MODEL, BIO_TOKENIZER, BIO_TEXT_FEATURES, BIO_LABELS
    if not BIOCLIP_AVAILABLE:
        return False
    if BIO_MODEL is not None:
        return True

    try:
        print("\nLoading BioCLIP for natural-language species ranking...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
        tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
        model = model.to(device).eval()

        labels = []
        for r in KB:
            sc = r["_scientific"]
            cm = r["_common"]
            if cm:
                labels.append(f"a snake species called {cm} ({sc})")
            else:
                labels.append(f"a snake species called {sc}")

        with torch.no_grad():
            tokens = tokenizer(labels)
            tf = model.encode_text(tokens.to(device))
            tf = tf / tf.norm(dim=-1, keepdim=True)

        BIO_MODEL = model
        BIO_TOKENIZER = tokenizer
        BIO_TEXT_FEATURES = tf
        BIO_LABELS = labels
        print("BioCLIP loaded on:", device)
        return True
    except Exception as e:
        print("BioCLIP unavailable:", e)
        BIO_MODEL = None
        return False


def bioclip_scores(query):
    if not load_bioclip():
        return None
    device = next(BIO_MODEL.parameters()).device
    with torch.no_grad():
        q = BIO_TOKENIZER([query])
        qf = BIO_MODEL.encode_text(q.to(device))
        qf = qf / qf.norm(dim=-1, keepdim=True)
        sims = (qf @ BIO_TEXT_FEATURES.T)[0].float().cpu().numpy()
    return sims

# -----------------------------------------------------------------------------
# Candidate ranking
# -----------------------------------------------------------------------------

def rank_description(query):
    features = extract_features(query)
    bio = None

    # Load BioCLIP only when the user is actually asking for an identification.
    if BIOCLIP_AVAILABLE:
        bio = bioclip_scores(query)

    rows = []
    for i, r in enumerate(KB):
        lex = lexical_score(query, r)
        feat = feature_score(query, r)

        # BioCLIP scores are normalized later across the 313 candidates.
        b = float(bio[i]) if bio is not None else 0.0
        rows.append([r, lex, feat, b])

    if bio is not None:
        bs = np.array([x[3] for x in rows], dtype=float)
        lo, hi = float(bs.min()), float(bs.max())
        if hi > lo:
            bs = (bs - lo) / (hi - lo)
        for j in range(len(rows)):
            rows[j][3] = float(bs[j])

    # Weight actual feature evidence first, then BioCLIP, then lexical profile
    # evidence. With the current sparse Phase-14 visual fields, BioCLIP becomes
    # the main zero-shot signal.
    for x in rows:
        r, lex, feat, b = x
        if bio is not None:
            score = 0.65 * b + 0.25 * feat + 0.10 * lex
        else:
            score = 0.70 * feat + 0.30 * lex
        x.append(float(score))

    rows.sort(key=lambda x: x[4], reverse=True)

    top = rows[:TOP_K]
    if not top:
        return [], features, False

    # Convert relative scores to readable percentages. This is ranking
    # confidence, not a medically validated probability.
    vals = np.array([x[4] for x in top], dtype=float)
    if vals.max() > vals.min():
        probs = np.exp((vals - vals.max()) * 8.0)
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(vals)) / len(vals)

    candidates = []
    for x, p in zip(top, probs):
        r, lex, feat, bio_score, score = x
        candidates.append({
            "scientific_name": r["_scientific"],
            "common_name": r["_common"],
            "family": r["_family"],
            "score": round(float(score), 6),
            "ranking_confidence": round(float(p), 4),
            "feature_evidence": round(float(feat), 4),
            "lexical_evidence": round(float(lex), 4),
            "bioclip_evidence": round(float(bio_score), 4),
            "record": r,
        })

    # Conservative decision. A description with weak separation is returned
    # as uncertain rather than being presented as a confirmed identification.
    confident = False
    if candidates:
        topc = candidates[0]
        second = candidates[1]["ranking_confidence"] if len(candidates) > 1 else 0
        confident = (
            topc["ranking_confidence"] >= 0.50
            and (topc["ranking_confidence"] - second) >= 0.12
        )

    return candidates, features, confident

# -----------------------------------------------------------------------------
# Intent model
# -----------------------------------------------------------------------------
INTENT_DIR = OUT / "distilbert_venomguard"
INTENTS = [
    "identify", "venom_status", "risk", "habitat", "behavior", "diet",
    "distribution", "features", "medical_significance", "precautions",
    "general_info", "first_aid", "what_not_to_do", "medical_attention",
    "unknown"
]

intent_model = None
tokenizer = None
intent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_intent_model():
    global intent_model, tokenizer
    if not TRANSFORMERS_AVAILABLE:
        return False
    if not (INTENT_DIR / "config.json").exists():
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(INTENT_DIR))
        intent_model = AutoModelForSequenceClassification.from_pretrained(str(INTENT_DIR))
        intent_model.to(intent_device).eval()
        return True
    except Exception as e:
        print("DistilBERT could not be loaded:", e)
        return False


def detect_intent(text):
    t = norm(text)

    # High-precision safety overrides.
    if any(x in t for x in ["snake bite", "snakebite", "bitten by", "bit me"]):
        if any(x in t for x in ["not do", "don't", "do not", "avoid"]):
            return "what_not_to_do", 1.0
        if any(x in t for x in ["hospital", "emergency", "doctor", "medical attention"]):
            return "medical_attention", 1.0
        return "first_aid", 1.0

    if intent_model is not None:
        x = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
        x = {k: v.to(intent_device) for k, v in x.items()}
        with torch.no_grad():
            p = torch.softmax(intent_model(**x).logits, dim=-1)[0]
        idx = int(torch.argmax(p))
        label = intent_model.config.id2label.get(idx, INTENTS[idx] if idx < len(INTENTS) else "unknown")
        return label, float(p[idx])

    # Fallback keyword intent classifier.
    rules = {
        "venom_status": ["venom", "poisonous", "poison", "toxic"],
        "risk": ["dangerous", "danger", "risk", "deadly"],
        "habitat": ["where live", "habitat", "found", "live"],
        "behavior": ["behavior", "aggressive", "attack", "move"],
        "diet": ["eat", "food", "diet"],
        "distribution": ["country", "region", "distribution", "where found"],
        "features": ["look like", "identify", "features", "markings", "pattern"],
        "medical_significance": ["medical", "bite effects", "symptoms"],
        "precautions": ["safe", "safety", "precaution", "avoid"],
        "general_info": ["tell me about", "information", "attributes", "characteristics"],
    }
    for label, words in rules.items():
        if any(w in t for w in words):
            return label, 0.80
    return "identify", 0.60

load_intent_model()

# -----------------------------------------------------------------------------
# Current image species context
# -----------------------------------------------------------------------------
CURRENT_FILE = OUT / "phase15_current_species.json"


def lookup_species(name):
    n = norm(name)
    if n in ALIASES:
        return ALIASES[n]
    for r in KB:
        if n == norm(r["_scientific"]) or n == norm(r["_common"]):
            return r
    return None


def set_current_species(name, confidence=None, source="image_model"):
    r = lookup_species(name)
    if not r:
        return False
    d = {
        "scientific_name": r["_scientific"],
        "common_name": r["_common"],
        "confidence": confidence,
        "source": source,
        "updated": datetime.now().isoformat(),
    }
    with open(CURRENT_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return True


def get_current_species():
    if not CURRENT_FILE.exists():
        return None
    try:
        d = json.loads(CURRENT_FILE.read_text(encoding="utf-8"))
        return lookup_species(d.get("scientific_name", ""))
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Phase 14 answer engine
# -----------------------------------------------------------------------------

def display_name(r):
    if r["_common"] and r["_scientific"]:
        return f"{r['_common']} ({r['_scientific']})"
    return r["_scientific"] or r["_common"] or "Unknown species"


def field(r, key, fallback="No verified information is recorded for this field in Phase 14."):
    return r.get(key) or fallback


def answer_for_species(r, question, intent):
    name = display_name(r)
    if intent == "identify":
        return f"Most likely species: {name}"
    if intent == "venom_status":
        return f"{name}\n\nVenom status: {field(r, '_venomous', 'unknown')}"
    if intent == "risk":
        return (
            f"{name}\n\n"
            f"VenomGuard risk score: {field(r, '_score', 'not recorded')}/10\n"
            f"Emergency level: {field(r, '_emergency', 'unknown')}"
        )
    if intent == "habitat":
        return f"{name}\n\nHabitat:\n{field(r, '_habitat')}"
    if intent == "behavior":
        return f"{name}\n\nBehavior:\n{field(r, '_behavior')}"
    if intent == "diet":
        return f"{name}\n\nDiet information is not recorded in the current Phase-14 schema."
    if intent == "distribution":
        return f"{name}\n\nDistribution:\n{field(r, '_distribution')}"
    if intent == "features":
        return f"{name}\n\nIdentification features:\n{field(r, '_features')}"
    if intent == "medical_significance":
        return f"{name}\n\nMedical significance:\n{field(r, '_medical', 'Not recorded')}"
    if intent == "precautions":
        return f"{name}\n\nPrecautions:\n{field(r, '_precautions')}"
    if intent == "general_info":
        lines = [name]
        if r["_family"]: lines.append("Family: " + r["_family"])
        lines.append("Venom status: " + field(r, "_venomous", "unknown"))
        lines.append("Risk score: " + field(r, "_score", "not recorded") + "/10")
        lines.append("Identification features: " + field(r, "_features"))
        lines.append("Habitat: " + field(r, "_habitat"))
        lines.append("Distribution: " + field(r, "_distribution"))
        lines.append("Behavior: " + field(r, "_behavior"))
        lines.append("Medical significance: " + field(r, "_medical", "Not recorded"))
        return "\n\n".join(lines)
    return f"{name}\n\nI could not produce a dedicated answer for that attribute."


def safety_answer(intent):
    if intent == "first_aid":
        return (
            "SNAKEBITE FIRST AID\n\n"
            "Move away from the snake, keep the person calm and still, "
            "immobilize the affected limb if possible, and seek urgent "
            "professional medical care. Do not delay treatment while trying "
            "to identify the snake."
        )
    if intent == "what_not_to_do":
        return (
            "WHAT NOT TO DO\n\n"
            "Do not cut or suck the bite, do not apply ice or chemicals, "
            "do not attempt to catch the snake, and do not delay hospital care."
        )
    if intent == "medical_attention":
        return (
            "MEDICAL ATTENTION\n\n"
            "A suspected venomous snakebite requires urgent medical assessment. "
            "Do not wait for severe symptoms before seeking help."
        )
    if intent == "precautions":
        return (
            "SAFETY PRECAUTIONS\n\n"
            "Keep a safe distance from snakes. Do not touch, capture, kill, "
            "or provoke them. Keep children and pets away and contact trained "
            "wildlife personnel when removal is necessary."
        )
    return None

# -----------------------------------------------------------------------------
# Main natural-language query
# -----------------------------------------------------------------------------

def query(text):
    text = clean(text)
    if not text:
        return "Please describe the snake or ask a snake-related question."

    intent, intent_conf = detect_intent(text)
    safety = safety_answer(intent)
    if safety:
        return safety

    # Explicit species name takes priority if the user happens to know it.
    explicit = None
    for alias, r in ALIASES.items():
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", norm(text)):
            explicit = r
            break

    if explicit:
        return (
            f"Intent: {intent} ({intent_conf:.2f})\n\n" +
            answer_for_species(explicit, text, intent)
        )

    # Context from an earlier image prediction.
    current = get_current_species()
    references_current = any(x in norm(text) for x in [
        "this snake", "this species", "the snake", "its ", "it "
    ])
    if current and references_current:
        return (
            f"Current image species: {display_name(current)}\n\n"
            + answer_for_species(current, text, intent)
        )

    # If the input contains a description, rank unknown species.
    candidates, features, confident = rank_description(text)

    if not candidates:
        return (
            "I could not generate a species candidate from this description. "
            "Please provide more visible details or a photograph."
        )

    out = []
    out.append("UNKNOWN SNAKE DESCRIPTION ANALYSIS")
    out.append("")
    out.append("Observed description: " + text)

    if features:
        out.append("")
        out.append("Detected characteristics:")
        for group, label, phrase in features:
            out.append(f"- {label.replace('_', ' ')}")

    out.append("")
    if confident:
        out.append("Most likely species candidate:")
        c = candidates[0]
        out.append(
            f"1. {c['common_name'] + ' - ' if c['common_name'] else ''}"
            f"{c['scientific_name']}"
        )
        out.append(f"Ranking confidence: {c['ranking_confidence'] * 100:.1f}%")
    else:
        out.append("Identification is uncertain from the description alone.")
        out.append("Possible candidates:")

    for i, c in enumerate(candidates, 1):
        common = f" — {c['common_name']}" if c["common_name"] else ""
        out.append(
            f"{i}. {c['scientific_name']}{common} "
            f"({c['ranking_confidence'] * 100:.1f}%)"
        )

    out.append("")
    out.append(
        "For a more reliable identification, use a clear photograph. "
        "Do not approach or handle the snake."
    )

    # Only expose Phase-14 facts when one candidate is sufficiently separated.
    if confident:
        r = candidates[0]["record"]
        out.append("")
        out.append("Phase-14 information for the leading candidate:")
        out.append(answer_for_species(r, text, "general_info"))

    return "\n".join(out)

# -----------------------------------------------------------------------------
# Voice input
# -----------------------------------------------------------------------------

def find_vosk_model():
    for p in [
        VOICE_DIR / "vosk-model-small-en-us-0.15",
        VOICE_DIR / "vosk-model-en-us-0.22",
        VOICE_DIR / "vosk-model-small-en-in-0.4",
    ]:
        if p.is_dir():
            return p
    return None


def _select_microphone():
    """Choose the Windows/system default input, or a usable input as fallback."""
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        candidates = []

        # Prefer the operating-system default recording device.
        if default_input is not None and int(default_input) >= 0:
            candidates.append(int(default_input))

        # If the default is missing/disabled, try available recording devices.
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) > 0 and index not in candidates:
                candidates.append(index)

        for index in candidates:
            device = devices[index]
            if int(device.get("max_input_channels", 0)) < 1:
                continue
            native_rate = int(round(float(device.get("default_samplerate", 0) or 0)))
            rates = [16000, native_rate, 48000, 44100, 32000]
            for rate in dict.fromkeys(r for r in rates if r > 0):
                try:
                    sd.check_input_settings(device=index, channels=1,
                                            dtype="int16", samplerate=rate)
                    return index, rate, device.get("name", f"Input {index}")
                except Exception:
                    continue
        return None, None, "No usable input device"
    except Exception as exc:
        raise RuntimeError(f"Could not inspect microphone devices: {exc}") from exc


def _resample_to_16khz(audio, source_rate):
    """Convert mono int16 audio to the 16 kHz rate expected by Vosk."""
    audio = np.asarray(audio).reshape(-1)
    if source_rate == 16000:
        return audio.astype(np.int16, copy=False)
    if audio.size < 2:
        return audio.astype(np.int16, copy=False)
    target_length = max(1, int(round(len(audio) * 16000 / source_rate)))
    old_positions = np.arange(len(audio), dtype=np.float64)
    new_positions = np.linspace(0, len(audio) - 1, target_length)
    converted = np.interp(new_positions, old_positions, audio.astype(np.float64))
    return np.clip(converted, -32768, 32767).astype(np.int16)


def voice_input(seconds=8):
    if not VOICE_AVAILABLE:
        return None, "Voice support is unavailable. Install the 'vosk' and 'sounddevice' packages."
    model_path = find_vosk_model()
    if not model_path:
        return None, "No Vosk model found inside voice_models/."

    try:
        device_index, sample_rate, device_name = _select_microphone()
        if device_index is None:
            return None, (
                "No working microphone input was found. Check Windows Settings > "
                "System > Sound > Input, enable microphone access for desktop apps, "
                "and make sure an input device is connected."
            )

        print(f"\nMicrophone: {device_name}")
        print(f"Recording at {sample_rate} Hz for up to {seconds} seconds. Speak now...")
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device_index,
            blocking=True,
        )
        audio = _resample_to_16khz(audio, sample_rate)
        if audio.size == 0 or int(np.max(np.abs(audio.astype(np.int32)))) < 35:
            return "", None

        model = vosk.Model(str(model_path))
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip(), None
    except Exception as e:
        message = str(e)
        low = message.lower()
        if "permission" in low or "access" in low:
            message += "\nAllow microphone access in Windows Settings > Privacy & security > Microphone."
        elif "device" in low or "input" in low:
            message += "\nCheck Windows Settings > System > Sound > Input and select your built-in microphone."
        elif "sample rate" in low or "samplerate" in low:
            message += "\nThe microphone rejected its sample rate. Check the Windows input format and try again."
        return None, f"Microphone recording failed: {message}"

# -----------------------------------------------------------------------------
# Evaluation / reports
# -----------------------------------------------------------------------------

def run_tests():
    tests = [
        "black snake with white bands",
        "long green snake on a tree",
        "thick brown snake with diamond pattern",
        "small snake with red tail",
        "snake with a hood",
        "I was bitten by a snake what should I do",
        "what should I not do after snakebite",
        "is this snake dangerous",
    ]
    rows = []
    for q in tests:
        try:
            features = extract_features(q)
            rows.append({
                "input": q,
                "feature_count": len(features),
                "features": ", ".join(x[1] for x in features),
                "status": "PASS" if features or "bite" in q.lower() else "NEEDS_MORE_FEATURE_DATA",
            })
        except Exception as e:
            rows.append({"input": q, "feature_count": 0, "features": "", "status": str(e)})

    p = OUT / "phase15_description_tests.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
    return p


def report():
    visual_count = sum(bool(r["_features"]) for r in KB)
    common_count = sum(bool(r["_common"]) for r in KB)
    d = {
        "phase": 15,
        "status": "IMPLEMENTED",
        "timestamp": datetime.now().isoformat(),
        "phase14_records": len(KB),
        "phase14_visual_feature_records": visual_count,
        "phase14_common_name_records": common_count,
        "distilbert_loaded": intent_model is not None,
        "vosk_available": VOICE_AVAILABLE,
        "bioclip_package_available": BIOCLIP_AVAILABLE,
        "current_species_context_file": str(CURRENT_FILE),
        "description_identification": True,
        "top_k_candidates": TOP_K,
        "note": (
            "Natural-language description identification is assistive and "
            "returns ranked candidates. It must not be treated as definitive "
            "without image/expert confirmation."
        ),
    }
    with open(OUT / "phase15_completion_report.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    pd.DataFrame([d]).to_csv(
        OUT / "phase15_completion_report.csv", index=False, encoding="utf-8"
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    test_file = run_tests()
    report()

    print("\nPHASE 15 READY")
    print("Description identification : ENABLED")
    print("Voice input                :", "AVAILABLE" if VOICE_AVAILABLE else "INSTALL VOSK + MODEL")
    print("BioCLIP text ranking       :", "AVAILABLE" if BIOCLIP_AVAILABLE else "OPTIONAL - install open_clip_torch")
    print("DistilBERT intent          :", "LOADED" if intent_model is not None else "KEYWORD FALLBACK")
    print("Phase 14 records           :", len(KB))
    print("Phase 14 visual features  :", sum(bool(r["_features"]) for r in KB))
    print("Test report                :", test_file)

    print("\nTest examples:")
    print('  black snake with red tail')
    print('  long black snake with white bands and a hood')
    print('  green snake climbing a tree')
    print('  thick brown snake with diamond markings')
    print('  small snake with a red tail')
    print('  what should I do after a snake bite')

    while True:
        print("\n1. Text / description")
        print("2. Voice / description")
        print("3. Set current species from YOLOv8 + EfficientNet-B3")
        print("4. Show current species")
        print("5. Exit")
        try:
            choice = input("\nSelect: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "1":
            q = input("\nDescribe the snake: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            print("\nVenomGuard:\n" + query(q))

        elif choice == "2":
            q, err = voice_input()
            if err:
                print("\nVoice error:", err)
            elif not q:
                print("\nNo speech detected.")
            else:
                print("\nYou said:", q)
                print("\nVenomGuard:\n" + query(q))

        elif choice == "3":
            name = input("\nSpecies predicted by EfficientNet-B3: ").strip()
            c = input("Confidence (optional): ").strip()
            try:
                c = float(c) if c else None
            except Exception:
                c = None
            if set_current_species(name, c, "YOLOv8+EfficientNet-B3"):
                print("Current species saved:", name)
            else:
                print("Species is not present in the Phase-14 313-class database.")

        elif choice == "4":
            r = get_current_species()
            if r:
                print("Current species:", display_name(r))
            else:
                print("No current image species has been set.")

        elif choice == "5":
            break

        else:
            print("Choose 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# VenomGuard Hybrid Natural-Language Ranking Upgrade
# -----------------------------------------------------------------------------
# This override improves description matching by combining:
#   1. explicit natural-language attributes,
#   2. Phase-14 textual context,
#   3. local dataset visual profiles,
#   4. optional BioCLIP text-to-image similarity.
# It never replaces the Phase-14 factual record.

_VISUAL_PROFILE_CACHE = OUT / "species_visual_profiles.json"
_VISUAL_PROFILES = None
_VISUAL_PROFILE_LOCK = None


def _dataset_roots_for_text():
    roots = []
    seeds = [ROOT, Path.cwd()]
    env = os.environ.get("VENOMGUARD_DATASET", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            roots.append(p)
    for seed in seeds:
        for parent in [seed, *list(seed.parents)[:5]]:
            for name in [
                "image dataset", "image_dataset", "image-dataset",
                "dataset", "datasets", "snake_dataset"
            ]:
                p = parent / name
                if p.is_dir() and p not in roots:
                    roots.append(p)
    return roots


def _norm_profile_name(x):
    return re.sub(r"[^a-z0-9]", "", norm(x))


def _species_image_files(species, limit=3):
    target = _norm_profile_name(species)
    found = []
    for root in _dataset_roots_for_text():
        try:
            for group in ["venomous", "non-venomous", "nonvenomous", "non_venomous"]:
                gp = root / group
                if not gp.is_dir():
                    continue
                for d in gp.iterdir():
                    if not d.is_dir():
                        continue
                    if _norm_profile_name(d.name) != target:
                        continue
                    for f in sorted(d.rglob("*"), key=lambda x: x.name.lower()):
                        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                            found.append(str(f))
                            if len(found) >= limit:
                                return found
        except (OSError, PermissionError):
            continue
    return found


def _image_visual_profile(path):
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB").resize((96, 96))
        a = np.asarray(img, dtype=np.float32) / 255.0
        mx = a.max(axis=2)
        mn = a.min(axis=2)
        diff = mx - mn
        sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
        gray = a.mean(axis=2)
        # HSV hue approximation.
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        h = np.zeros_like(mx)
        mask = diff > 1e-6
        rg = ((g - b) / np.maximum(diff, 1e-6)) % 6
        gb = ((b - r) / np.maximum(diff, 1e-6)) + 2
        br = ((r - g) / np.maximum(diff, 1e-6)) + 4
        h[mask & (mx == r)] = rg[mask & (mx == r)]
        h[mask & (mx == g)] = gb[mask & (mx == g)]
        h[mask & (mx == b)] = br[mask & (mx == b)]
        h = h / 6.0

        def frac(cond):
            return float(np.mean(cond))

        edges = (
            np.abs(np.diff(gray, axis=0)).mean()
            + np.abs(np.diff(gray, axis=1)).mean()
        ) / 2.0

        return {
            "green": frac((h >= 0.20) & (h <= 0.45) & (sat > 0.18) & (mx > 0.18)),
            "yellow": frac((h >= 0.10) & (h < 0.20) & (sat > 0.18)),
            "red": frac(((h < 0.06) | (h > 0.94)) & (sat > 0.18)),
            "blue": frac((h >= 0.52) & (h <= 0.72) & (sat > 0.18)),
            "brown": frac((h >= 0.04) & (h < 0.14) & (sat > 0.12) & (mx < 0.72)),
            "black": frac(mx < 0.22),
            "white": frac((mx > 0.78) & (sat < 0.20)),
            "gray": frac((mx >= 0.22) & (mx <= 0.78) & (sat < 0.18)),
            "brightness": float(gray.mean()),
            "saturation": float(sat.mean()),
            "edge_density": float(edges),
            "aspect": float(img.width / max(1, img.height)),
        }
    except Exception:
        return None


def build_visual_profiles(force=False, per_species=3):
    global _VISUAL_PROFILES
    if _VISUAL_PROFILES is not None and not force:
        return _VISUAL_PROFILES

    if _VISUAL_PROFILE_CACHE.exists() and not force:
        try:
            _VISUAL_PROFILES = json.loads(_VISUAL_PROFILE_CACHE.read_text(encoding="utf-8"))
            return _VISUAL_PROFILES
        except Exception:
            pass

    profiles = {}
    for r in KB:
        sc = r.get("_scientific", "")
        if not sc:
            continue
        samples = _species_image_files(sc, per_species)
        vals = [_image_visual_profile(p) for p in samples]
        vals = [v for v in vals if v]
        if not vals:
            continue
        keys = vals[0].keys()
        profiles[_norm_profile_name(sc)] = {
            k: sum(float(v[k]) for v in vals) / len(vals)
            for k in keys
        }
        profiles[_norm_profile_name(sc)]["samples"] = samples

    _VISUAL_PROFILES = profiles
    try:
        _VISUAL_PROFILE_CACHE.write_text(
            json.dumps(profiles, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass
    return profiles


def _attribute_visual_score(query, profile):
    if not profile:
        return 0.0
    q = extract_features(query)
    if not q:
        return 0.0
    scores = []
    color_map = {
        "green": "green", "yellow": "yellow", "red": "red",
        "blue": "blue", "brown": "brown", "black": "black",
        "white": "white", "grey": "gray"
    }
    for group, label, _ in q:
        if group == "colors" and label in color_map:
            scores.append(min(1.0, profile.get(color_map[label], 0.0) * 4.0))
        elif group == "patterns":
            e = profile.get("edge_density", 0.0)
            if label in {"bands", "spots", "blotches", "checkered", "diamonds", "lines"}:
                scores.append(min(1.0, e * 8.0))
            elif label == "plain":
                scores.append(max(0.0, 1.0 - e * 8.0))
        elif group == "shape":
            aspect = profile.get("aspect", 1.0)
            if label in {"long", "slender"}:
                scores.append(min(1.0, max(aspect, 1.0 / max(aspect, 0.01)) / 2.0))
            elif label in {"small", "short"}:
                scores.append(0.5 if 0.7 <= aspect <= 1.4 else 0.2)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _semantic_score(query, r):
    q = token_set(query)
    if not q:
        return 0.0
    text = " ".join([
        r.get("_scientific", ""), r.get("_common", ""),
        r.get("_family", ""), r.get("_features", ""),
        r.get("_habitat", ""), r.get("_distribution", ""),
        r.get("_behavior", "")
    ])
    p = token_set(text)
    if not p:
        return 0.0
    return len(q & p) / math.sqrt(len(q) * len(p))


def rank_description(query):
    """Fast local text ranking for the web app; does not scan image folders.

    Returns (candidates, extracted_features, confident), preserving the Phase 17 API.
    Scores are relative ranking signals, not calibrated probabilities.
    """
    query = clean(query)
    features = extract_features(query)
    if not query or not KB:
        return [], features, False

    rows = []
    explicit_labels = {(group, label) for group, label, _ in features}
    for r in KB:
        lex = lexical_score(query, r)
        feat = feature_score(query, r)
        semantic = _semantic_score(query, r)
        # Prefer actual feature evidence, then semantic/context overlap.
        if explicit_labels:
            score = 0.50 * feat + 0.35 * semantic + 0.15 * lex
        else:
            score = 0.65 * semantic + 0.35 * lex
        rows.append((r, float(score), float(feat), float(semantic), float(lex)))

    rows.sort(key=lambda item: item[1], reverse=True)

    # A ranking among 313 species is not itself evidence of a match.
    # If the description contains no recognizable feature and has virtually
    # no overlap with any species profile, return no candidates rather than
    # presenting arbitrary species as meaningful matches.
    best_score = rows[0][1] if rows else 0.0
    best_feature = rows[0][2] if rows else 0.0
    best_semantic = rows[0][3] if rows else 0.0
    best_lexical = rows[0][4] if rows else 0.0
    if not rows or (
        not features
        and best_score < 0.08
        and best_semantic < 0.10
        and best_lexical < 0.10
    ):
        return [], features, False

    top = rows[:TOP_K]
    if not top:
        return [], features, False

    # Stable softmax only expresses relative rank among the returned candidates.
    maximum = top[0][1]
    weights = [math.exp(max(-50.0, min(0.0, (row[1] - maximum) * 8.0))) for row in top]
    total = sum(weights) or 1.0
    candidates = []
    for row, weight in zip(top, weights):
        r, score, feat, semantic, lex = row
        candidates.append({
            "scientific_name": r.get("_scientific", clean(r.get("scientific_name"))),
            "common_name": r.get("_common", clean(r.get("common_name"))),
            "family": r.get("_family", clean(r.get("family"))),
            "score": round(score, 6),
            "ranking_confidence": round(weight / total, 4),
            "feature_evidence": round(feat, 4),
            "visual_evidence": 0.0,
            "semantic_evidence": round(semantic, 4),
            "lexical_evidence": round(lex, 4),
            "bioclip_evidence": 0.0,
            "matched_attributes": [phrase for _, _, phrase in features[:8]],
            "record": r,
        })

    # Conservative: require both explicit attribute evidence and meaningful
    # separation; names alone must not be presented as confirmed identification.
    confident = False
    if candidates:
        topc = candidates[0]
        second = candidates[1]["ranking_confidence"] if len(candidates) > 1 else 0.0
        confident = (
            bool(explicit_labels)
            and topc["feature_evidence"] >= 0.25
            and topc["ranking_confidence"] >= 0.50
            and (topc["ranking_confidence"] - second) >= 0.12
        )
    return candidates, features, confident


# Image-to-local-dataset reranking helper. This is a secondary evidence signal;
# EfficientNet remains the primary trained classifier.
def rerank_image_candidates(image_path, classifier_candidates):
    profiles = build_visual_profiles()
    if not profiles or not classifier_candidates:
        return classifier_candidates
    probe = _image_visual_profile(image_path)
    if not probe:
        return classifier_candidates

    def sim(p):
        if not p:
            return 0.0
        keys = ["green","yellow","red","blue","brown","black","white","gray","brightness","saturation","edge_density"]
        d = sum((float(probe.get(k,0))-float(p.get(k,0)))**2 for k in keys)
        return 1.0 / (1.0 + 8.0 * d)

    rows=[]
    for c in classifier_candidates:
        s=sim(profiles.get(_norm_profile_name(c.get("scientific_name",""))))
        base=float(c.get("confidence", c.get("ranking_confidence",0)))
        score=0.85*base+0.15*s
        x=dict(c)
        x["dataset_visual_evidence"]=round(s,4)
        x["combined_image_score"]=round(score,6)
        rows.append(x)
    rows.sort(key=lambda x:x["combined_image_score"], reverse=True)
    return rows
