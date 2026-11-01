
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
from emergency_alerts import emergency_alerts_bp, send_venomous_detection_alert
app.register_blueprint(emergency_alerts_bp)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ENGINE = None
ENGINE_ERROR = None

def load_engine():
    global ENGINE, ENGINE_ERROR
    if ENGINE is not None:
        return ENGINE
    try:
        import phase17_unified as p17
        p17.load_classes()
        p17.load_kb()
        # EfficientNet is required for image classification.
        p17.load_eff()

        # YOLOv8 is mandatory. Never classify the full image when detection
        # is unavailable; EfficientNet is only allowed to receive a valid crop.
        p17.load_yolo()

        # Phase 15/BioCLIP is deliberately NOT loaded for image analysis.
        # It is loaded only when the user asks for text/voice candidate ranking.
        ENGINE = p17
        return ENGINE
    except Exception as e:
        ENGINE_ERROR = str(e)
        return None


# Return JSON for API errors instead of Flask HTML error pages.
@app.errorhandler(404)
def api_404(error):
    if request.path.startswith('/api/'):
        return jsonify({"ok": False, "error": f"API endpoint not found: {request.path}"}), 404
    return error

@app.errorhandler(500)
def api_500(error):
    if request.path.startswith('/api/'):
        return jsonify({"ok": False, "error": "VenomGuard server error. Check the terminal for the detailed Python error."}), 500
    return error

@app.get("/")
def home():
    return render_template("index.html")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Candidate sample-image index
# ---------------------------------------------------------------------------
# The UI must show REAL local dataset images, not a generic placeholder.
# Dataset discovery is intentionally broad because users may keep
# "image dataset" beside, above, or below the VenomGuard application folder.
_SAMPLE_ROOT_NAMES = [
    "image dataset", "image_dataset", "image-dataset",
    "images", "dataset", "datasets", "data",
    "phase7_split", "phase7_split_dataset", "split_dataset",
    "snake_dataset", "snake images", "snake_images"
]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_sample_cache = {}
_sample_roots_cache = None
_sample_folder_index = None
_PHASE14_RICH = None


def _norm_species(s):
    s = str(s).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    return "".join(ch for ch in s if ch.isalnum())


def _possible_project_roots():
    roots = []
    # app folder, current working directory, and several ancestors.
    seeds = [ROOT, Path.cwd()]
    for seed in list(seeds):
        for p in [seed, *list(seed.parents)[:5]]:
            if p.exists() and p.is_dir() and p not in roots:
                roots.append(p)
    return roots


def _candidate_roots():
    global _sample_roots_cache
    if _sample_roots_cache is not None:
        return _sample_roots_cache

    roots = []
    for parent in _possible_project_roots():
        for name in _SAMPLE_ROOT_NAMES:
            p = parent / name
            if p.is_dir() and p not in roots:
                roots.append(p)

        # Also accept folders whose names contain "image dataset".
        try:
            for p in parent.iterdir():
                if p.is_dir() and "image" in p.name.lower() and "dataset" in p.name.lower():
                    if p not in roots:
                        roots.append(p)
        except OSError:
            pass

    # Environment variable can explicitly point to the dataset.
    env_path = os.environ.get("VENOMGUARD_DATASET", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir() and p not in roots:
            roots.insert(0, p)

    _sample_roots_cache = roots
    return roots


def _build_sample_folder_index():
    """Build a folder-name -> first image index once, without reading images."""
    global _sample_folder_index
    if _sample_folder_index is not None:
        return _sample_folder_index

    idx = {}
    for root in _candidate_roots():
        # Expected structure: root/group/species/images.
        # os.walk is performed only once and only indexes directories/files.
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                if not filenames:
                    continue
                image_files = [
                    Path(dirpath) / f for f in filenames
                    if Path(f).suffix.lower() in _IMAGE_EXTS
                ]
                if not image_files:
                    continue

                dname = _norm_species(Path(dirpath).name)
                if dname and dname not in idx:
                    idx[dname] = sorted(image_files, key=lambda p: p.name.lower())[0]

                # Also index parent/group/species combinations where useful.
                parent = Path(dirpath).parent.name
                if parent:
                    combined = _norm_species(parent + " " + Path(dirpath).name)
                    if combined and combined not in idx:
                        idx[combined] = sorted(image_files, key=lambda p: p.name.lower())[0]
        except (OSError, PermissionError):
            continue

    _sample_folder_index = idx
    return idx


def find_sample_image(scientific_name):
    """Return a REAL local representative image for a species."""
    key = _norm_species(scientific_name)
    if key in _sample_cache:
        return _sample_cache[key]

    variants = {
        key,
        _norm_species(str(scientific_name).replace("_", " ")),
        _norm_species(str(scientific_name).replace("_", "-")),
    }

    # Fast exact lookup in indexed species folders.
    idx = _build_sample_folder_index()
    for v in variants:
        p = idx.get(v)
        if p and p.exists():
            _sample_cache[key] = p
            return p

    # Direct/fallback search through expected dataset structure.
    for root in _candidate_roots():
        for group in ["venomous", "non-venomous", "nonvenomous", "non_venomous"]:
            gp = root / group
            if not gp.is_dir():
                continue
            try:
                for d in gp.iterdir():
                    if d.is_dir() and _norm_species(d.name) in variants:
                        files = sorted(
                            (p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS),
                            key=lambda p: p.name.lower()
                        )
                        if files:
                            _sample_cache[key] = files[0]
                            return files[0]
            except (OSError, PermissionError):
                continue

    _sample_cache[key] = None
    return None


def dataset_image_status():
    roots = [str(p) for p in _candidate_roots()]
    idx = _build_sample_folder_index()
    return {
        "dataset_roots": roots,
        "indexed_species_folders": len(idx),
        "sample_cache_entries": len(_sample_cache),
    }



def _knowledge_candidates():
    """Return Phase 14 knowledge files in priority order."""
    return [
        ROOT / "phase14_final_outputs" / "snake_knowledge_final.json",
        ROOT / "knowledge_base" / "snake_knowledge_final.json",
        ROOT / "knowledge_base" / "snake_knowledge.json",
        ROOT / "snake_knowledge.json",
    ]


def _load_phase14_records():
    """Load the richest available Phase 14 record set once."""
    global _PHASE14_RICH
    if _PHASE14_RICH is not None:
        return _PHASE14_RICH

    records = {}
    for path in _knowledge_candidates():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("species", "snakes", "records", "data", "knowledge"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                name = row.get("scientific_name") or row.get("accepted_name")
                if name:
                    key = _norm_species(str(name))
                    # Prefer the first file only when it is the richest one.
                    old = records.get(key)
                    if old is None or len(row) > len(old):
                        records[key] = row
            if records:
                # Do not stop: a later final file may contain richer fields.
                continue
        except Exception:
            continue

    _PHASE14_RICH = records
    return records


def species_record(scientific_name):
    key = _norm_species(scientific_name)
    rich = _load_phase14_records()
    r = rich.get(key)

    if r is None:
        engine = load_engine()
        if engine is not None:
            target = _norm_species(scientific_name)
            for k, v in getattr(engine, "KB", {}).items():
                if _norm_species(k) == target:
                    r = v
                    break

    if not r:
        return None

    # Preserve the complete useful Phase 14 record rather than dropping fields.
    out = dict(r)
    out["scientific_name"] = r.get("scientific_name") or r.get("accepted_name") or scientific_name
    out["accepted_name"] = r.get("accepted_name", "")
    out["common_name"] = r.get("common_name", "")
    out["family"] = r.get("family", "")
    out["genus"] = r.get("genus", "")
    out["distribution"] = r.get("distribution", "")
    out["venomous"] = r.get("venomous", r.get("venom_status", ""))
    out["venom_category"] = r.get("venom_category", "")
    out["venomous_scale"] = r.get("venomous_scale", r.get("venom_scale", ""))
    out["who_risk_category"] = r.get("who_risk_category", r.get("who_medically_important", ""))
    out["emergency_level"] = r.get("emergency_level", "")
    out["medical_significance"] = r.get("medical_significance", "")
    out["who_medically_important"] = r.get("who_medically_important", "")
    out["identification_features"] = r.get("identification_features", "")
    out["habitat"] = r.get("habitat", "")
    out["behavior"] = r.get("behavior", "")
    out["diet"] = r.get("diet", "")
    out["precautions"] = r.get("precautions", "")
    out["what_not_to_do"] = r.get("what_not_to_do", "")
    out["snakebite_first_aid"] = r.get("snakebite_first_aid", "")
    out["medical_attention"] = r.get("medical_attention", "")
    out["verification_status"] = r.get("verification_status", "")
    out["conflict_status"] = r.get("conflict_status", "")
    out["conflicting_sources"] = r.get("conflicting_sources", "")
    out["evidence"] = r.get("evidence", "")
    out["source"] = r.get("source", "")
    out["source_url"] = r.get("source_url", "")
    out["last_checked"] = r.get("last_checked", "")
    return out


def candidate_payload(c, selected_conf=None):
    sc=c.get("scientific_name", "") if isinstance(c,dict) else str(c)
    rec=species_record(sc) or {"scientific_name":sc}
    img=find_sample_image(sc)
    if img:
        mime=mimetypes.guess_type(str(img))[0] or "image/jpeg"
        rec["sample_image_url"]="/api/species-image/"+sc.replace("/","_")
    else:
        rec["sample_image_url"]=""
        rec["sample_image_available"]=False
    rec["sample_image_available"]=bool(img)
    if isinstance(c,dict):
        rec["ranking_confidence"]=c.get("ranking_confidence", c.get("confidence", ""))
        rec["score"]=c.get("score", "")
        rec["feature_evidence"]=c.get("feature_evidence", "")
        rec["bioclip_evidence"]=c.get("bioclip_evidence", "")
    return rec



# ---------------------------------------------------------------------------
# Online knowledge fallback
# ---------------------------------------------------------------------------
# Online mode is deliberately opt-in. Core inference never calls these
# functions unless the browser explicitly asks for online enrichment.
_ONLINE_TIMEOUT = 10
_ONLINE_HEADERS = {
    "User-Agent": "VenomGuard-AI/1.0 (snake-information research tool)"
}

def _clean_text(value, limit=5000):
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]

def _http_json(url, params=None):
    if requests is None:
        raise RuntimeError("Online mode requires the 'requests' package.")
    r = requests.get(url, params=params or {}, headers=_ONLINE_HEADERS, timeout=_ONLINE_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _http_html(url):
    if requests is None:
        raise RuntimeError("Online mode requires the 'requests' package.")
    r = requests.get(url, headers=_ONLINE_HEADERS, timeout=_ONLINE_TIMEOUT)
    r.raise_for_status()
    return r.text

def _wikipedia_search(query):
    data = _http_json("https://en.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": query + " snake",
        "format": "json", "utf8": 1, "srlimit": 5
    })
    out=[]
    for item in data.get("query", {}).get("search", []):
        title=item.get("title", "")
        pageid=item.get("pageid")
        if not title or not pageid:
            continue
        try:
            summary=_http_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title, safe='')}")
        except Exception:
            summary={}
        out.append({
            "title": title,
            "url": summary.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"),
            "description": _clean_text(summary.get("extract") or BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" ") if BeautifulSoup else item.get("snippet", "")),
            "image": (summary.get("thumbnail") or {}).get("source", ""),
            "source": "Wikipedia"
        })
    return out

def _inaturalist_search(query):
    data=_http_json("https://api.inaturalist.org/v1/taxa", {
        "q": query, "rank": "species", "per_page": 8, "order_by": "observations_count", "order": "desc"
    })
    out=[]
    for t in data.get("results", []):
        if t.get("rank") not in (None, "species"):
            continue
        photo=(t.get("default_photo") or {}).get("medium_url") or (t.get("default_photo") or {}).get("square_url") or ""
        out.append({
            "title": t.get("preferred_common_name") or t.get("name") or "Unknown snake",
            "scientific_name": t.get("name", ""),
            "common_name": t.get("preferred_common_name", ""),
            "url": t.get("uri") or "https://www.inaturalist.org/observations",
            "description": _clean_text(t.get("wikipedia_summary") or ""),
            "image": photo,
            "source": "iNaturalist",
            "observations": t.get("observations_count"),
            "conservation_status": (t.get("conservation_status") or {}).get("status_name", "") if isinstance(t.get("conservation_status"), dict) else ""
        })
    return out

def _gbif_search(query):
    try:
        d=_http_json("https://api.gbif.org/v1/species/match", {"name": query})
    except Exception:
        return []
    if not d.get("scientificName"):
        return []
    return [{
        "title": d.get("scientificName"),
        "scientific_name": d.get("scientificName"),
        "common_name": "",
        "url": f"https://www.gbif.org/species/{d.get('usageKey')}" if d.get("usageKey") else "https://www.gbif.org/",
        "description": "Taxonomic match from GBIF: " + ", ".join(f"{k}={d.get(k)}" for k in ("rank", "family", "genus", "species") if d.get(k)),
        "image": "",
        "source": "GBIF",
        "match_confidence": d.get("confidence"),
        "status": d.get("status", "")
    }]

def _duckduckgo_search(query):
    if BeautifulSoup is None:
        return []
    try:
        html=_http_html("https://html.duckduckgo.com/html/" + "?q=" + requests.utils.quote(query))
    except Exception:
        return []
    soup=BeautifulSoup(html, "html.parser")
    out=[]
    allowed=("inaturalist.org", "reptile-database.reptarium.cz", "animaldiversity.org", "iucnredlist.org", "who.int", "wikipedia.org", "gbif.org")
    for item in soup.select(".result")[:12]:
        a=item.select_one(".result__a")
        sn=item.select_one(".result__snippet")
        if not a:
            continue
        href=a.get("href", "")
        text=_clean_text(sn.get_text(" ") if sn else "", 1000)
        title=_clean_text(a.get_text(" "), 200)
        if any(d in href.lower() for d in allowed):
            out.append({"title":title, "url":href, "description":text, "image":"", "source":"Web search"})
    return out

def _merge_online_results(query):
    wiki=_wikipedia_search(query)
    ina=_inaturalist_search(query)
    gbif=_gbif_search(query)
    web=_duckduckgo_search(query + " snake species identification")
    merged=[]
    seen=set()
    for row in wiki+ina+gbif+web:
        key=_norm_species(row.get("scientific_name") or row.get("title") or row.get("url"))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(row)
    return merged[:12]

def _online_profile_from_results(query, results):
    # Prefer an explicit taxonomic match, then iNaturalist/Wikipedia. Missing
    # facts remain blank so the UI never invents biological information.
    best=results[0] if results else {}
    scientific=best.get("scientific_name") or best.get("title") or query
    profile={
        "scientific_name": scientific,
        "accepted_name": scientific,
        "common_name": best.get("common_name", ""),
        "family": "", "genus": "", "distribution": "",
        "venomous": "", "venom_category": "", "venomous_scale": "",
        "who_medically_important": "", "who_risk_category": "",
        "emergency_level": "", "medical_significance": "",
        "identification_features": "", "habitat": "", "behavior": "",
        "diet": "", "precautions": "Keep a safe distance and do not handle an unidentified snake.",
        "what_not_to_do": "Do not handle, chase, or attempt to kill an unidentified snake.",
        "snakebite_first_aid": "If a bite has occurred, seek emergency medical care immediately and follow local medical guidance.",
        "medical_attention": "Urgent medical assessment is recommended after any suspected snakebite.",
        "verification_status": "Online sources — not part of the local Phase 14 dataset",
        "conflict_status": "", "conflicting_sources": "",
        "evidence": "", "source": "Online snake-information search",
        "source_url": best.get("url", ""),
        "last_checked": "Online lookup at request time",
        "online": True, "query": query,
        "online_sources": results,
        "sample_image_url": best.get("image", ""),
    }
    desc=[]
    for row in results[:5]:
        if row.get("description"):
            desc.append(f"{row.get('source')}: {row.get('description')}")
    profile["evidence"]="\n".join(desc)
    if best.get("observations") is not None:
        profile["evidence"] += f"\niNaturalist observations: {best.get('observations')}"
    if best.get("conservation_status"):
        profile["evidence"] += f"\nConservation status reported by iNaturalist: {best.get('conservation_status')}"
    return profile

@app.post("/api/online-search")
def online_search():
    data=request.get_json(silent=True) or {}
    query=_clean_text(data.get("query", ""), 500)
    if not query:
        return jsonify({"ok":False,"error":"Enter a snake description or species name for online search."}),400
    try:
        results=_merge_online_results(query)
        profile=_online_profile_from_results(query, results)
        return jsonify({"ok":True,"query":query,"found":bool(results),"profile":profile,"results":results,"message":"Online sources searched: Wikipedia, iNaturalist, GBIF and selected snake-information websites."})
    except Exception as e:
        return jsonify({"ok":False,"error":"Online search failed. Check your internet connection.","detail":str(e)}),502

@app.get("/api/species-image/<path:scientific_name>")
def species_image(scientific_name):
    p=find_sample_image(scientific_name)
    if not p or not p.exists():
        return jsonify({
            "ok": False,
            "error": "No local sample image was found for this species.",
            "species": scientific_name,
            "dataset": dataset_image_status(),
        }), 404
    return send_file(p)

@app.get("/api/dataset-status")
def dataset_status():
    return jsonify({"ok": True, **dataset_image_status()})


@app.get("/api/status")
def status():
    models = {
        "YOLOv8": ROOT / "models" / "yolov8_snake" / "training" / "weights" / "best.pt",
        "EfficientNet-B3": ROOT / "models" / "efficientnet_b3" / "best.pt",
        "Class mapping": ROOT / "knowledge_base" / "class_to_idx.json",
        "Phase 14 knowledge": ROOT / "knowledge_base" / "snake_knowledge.json",
        "Phase 15": ROOT / "phase15_complete.py",
        "Phase 17": ROOT / "phase17_unified.py",
    }
    ready = {k: v.exists() for k, v in models.items()}
    return jsonify({
        "ready": all(ready.values()) if ready else False,
        "components": ready,
        "local_only": True,
        "email_integration": True,
        "streamlit": False,
        "engine_loaded": ENGINE is not None,
        "engine_error": ENGINE_ERROR,
    })

@app.get("/api/species")
def species_search():
    q = request.args.get("q", "").strip().lower()
    engine = load_engine()
    if engine is None:
        return jsonify({"ok": False, "error": "Inference engine unavailable.", "detail": ENGINE_ERROR}), 500
    records=[]
    for key, record in getattr(engine,"KB",{}).items():
        if not q or q in key or q in str(record.get("common_name","")).lower():
            records.append(candidate_payload({"scientific_name":record.get("scientific_name",key)}))
        if len(records)>=30: break
    return jsonify({"ok":True,"count":len(records),"records":records})

@app.get("/api/species/<path:scientific_name>")
def species_details(scientific_name):
    r=species_record(scientific_name)
    if not r:
        return jsonify({"ok":False,"error":"Species not found in Phase 14."}),404
    return jsonify({"ok":True,"record":candidate_payload({"scientific_name":scientific_name})})

def _server_location_lookup():
    """Resolve approximate public-network location on the server, never in browser JS."""
    if requests is None:
        return "Location unavailable (IP geolocation service is not configured)."
    try:
        # With localhost use the server's public egress IP. For a remote deployment,
        # use the direct peer address and do not trust arbitrary forwarded headers.
        peer = (request.remote_addr or "").strip()
        import ipaddress
        ip_obj = ipaddress.ip_address(peer) if peer else None
        if not ip_obj or ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved:
            url = "https://ipapi.co/json/"
        else:
            url = f"https://ipapi.co/{peer}/json/"
        response = requests.get(url, headers={"User-Agent": "VenomGuard-AI/1.0"}, timeout=5)
        response.raise_for_status()
        info = response.json()
        parts = [info.get("city"), info.get("region"), info.get("country_name")]
        place = ", ".join(str(x).strip() for x in parts if x)
        lat, lon = info.get("latitude"), info.get("longitude")
        coords = f" (approx. {lat}, {lon})" if lat is not None and lon is not None else ""
        return (place + coords) if place else "Location unavailable (IP lookup returned no location)."
    except Exception as error:
        print("[LOCATION LOOKUP]", error)
        return "Location unavailable (server-side IP lookup failed)."

def _dataset_medical_summary(record):
    """Build emergency guidance only from fields stored in the Phase 14 dataset."""
    labels = [
        ("Medical significance", "medical_significance"),
        ("WHO medical importance", "who_medically_important"),
        ("WHO risk category", "who_risk_category"),
        ("Emergency level", "emergency_level"),
        ("Snakebite first aid", "snakebite_first_aid"),
        ("Medical attention", "medical_attention"),
        ("Precautions", "precautions"),
        ("What not to do", "what_not_to_do"),
    ]
    lines = [f"{label}: {record.get(key)}" for label, key in labels
             if record.get(key) not in (None, "", [], {})]
    return "\n".join(lines) if lines else "No medical guidance is recorded for this species in the local dataset. Seek emergency medical care immediately if a bite is suspected."

@app.post("/api/emergency/auto-send")
def automatic_venomous_alert():
    """Validate the species against the local knowledge base before sending."""
    data = request.get_json(silent=True) or {}
    scientific_name = str(data.get("scientific_name") or "").strip()[:180]
    if not scientific_name:
        return jsonify({"ok": False, "error": "Detected species name is missing."}), 400
    record = species_record(scientific_name)
    if not record:
        return jsonify({"ok": False, "error": "Species record could not be verified; no alert was sent."}), 404
    venom_status = str(record.get("venomous") or record.get("venom_status") or "").strip()
    accepted = {"venomous", "yes", "true", "highly venomous", "medically significant"}
    if venom_status.lower() not in accepted:
        return jsonify({"ok": True, "sent": False, "reason": "Species is not explicitly marked venomous."})
    # Location and medical guidance are obtained by the backend. The browser
    # does not request GPS or submit user-entered medical details.
    location = _server_location_lookup()
    medical_information = _dataset_medical_summary(record)
    result = send_venomous_detection_alert(
        species=str(record.get("scientific_name") or scientific_name),
        common_name=str(record.get("common_name") or data.get("common_name") or "Not provided"),
        venom_status=venom_status,
        confidence=str(data.get("confidence") or "")[:80],
        location=location[:500],
        medical_information=medical_information[:3000],
    )
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code

@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image was uploaded."}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Please choose an image."}), 400

    engine = load_engine()
    if engine is None:
        return jsonify({
            "ok": False,
            "error": "VenomGuard inference engine could not be loaded.",
            "detail": ENGINE_ERROR,
        }), 500

    suffix = Path(f.filename).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        temp_path = Path(tmp.name)

    try:
        result = engine.image_infer(temp_path)
        if result.get("yolo_status") == "NO_SNAKE_FOUND":
            # One consistent UI/API message for every negative detection.
            result["decision"] = "NO SNAKE DETECTED"
            result["message"] = "No snake detected"
            result["scientific_name"] = ""
            result["common_name"] = ""
            result["classification_source"] = "NONE"
            result["classification_confidence"] = 0
            result["confidence_percent"] = 0
            result["top5"] = "[]"
            result["species_details"] = {}
        elif result.get("yolo_status") == "SPECIES_UNCERTAIN":
            # Never let a weak classifier score leak a random species label.
            result["decision"] = "SPECIES UNCERTAIN"
            result["scientific_name"] = ""
            result["common_name"] = ""
            result["top5"] = "[]"
            result["species_details"] = {}
        try:
            top5_raw = json.loads(result.get("top5", "[]"))
            # EfficientNet-B3 already performs a multi-view ensemble in Phase 17.
            # Do not apply the older color-profile reranker here: it can reorder
            # visually similar species using coarse color statistics and can
            # make an otherwise correct classifier result worse.
            result["top5_data"] = top5_raw[:5]
            result["candidate_cards"] = [candidate_payload(c) for c in top5_raw[:5]]
        except Exception:
            result["candidate_cards"] = []
        result["original_filename"] = f.filename

        # Attach the complete Phase 14 record for the detected species.
        # This allows the image workflow to display the same detailed
        # species information available to text/voice detection.
        detected_name = result.get("scientific_name", "")
        if detected_name and result.get("yolo_status") != "SPECIES_UNCERTAIN":
            result["species_details"] = candidate_payload({
                "scientific_name": detected_name
            })
        else:
            result["species_details"] = {}

        # Browser preview image.
        data = temp_path.read_bytes()
        result["preview_data"] = "data:image/" + ("jpeg" if suffix in {".jpg", ".jpeg"} else suffix.lstrip(".")) + ";base64," + base64.b64encode(data).decode()

        # Convert JSON strings returned by Phase 17 into arrays where possible.
        try:
            result["top5_data"] = json.loads(result.get("top5", "[]"))
        except Exception:
            result["top5_data"] = []

        try:
            result["box_data"] = json.loads(result.get("yolo_box", "[]")) if result.get("yolo_box") else []
        except Exception:
            result["box_data"] = []

        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.post("/api/candidates")
def candidates():
    data=request.get_json(silent=True) or {}
    query=str(data.get("text","")).strip()
    if not query:
        return jsonify({"ok":False,"error":"Enter a description."}),400
    engine=load_engine()
    if engine is None:
        return jsonify({"ok":False,"error":"Inference engine unavailable.","detail":ENGINE_ERROR}),500
    try:
        # Load Phase 15 only for this operation. Image analysis never pays this cost.
        engine.load_phase15()
        p15=getattr(engine,"P15",None)
        if p15 is None:
            return jsonify({"ok":False,"error":"Phase 15 description engine is unavailable."}),500
        candidates,features,confident=p15.rank_description(query)
        out=[candidate_payload(c) for c in candidates[:5]]
        return jsonify({"ok":True,"confident":bool(confident),"features":[x[1] for x in features],"candidates":out})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.post("/api/description")
def description():
    q = (request.json or {}).get("text", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Enter a snake description."}), 400

    engine = load_engine()
    if engine is None:
        return jsonify({"ok": False, "error": "Inference engine unavailable.", "detail": ENGINE_ERROR}), 500

    p15 = getattr(engine, "P15", None)
    if p15 is None or not hasattr(p15, "rank_description"):
        return jsonify({
            "ok": True,
            "available": False,
            "message": "Phase 15 description engine is not available in this installation."
        })

    try:
        candidates, features, confident = p15.rank_description(q)
        return jsonify({
            "ok": True,
            "available": True,
            "confident": bool(confident),
            "features": [x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else str(x) for x in features],
            "candidates": candidates,
            "message": "Assistive ranking only. Do not handle an unidentified snake."
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



# ---------------------------------------------------------------------------
# Neural AI Voice Talk (Online Mode)
# Uses Microsoft Edge neural voices through edge-tts. No API key is required.
# Offline Mode keeps browser/installed voices as the local fallback.
# ---------------------------------------------------------------------------
EDGE_VOICE_PROFILES = {
    # English India — real neural voices
    "en-female-aashi": ("en-IN-AashiNeural", 1.0, 0),
    "en-female-aarti": ("en-IN-AartiNeural", 1.0, 0),
    "en-female-ananya": ("en-IN-AnanyaNeural", 1.0, 0),
    "en-female-kavya": ("en-IN-KavyaNeural", 1.0, 0),
    "en-female-neerja": ("en-IN-NeerjaNeural", 1.0, 0),
    "en-male-aarav": ("en-IN-AaravNeural", 1.0, 0),
    "en-male-arjun": ("en-IN-ArjunNeural", 1.0, 0),
    "en-male-kunal": ("en-IN-KunalNeural", 1.0, 0),
    "en-male-prabhat": ("en-IN-PrabhatNeural", 1.0, 0),
    "en-male-rehaan": ("en-IN-RehaanNeural", 1.0, 0),
    # Tamil India — Pallavi/Valluvar are the India voices. The other Tamil
    # voices are genuine neural voices from other Tamil locales.
    "ta-female-pallavi": ("ta-IN-PallaviNeural", 1.0, 0),
    "ta-male-valluvar": ("ta-IN-ValluvarNeural", 1.0, 0),
    "ta-female-kani": ("ta-MY-KaniNeural", 1.0, 0),
    "ta-male-surya": ("ta-MY-SuryaNeural", 1.0, 0),
    "ta-female-venba": ("ta-SG-VenbaNeural", 1.0, 0),
    "ta-male-anbu": ("ta-SG-AnbuNeural", 1.0, 0),
    "ta-female-saranya": ("ta-LK-SaranyaNeural", 1.0, 0),
    "ta-male-kumar": ("ta-LK-KumarNeural", 1.0, 0),
}


@app.get("/api/tts-status")
def tts_status():
    try:
        import edge_tts  # noqa: F401
        return jsonify({"ok": True, "online_neural": True, "profiles": list(EDGE_VOICE_PROFILES)})
    except Exception:
        return jsonify({"ok": True, "online_neural": False, "profiles": []})

@app.post("/api/tts")
def tts_generate():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    profile = str(data.get("profile") or "en-female-aarti")
    if not text:
        return jsonify({"ok": False, "error": "No text was provided."}), 400
    if len(text) > 5000:
        text = text[:5000]
    voice_cfg = EDGE_VOICE_PROFILES.get(profile)
    if voice_cfg is None:
        return jsonify({"ok": False, "error": "Unknown voice profile."}), 400
    try:
        import asyncio
        import edge_tts
        voice, rate, pitch = voice_cfg
        rate_pct = int(round((rate - 1.0) * 100))
        rate_str = f"{rate_pct:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
        buf = io.BytesIO()
        async def _make():
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    buf.write(chunk["data"])
        asyncio.run(_make())
        audio = buf.getvalue()
        if not audio:
            raise RuntimeError("Neural TTS returned no audio.")
        return send_file(io.BytesIO(audio), mimetype="audio/mpeg", download_name="venomguard_voice.mp3")
    except Exception as e:
        return jsonify({"ok": False, "error": f"Online neural TTS failed: {e}"}), 502


# ---------------------------------------------------------------------------
# Voice-to-text support.
# Browser SpeechRecognition is used directly when available; this avoids the
# old double-recording/fallback problem. Local Vosk remains available as the
# offline fallback when a model is installed in voice_models/en or /ta.
# ---------------------------------------------------------------------------
_VOSK_MODELS = {}

def _voice_model_dir(language):
    return ROOT / "voice_models" / ("ta" if str(language).lower().startswith("ta") else "en")

def _voice_model_available(language):
    d = _voice_model_dir(language)
    return d.exists() and any(d.iterdir())

@app.get("/api/voice-status")
def voice_status():
    return jsonify({
        "ok": True,
        "vosk": {
            "en": _voice_model_available("en"),
            "ta": _voice_model_available("ta")
        }
    })

@app.post("/api/voice-transcribe")
def voice_transcribe():
    audio = request.files.get("audio")
    language = (request.form.get("language") or "en").lower()
    if audio is None:
        return jsonify({"ok": False, "error": "No audio was recorded."}), 400

    model_dir = _voice_model_dir(language)
    if not _voice_model_available(language):
        return jsonify({
            "ok": False,
            "offline_available": False,
            "error": "No local Vosk model is installed for this language. Use browser voice recognition or add a Vosk model."
        }), 503

    wav_path = None
    try:
        import wave
        import json as _json
        import numpy as np
        from vosk import Model, KaldiRecognizer

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.save(tmp.name)
            wav_path = Path(tmp.name)

        # Vosk works best with mono 16-bit PCM at a model-friendly sample rate.
        # Resample the browser WAV to 16 kHz instead of passing the browser's
        # native 44.1/48 kHz rate directly to the recognizer.
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if channels != 1 or width != 2:
            return jsonify({"ok": False, "error": "Voice audio must be mono 16-bit PCM WAV."}), 400

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if rate != 16000 and len(samples):
            new_len = max(1, int(round(len(samples) * 16000 / rate)))
            old_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
            samples = np.interp(new_x, old_x, samples)
        pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

        key = str(model_dir.resolve())
        model = _VOSK_MODELS.get(key)
        if model is None:
            model = Model(str(model_dir))
            _VOSK_MODELS[key] = model

        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        for start in range(0, len(pcm), 8000):
            recognizer.AcceptWaveform(pcm[start:start + 8000])
        result = _json.loads(recognizer.FinalResult())
        text = (result.get("text") or "").strip()
        return jsonify({"ok": True, "offline_available": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            if wav_path:
                wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def _warm_engine():
    try:
        engine = load_engine()
        if engine is None:
            return
        print("[READY] VenomGuard core models warmed.")
        # Phase 15 is prepared in the background so the first text/voice query
        # does not pay the import/model initialization cost.
        try:
            engine.load_phase15()
            print("[READY] Phase 15 text/voice engine warmed.")
        except Exception as e:
            print("[WARN] Phase 15 warm-up skipped:", e)
    except Exception as e:
        print("[WARN] Model warm-up failed:", e)


if __name__ == "__main__":
    # 127.0.0.1 keeps this service on the user's computer.
    import threading
    print("\n" + "=" * 58, flush=True)
    print("              VENOMGUARD AI - LOCAL SERVER", flush=True)
    print("  Open this address in your browser:", flush=True)
    print("  http://127.0.0.1:5000", flush=True)
    print("=" * 58 + "\n", flush=True)
    threading.Thread(target=_warm_engine, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
