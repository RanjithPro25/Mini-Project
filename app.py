
# VenomGuard AI - Phase 18 + 19
# Localhost application
#
# Run:
#   python -m pip install -r requirements_phase18_19.txt
#   streamlit run app.py
#
# Expected existing project files:
#   models/yolov8_snake/training/weights/best.pt
#   models/efficientnet_b3/best.pt
#   knowledge_base/class_to_idx.json
#   knowledge_base/snake_knowledge.json
#   phase17_unified.py
#
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image, ImageDraw

from phase18_ui import (
    confidence_label,
    field,
    make_xai,
    recommendation,
    normalize_venom,
)
from phase19_online import (
    internet_available,
    online_species_lookup,
    online_text_answer,
    mode_description,
)

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "assets" / "venomguard_logo.svg"

st.set_page_config(
    page_title="VenomGuard AI",
    page_icon=str(LOGO) if LOGO.exists() else "🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
          --vg-green:#20d46b;
          --vg-deep:#061d13;
          --vg-panel:rgba(5,28,19,.82);
          --vg-border:rgba(116,255,163,.20);
          --vg-text:#effff5;
          --vg-muted:#a9c8b5;
        }
        html, body, [class*="css"] {
          font-family: Inter, sans-serif;
        }
        .stApp {
          color:var(--vg-text);
          background:
            radial-gradient(circle at 8% 10%, rgba(49,180,86,.26), transparent 24%),
            radial-gradient(circle at 92% 20%, rgba(0,128,74,.24), transparent 28%),
            radial-gradient(circle at 50% 110%, rgba(32,212,107,.18), transparent 32%),
            linear-gradient(135deg,#03140d 0%,#062719 42%,#03160e 100%);
        }
        .stApp:before {
          content:"";
          position:fixed; inset:0; pointer-events:none; z-index:0;
          opacity:.18;
          background-image:
            radial-gradient(ellipse at 12% 22%, #65f08b 0 1px, transparent 2px),
            radial-gradient(ellipse at 82% 68%, #65f08b 0 1px, transparent 2px),
            radial-gradient(ellipse at 38% 82%, #65f08b 0 1px, transparent 2px);
          background-size:90px 120px,130px 150px,170px 190px;
        }
        .block-container {padding-top:1.2rem; max-width:1450px;}
        .vg-card {
          background:linear-gradient(145deg,rgba(11,53,34,.86),rgba(3,25,16,.82));
          border:1px solid var(--vg-border);
          border-radius:22px;
          padding:20px;
          box-shadow:0 18px 50px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
          margin-bottom:16px;
        }
        .vg-title {font-size:2.35rem;font-weight:800;letter-spacing:-1.5px;margin:0;}
        .vg-subtitle {color:var(--vg-muted);margin-top:4px;font-size:1rem;}
        .badge {
          display:inline-block;padding:6px 11px;border-radius:999px;
          font-size:.78rem;font-weight:800;letter-spacing:.5px;
          background:rgba(32,212,107,.12);border:1px solid rgba(32,212,107,.35);
        }
        .metric {
          background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07);
          border-radius:16px;padding:14px 16px;height:100%;
        }
        .metric-label {color:#9fbeaa;font-size:.78rem;text-transform:uppercase;letter-spacing:.7px;}
        .metric-value {font-size:1.25rem;font-weight:800;margin-top:4px;}
        .safe {border-left:4px solid #20d46b;}
        .warn {border-left:4px solid #f5c542;}
        .danger {border-left:4px solid #ff5e67;}
        .small {font-size:.84rem;color:#9fbeaa;}
        .hero {
          display:flex;align-items:center;gap:18px;
          padding:18px 22px;margin-bottom:18px;
          background:linear-gradient(100deg,rgba(8,54,31,.94),rgba(4,27,17,.80));
          border:1px solid rgba(93,255,143,.22);border-radius:26px;
        }
        .hero img {width:78px;height:78px;}
        .mode {
          border:1px solid rgba(32,212,107,.25);border-radius:16px;padding:10px 14px;
          background:rgba(0,0,0,.20);
        }
        section[data-testid="stSidebar"] {
          background:linear-gradient(180deg,#041b11,#062416 60%,#03160e);
          border-right:1px solid rgba(87,255,140,.13);
        }
        .stButton>button {
          border-radius:13px;border:1px solid rgba(104,255,153,.20);
          background:linear-gradient(135deg,#0b6b3a,#10964e);
          color:white;font-weight:700;min-height:44px;
          box-shadow:0 8px 24px rgba(0,0,0,.18);
        }
        .stButton>button:hover {border-color:#55ef8c;transform:translateY(-1px);}
        div[data-baseweb="tab-list"] {gap:8px;}
        button[data-baseweb="tab"] {border-radius:12px;padding:8px 14px;}
        .footer {
          color:#8eae9b;font-size:.78rem;text-align:center;padding:18px 0 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_local_engine():
    import importlib.util

    p = ROOT / "phase17_unified.py"
    if not p.exists():
        raise FileNotFoundError("phase17_unified.py is missing from the VenomGuard root folder.")

    spec = importlib.util.spec_from_file_location("vg_phase17_ui", str(p))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.load_classes()
    module.load_kb()
    module.load_yolo()
    module.load_eff()
    return module


def current_kb_record(engine, scientific_name: str) -> Dict[str, Any]:
    return engine.KB.get(engine.norm(scientific_name), {})


def annotate(image: Image.Image, box: Optional[List[float]]) -> Image.Image:
    out = image.copy().convert("RGB")
    if not box:
        return out
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=(55, 240, 120), width=max(3, out.width // 250))
    draw.rounded_rectangle((x1, max(0, y1 - 28), min(out.width, x1 + 150), y1),
                           radius=8, fill=(7, 74, 42))
    draw.text((x1 + 8, max(1, y1 - 24)), "SNAKE DETECTED", fill="white")
    return out


def run_image(engine, uploaded) -> Dict[str, Any]:
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        temp_path = Path(f.name)

    try:
        result = engine.image_infer(temp_path)
        try:
            result["top5_list"] = json.loads(result.get("top5", "[]"))
        except Exception:
            result["top5_list"] = []
        result["record"] = current_kb_record(engine, result.get("scientific_name", ""))
        box = None
        if result.get("yolo_box"):
            try:
                box = json.loads(result["yolo_box"])
            except Exception:
                box = None
        result["display_image"] = annotate(Image.open(temp_path), box)
        result["yolo_box_list"] = box
        result["xai"] = make_xai(
            result["record"],
            float(result.get("classification_confidence", 0)),
            result.get("classification_source", "local inference"),
        )
        result["recommendation_detail"] = recommendation(
            result["record"], float(result.get("classification_confidence", 0))
        )
        return result
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
logo_uri = ""
if LOGO.exists():
    logo_uri = "data:image/svg+xml;base64," + base64.b64encode(
        LOGO.read_text(encoding="utf-8").encode()
    ).decode()

st.markdown(
    f"""
    <div class="hero">
      <img src="{logo_uri}" alt="VenomGuard logo">
      <div style="flex:1">
        <div class="vg-title">VenomGuard <span style="color:#43ed7b">AI</span></div>
        <div class="vg-subtitle">Snake detection • Species identification • Safety guidance</div>
      </div>
      <div class="mode"><b>Phase 18–19</b><br><span class="small">Localhost AI platform</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    if logo_uri:
        st.markdown(
            f'<div style="text-align:center"><img src="{logo_uri}" width="112"></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### VenomGuard AI")
    page = st.radio(
        "Navigation",
        ["Image Analysis", "Text Description", "Safety Center", "System Status"],
        index=0,
    )
    st.divider()
    mode = st.radio(
        "AI Mode",
        ["Offline", "Online"],
        horizontal=True,
        help="Offline uses local models only. Online adds live public metadata; images are never uploaded.",
    )
    st.caption(mode_description(mode))
    st.divider()
    st.markdown("**Safety rule**")
    st.caption(
        "Never handle a wild snake. Identification is assistive and does not replace "
        "professional medical care."
    )

# ---------------------------------------------------------------------
# Online status
# ---------------------------------------------------------------------
online_ok = False
if mode == "Online":
    with st.spinner("Checking online services..."):
        online_ok = internet_available()

# ---------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------
if page == "Image Analysis":
    st.markdown("## Analyze a snake image")
    st.caption("YOLOv8 detects the snake → EfficientNet-B3 classifies the species → Phase 14/16 provides safety.")

    uploaded = st.file_uploader(
        "Upload a snake image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="The image is processed locally by the trained VenomGuard models.",
    )

    if uploaded is None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric"><div class="metric-label">Detection</div><div class="metric-value">YOLOv8</div><div class="small">Snake localization</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric"><div class="metric-label">Classification</div><div class="metric-value">EfficientNet-B3</div><div class="small">313 trained species</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric"><div class="metric-label">Knowledge</div><div class="metric-value">Phase 14</div><div class="small">Factual safety source</div></div>', unsafe_allow_html=True)
        st.info("Choose an image to begin. Offline mode keeps the complete analysis on this computer.")
    else:
        if st.button("Analyze Image", type="primary", use_container_width=True):
            try:
                with st.spinner("Running VenomGuard local inference..."):
                    engine = load_local_engine()
                    result = run_image(engine, uploaded)
                st.session_state["last_result"] = result
                st.session_state["last_mode"] = mode
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        result = st.session_state.get("last_result")
        if result:
            sc = result.get("scientific_name", "Unknown")
            common = result.get("common_name") or field(result.get("record", {}), "common_name", "common") or "Not recorded"
            conf = float(result.get("classification_confidence", 0))
            cat = confidence_label(conf)
            rec = result["recommendation_detail"]

            left, right = st.columns([1.18, 1], gap="large")
            with left:
                st.markdown('<div class="vg-card"><h3>Detection preview</h3>', unsafe_allow_html=True)
                st.image(result["display_image"], use_container_width=True)
                ystat = result.get("yolo_status", "UNKNOWN")
                yconf = result.get("yolo_confidence", "")
                st.markdown(
                    f'<span class="badge">YOLOv8: {ystat}</span> '
                    f'<span class="badge">Classifier: {result.get("classification_source","")}</span>',
                    unsafe_allow_html=True,
                )
                if yconf != "":
                    st.caption(f"YOLO detection confidence: {float(yconf)*100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown('<div class="vg-card">', unsafe_allow_html=True)
                st.markdown("### Predicted species")
                st.markdown(f"## {common}")
                st.caption(sc.replace("_", " "))
                st.progress(min(max(conf, 0.0), 1.0))
                st.markdown(f"**{conf*100:.2f}% — {cat}**")
                v = normalize_venom(field(result["record"], "venomous", "venom_status", "venomous_status"))
                st.markdown(
                    f'<div class="metric"><div class="metric-label">Venom status</div><div class="metric-value">{v}</div><div class="small">Phase 14 source of truth</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="metric"><div class="metric-label">Risk / decision</div><div class="metric-value">{rec["risk"]}</div><div class="small">{rec["decision"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            tabs = st.tabs(["Result Details", "Safety Guidance", "AI Explanation", "Species Information", "Online Enrichment"])

            with tabs[0]:
                a,b,c,d = st.columns(4)
                a.metric("Species", sc.replace("_", " "))
                b.metric("Confidence", f"{conf*100:.2f}%")
                c.metric("YOLO", "Detected" if result.get("yolo_status") == "DETECTED" else "Fallback")
                d.metric("Input", result.get("classification_source", "local"))
                st.markdown("#### Top predictions")
                for i, item in enumerate(result.get("top5_list", [])[:5], 1):
                    name = item.get("scientific_name", "").replace("_", " ")
                    score = float(item.get("confidence", 0)) * 100
                    st.write(f"{i}. **{name}** — {score:.2f}%")

            with tabs[1]:
                cls = "danger" if rec["risk"] == "HIGH" else "warn" if rec["risk"] == "UNRESOLVED" else "safe"
                st.markdown(f'<div class="vg-card {cls}"><h3>{rec["decision"]}</h3>', unsafe_allow_html=True)
                for item in rec["items"]:
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            with tabs[2]:
                st.markdown('<div class="vg-card">', unsafe_allow_html=True)
                st.markdown("### Why this result?")
                for item in result["xai"]:
                    st.markdown(f"✓ {item}")
                st.caption(
                    "XAI describes available evidence; it does not prove biological identity."
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with tabs[3]:
                record = result.get("record", {})
                fields = [
                    ("Family", field(record, "family", "family_name")),
                    ("Genus", field(record, "genus")),
                    ("Distribution", field(record, "distribution", "origin_country")),
                    ("Habitat", field(record, "habitat")),
                    ("Behavior", field(record, "behavior")),
                    ("Diet", field(record, "diet")),
                ]
                for label, value in fields:
                    if value:
                        st.markdown(f"**{label}:** {value}")

            with tabs[4]:
                if mode != "Online":
                    st.info("Switch AI Mode to Online to load live public metadata.")
                elif not online_ok:
                    st.warning("Online services are unavailable. Local analysis is still valid.")
                else:
                    with st.spinner("Loading iNaturalist / GBIF metadata..."):
                        online = online_species_lookup(sc.replace("_", " "))
                    if online.get("ok"):
                        st.success(online.get("message", "Online metadata loaded."))
                        if online.get("photo_url"):
                            st.image(online["photo_url"], width=240)
                        st.write("**Online common name:**", online.get("common_name") or "Not recorded")
                        st.write("**Observations:**", online.get("observations_count", "Not available"))
                        if online.get("wikipedia_url"):
                            st.markdown(f"[Open species reference]({online['wikipedia_url']})")
                        if online.get("gbif_key"):
                            st.caption(f"GBIF taxon key: {online['gbif_key']}")
                        st.caption("Online data supplements the local database; it does not override local safety facts.")
                    else:
                        st.warning(online.get("message", "No online metadata available."))

# ---------------------------------------------------------------------
# Text description
# ---------------------------------------------------------------------
elif page == "Text Description":
    st.markdown("## Unknown snake description")
    st.caption("Describe visible characteristics. Results are ranked candidates, not guaranteed species identification.")

    q = st.text_area(
        "Describe the snake",
        placeholder="Example: long green snake with a pointed head, large eye, climbing a tree",
        height=130,
    )

    if st.button("Identify from Description", type="primary"):
        if not q.strip():
            st.warning("Enter a description first.")
        else:
            try:
                import importlib.util
                p = ROOT / "phase15_complete.py"
                if not p.exists():
                    raise FileNotFoundError("phase15_complete.py was not found.")
                spec = importlib.util.spec_from_file_location("vg_p15_app", str(p))
                p15 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(p15)
                with st.spinner("Ranking candidates from the 313-species knowledge base..."):
                    candidates, features, confident = p15.rank_description(q.strip())

                if confident:
                    st.success("A leading candidate was found, but photograph-based confirmation is still recommended.")
                else:
                    st.warning("Identification is uncertain. Do not handle the snake.")

                if features:
                    st.write("**Observed characteristics:**", ", ".join(x[1].replace("_", " ") for x in features))

                for i, c in enumerate(candidates[:5], 1):
                    name = c.get("scientific_name", "").replace("_", " ")
                    common = c.get("common_name") or "Common name not recorded"
                    score = float(c.get("ranking_confidence", 0)) * 100
                    with st.container(border=True):
                        st.markdown(f"### {i}. {common}")
                        st.caption(name)
                        st.progress(min(max(score / 100, 0), 1))
                        st.write(f"Ranking confidence: **{score:.2f}%**")

                if mode == "Online" and online_ok and candidates:
                    best = candidates[0].get("scientific_name", "")
                    with st.expander("Online enrichment for leading candidate"):
                        online = online_species_lookup(best.replace("_", " "))
                        if online.get("ok"):
                            st.write("Online common name:", online.get("common_name") or "Not recorded")
                            if online.get("wikipedia_url"):
                                st.markdown(f"[Open species reference]({online['wikipedia_url']})")
                        else:
                            st.write(online.get("message", "No online metadata."))
            except Exception as e:
                st.error(f"Description analysis failed: {e}")

    st.divider()
    st.markdown("### Voice input")
    st.caption("Optional: install Vosk + a local Vosk English model for offline speech-to-text.")
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Record a description")
        if audio:
            st.audio(audio)
            st.info("Audio recording is captured. Connect a Vosk model under voice_models/ to enable offline transcription.")
    else:
        st.caption("Your installed Streamlit version does not expose browser microphone capture. Text mode remains available.")

# ---------------------------------------------------------------------
# Safety Center
# ---------------------------------------------------------------------
elif page == "Safety Center":
    st.markdown("## Safety Center")
    st.markdown(
        """
        <div class="vg-card danger">
        <h3>Possible snakebite</h3>
        <p>Move away from the snake, keep the affected limb still, remove tight items,
        and seek urgent professional medical care.</p>
        <p><b>Do not</b> cut the wound, suck venom, apply a tourniquet, or rely on
        traditional remedies instead of medical care.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "VenomGuard is an assistive identification system. It cannot diagnose a bite "
        "or replace a clinician, poison center, or emergency service."
    )

# ---------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------
else:
    st.markdown("## System Status")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Project", "VenomGuard AI")
    with cols[1]:
        st.metric("Species classes", "313")
    with cols[2]:
        st.metric("AI mode", mode)
    with cols[3]:
        st.metric("Online", "Connected" if online_ok else "Off")

    st.markdown('<div class="vg-card">', unsafe_allow_html=True)
    st.markdown("### Required local components")
    checks = [
        ("YOLOv8 snake detector", ROOT / "models/yolov8_snake/training/weights/best.pt"),
        ("EfficientNet-B3", ROOT / "models/efficientnet_b3/best.pt"),
        ("Class mapping", ROOT / "knowledge_base/class_to_idx.json"),
        ("Phase 14 knowledge", ROOT / "knowledge_base/snake_knowledge.json"),
        ("Phase 17 engine", ROOT / "phase17_unified.py"),
        ("Phase 15 engine", ROOT / "phase15_complete.py"),
    ]
    for label, path in checks:
        if path.exists():
            st.success(f"{label}: FOUND")
        else:
            st.error(f"{label}: MISSING — {path}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="vg-card">
        <h3>Privacy architecture</h3>
        <ul>
          <li><b>Offline:</b> image inference and knowledge lookup stay on the local machine.</li>
          <li><b>Online:</b> live species metadata may be retrieved from iNaturalist/GBIF.</li>
          <li><b>Images:</b> the default online enrichment path never uploads the user's image.</li>
          <li><b>Safety:</b> local Phase-14 knowledge remains the source of truth for venom/safety fields.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">VenomGuard AI • Phase 18–19 • 313 Snake Species • Offline & Online • Localhost</div>',
    unsafe_allow_html=True,
)
