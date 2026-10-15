# VenomGuard AI — Phase 18 + Phase 19

## What is finished

### Phase 18 — Localhost UI
- Attractive green/jungle-inspired Streamlit interface.
- Snake-detection logo included as `assets/venomguard_logo.svg`.
- Image upload and local inference.
- YOLOv8 snake detection.
- EfficientNet-B3 313-class classification.
- Confidence levels: High >=70%, Uncertain 50–69.99%, Low <50%.
- Phase-14 knowledge lookup.
- Safety recommendation and XAI evidence panels.
- Unknown snake text-description mode using the existing Phase 15 engine.
- Safety Center.
- System Status page.
- Local result remains on the computer.

### Phase 19 — Offline + Online
- **Offline mode:** local models + local knowledge. No image is uploaded.
- **Online mode:** local model inference is still used; only species-name metadata is sent to public iNaturalist/GBIF APIs.
- Online mode never uploads the user's image.
- Optional `VENOMGUARD_ONLINE_API_URL` can be configured for a user-owned online text endpoint.
- If the internet fails, the local analysis still works.

## Expected existing project structure

```text
Venomguard/
├── app.py
├── phase18_ui.py
├── phase19_online.py
├── phase17_unified.py
├── phase15_complete.py
├── models/
│   ├── yolov8_snake/
│   │   └── training/weights/best.pt
│   └── efficientnet_b3/
│       └── best.pt
├── knowledge_base/
│   ├── class_to_idx.json
│   └── snake_knowledge.json
└── assets/
    └── venomguard_logo.svg
```

## Install

Open PowerShell in the VenomGuard folder:

```powershell
python -m pip install -U -r requirements_phase18_19.txt
```

## Run

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

Or double-click:

```text
run_venomguard.bat
```

## Important model rule

Use the actual trained Colab YOLO checkpoint as:

```text
models\yolov8_snake\training\weights\best.pt
```

Use the trained EfficientNet-B3 checkpoint as:

```text
models\efficientnet_b3\best.pt
```

Do not replace these with generic YOLO-World/YOLO-E weights.

## Online privacy

The default online enrichment sends only a species name such as:

```text
Bungarus caeruleus
```

It does not send the uploaded image.

## Optional online AI endpoint

If you have your own HTTPS text AI API, set:

```powershell
$env:VENOMGUARD_ONLINE_API_URL="https://your-server.example/api/chat"
```

The endpoint should accept:

```json
{"message":"What should I do if this snake is dangerous?"}
```

and return:

```json
{"answer":"..."}
```

The app continues to work without this endpoint.

## Voice

Browser microphone capture is exposed when the installed Streamlit version supports `st.audio_input`.
For actual offline transcription, place a Vosk English model under:

```text
voice_models\
```

The existing Phase 15 voice pipeline remains available.

## Safety

VenomGuard is an assistive identification system. A model prediction is not a medical diagnosis. If a snakebite occurred, seek urgent professional medical care.
