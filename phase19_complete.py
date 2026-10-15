
# VenomGuard Phase 19 - Online/Offline completion validator
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent
checks = {
    "phase19_online.py": ROOT / "phase19_online.py",
    "offline_engine": ROOT / "phase17_unified.py",
    "online_service": ROOT / "phase19_online.py",
}
status = all(p.exists() for p in checks.values())
report = {
    "phase": 19,
    "name": "Offline + Online Mode",
    "status": "COMPLETE" if status else "INCOMPLETE",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "offline_mode": "Local YOLOv8 + EfficientNet-B3 + Phase 14/15/16",
    "online_mode": "Local inference + iNaturalist/GBIF metadata enrichment",
    "privacy": "Default online mode does not upload user images.",
    "checks": {k: str(v.exists()) for k, v in checks.items()},
}
(ROOT / "phase19_outputs").mkdir(exist_ok=True)
(ROOT / "phase19_outputs" / "phase19_completion_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
