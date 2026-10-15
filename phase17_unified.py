# VenomGuard Phase 17 - Unified Image + Unknown Description Inference
# Python 3.10+
# Run: python .\phase17_unified.py
#
# IMAGE: YOLOv8 (optional crop) -> EfficientNet-B3 313-class classifier -> Phase 14
# If YOLO misses, the image is NOT rejected. EfficientNet classifies the full image.
# TEXT: Phase 15 description ranking -> Phase 14.
# Folder mode is resume-safe and skips unchanged images already in the CSV.

from __future__ import annotations
import csv, hashlib, importlib.util, json, re, sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
YOLO_MODEL = ROOT / "models" / "yolov8_snake" / "training" / "weights" / "best.pt"
EFFICIENTNET_MODEL = ROOT / "models" / "efficientnet_b3" / "best.pt"
CLASS_FILE = ROOT / "knowledge_base" / "class_to_idx.json"
KNOWLEDGE_FILE = ROOT / "knowledge_base" / "snake_knowledge.json"
OUT = ROOT / "phase17_outputs"
OUT.mkdir(parents=True, exist_ok=True)
RESULT_CSV = OUT / "phase17_image_results.csv"
REPORT_JSON = OUT / "phase17_unified_completion_report.json"
REPORT_CSV = OUT / "phase17_unified_completion_report.csv"
YOLO_CONF = 0.50
HIGH = 0.70
UNCERTAIN = 0.50
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO = None
EFF = None
YOLO_LOAD_ERROR = ""
CLASS_TO_IDX = {}
IDX_TO_CLASS = {}
KB = {}
P15 = None


def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", clean(x).lower().replace("_", " ").replace("-", " "))).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_classes():
    global CLASS_TO_IDX, IDX_TO_CLASS
    data = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) != 313: raise ValueError(f"Expected 313 classes, found {len(data) if isinstance(data,dict) else 'invalid'}")
    CLASS_TO_IDX = {str(k): int(v) for k, v in data.items()}
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def load_kb():
    global KB
    candidates = [
        ROOT / "phase14_final_outputs" / "snake_knowledge_final.json",
        ROOT / "knowledge_base" / "snake_knowledge_final.json",
        KNOWLEDGE_FILE,
        ROOT / "snake_knowledge.json"
    ]
    p = next((x for x in candidates if x.exists()), None)
    if not p: raise FileNotFoundError("Phase 14 snake_knowledge JSON not found")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("species","snakes","records","data","knowledge"):
            if isinstance(data.get(k), list): data = data[k]; break
    if not isinstance(data, list): raise ValueError("Unsupported Phase 14 JSON structure")
    KB = {}
    for r in data:
        if isinstance(r, dict):
            s = clean(r.get("scientific_name") or r.get("scientific") or r.get("species"))
            if s: KB[norm(s)] = r
    return p


def load_yolo():
    global YOLO, YOLO_LOAD_ERROR
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"YOLOv8 model not found: {YOLO_MODEL}")
    try:
        from ultralytics import YOLO as UYOLO
        YOLO = UYOLO(str(YOLO_MODEL))
        YOLO_LOAD_ERROR = ""
        print("[PASS] YOLOv8 loaded:", YOLO_MODEL)
    except Exception as e:
        YOLO = None
        YOLO_LOAD_ERROR = str(e)
        raise


def state_dict_from_checkpoint(c):
    if hasattr(c, "state_dict"): return c.state_dict()
    if isinstance(c, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            v = c.get(k)
            if isinstance(v, dict): return v
            if hasattr(v, "state_dict"): return v.state_dict()
        if c and all(isinstance(k, str) for k in c):
            if any(torch.is_tensor(v) for v in c.values()): return c
    raise ValueError("Could not find EfficientNet state_dict")


def load_eff():
    global EFF
    if not EFFICIENTNET_MODEL.exists(): raise FileNotFoundError(f"EfficientNet-B3 model not found: {EFFICIENTNET_MODEL}")
    ckpt = torch.load(str(EFFICIENTNET_MODEL), map_location="cpu", weights_only=False)
    state = state_dict_from_checkpoint(ckpt)
    state = {((k[7:] if k.startswith("module.") else k[6:] if k.startswith("model.") else k)):v for k,v in state.items()}
    m = models.efficientnet_b3(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 313)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing: raise ValueError(f"EfficientNet checkpoint incompatible; missing {len(missing)} keys")
    cmap = ckpt.get("class_to_idx") if isinstance(ckpt, dict) else None
    if isinstance(cmap, dict) and cmap != CLASS_TO_IDX: raise ValueError("Checkpoint class_to_idx does not match project class_to_idx.json")
    EFF = m.to(DEVICE).eval()
    print("[PASS] EfficientNet-B3 loaded: 313 classes, device", DEVICE)


TRANSFORM = transforms.Compose([
    transforms.Resize(320), transforms.CenterCrop(300), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def _probabilities(img):
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        return torch.softmax(EFF(x), 1)[0]


def classify(img, crop_img=None):
    """Classify with a small test-time ensemble to reduce crop/background errors.

    The same EfficientNet model is evaluated on the original view and its
    horizontal flip. When YOLO produced a valid crop, the crop and its flip
    are also evaluated. Probabilities are averaged before selecting Top-5.
    This keeps the class mapping unchanged while making the prediction less
    sensitive to one bad crop or one viewing direction.
    """
    views = [img.convert("RGB"), img.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    if crop_img is not None:
        c = crop_img.convert("RGB")
        views.extend([c, c.transpose(Image.Transpose.FLIP_LEFT_RIGHT)])

    probs = [_probabilities(v) for v in views]
    p = torch.stack(probs, dim=0).mean(dim=0)
    vals, ids = torch.topk(p, min(5, len(p)))
    return [{"scientific_name": IDX_TO_CLASS[int(i)], "confidence": float(v)} for v, i in zip(vals, ids)]


def yolo_best(path):
    # YOLO is optional at runtime. EfficientNet must still be able to
    # classify the full image when Ultralytics is unavailable.
    if YOLO is None:
        return None
    rs = YOLO.predict(source=str(path), conf=YOLO_CONF, imgsz=640, verbose=False)
    if not rs or rs[0].boxes is None or len(rs[0].boxes)==0: return None
    b = rs[0].boxes
    i = int(torch.argmax(b.conf).item())
    return {"confidence":float(b.conf[i].item()), "box":[float(x) for x in b.xyxy[i].tolist()]}


def crop(img, d):
    w,h=img.size; x1,y1,x2,y2=d["box"]
    x1=max(0,min(w-1,int(x1))); y1=max(0,min(h-1,int(y1)))
    x2=max(x1+1,min(w,int(x2))); y2=max(y1+1,min(h,int(y2)))
    c=img.crop((x1,y1,x2,y2))
    return c if c.width>=20 and c.height>=20 else img


def venom(r):
    if not r: return "UNRESOLVED"
    v=norm(r.get("venomous") or r.get("venom_status") or r.get("venomous_status"))
    if v in {"yes","true","1","venomous","poisonous"}: return "VENOMOUS"
    if v in {"no","false","0","non venomous","nonvenomous"}: return "NON-VENOMOUS"
    return "UNRESOLVED"


def recommendation(sc, conf):
    r=KB.get(norm(sc))
    if conf < UNCERTAIN:
        return {"risk":"UNKNOWN / LOW CONFIDENCE","decision":"DO NOT RELY ON SPECIES PREDICTION","actions":["Keep a safe distance.","Do not handle the snake.","If a bite occurred, seek urgent professional medical care."]}
    if not r:
        return {"risk":"UNKNOWN","decision":"DO NOT HANDLE","actions":["Keep a safe distance.","Do not handle the snake.","If a bite occurred, seek urgent professional medical care."]}
    v=venom(r)
    if v=="VENOMOUS":
        a=["Do not approach or handle the snake.","Maintain a safe distance.","If a bite occurred, seek urgent professional medical care immediately."]
        if clean(r.get("snakebite_first_aid")): a.append(clean(r["snakebite_first_aid"]))
        elif clean(r.get("medical_attention")): a.append(clean(r["medical_attention"]))
        return {"risk":"HIGH","decision":"URGENT CAUTION","actions":a}
    if v=="NON-VENOMOUS": return {"risk":"LOWER IDENTIFIED RISK","decision":"KEEP DISTANCE","actions":["Do not approach or handle the snake.","Allow the snake to move away."]}
    return {"risk":"UNRESOLVED","decision":"CAUTION","actions":["Do not approach or handle the snake.","Treat venom status as unresolved.","If a bite occurred, seek urgent professional medical care."]}


def image_infer(path):
    path = Path(path)
    img=Image.open(path).convert("RGB")
    d=yolo_best(path)
    if d:
        source = "YOLOv8 crop"
    elif YOLO is None:
        source = "EfficientNet full-image fallback (YOLO unavailable)"
    else:
        source = "EfficientNet full-image fallback (YOLO no detection)"
    used=crop(img,d) if d else None
    top=classify(img, used)
    p=top[0]; sc=p["scientific_name"]; conf=p["confidence"]
    r=KB.get(norm(sc)); rec=recommendation(sc,conf)
    return {"timestamp":datetime.now().isoformat(timespec="seconds"),"image":str(path.resolve()),"image_sha256":sha256(path),"width":img.width,"height":img.height,"yolo_status":"DETECTED" if d else "NO_DETECTION_FALLBACK","yolo_confidence":d["confidence"] if d else "","yolo_box":json.dumps(d["box"]) if d else "","classification_source":source,"scientific_name":sc,"common_name":clean(r.get("common_name")) if r else "","classification_confidence":conf,"confidence_percent":round(conf*100,2),"confidence_category":"HIGH CONFIDENCE" if conf>=HIGH else "UNCERTAIN" if conf>=UNCERTAIN else "LOW CONFIDENCE","venom_status":venom(r),"risk":rec["risk"],"decision":rec["decision"],"top5":json.dumps(top),"error":""}

FIELDS=["timestamp","image","image_sha256","width","height","yolo_status","yolo_confidence","yolo_box","classification_source","scientific_name","common_name","classification_confidence","confidence_percent","confidence_category","venom_status","risk","decision","top5","error"]


def read_rows():
    d={}
    if RESULT_CSV.exists():
        try:
            with RESULT_CSV.open("r",encoding="utf-8-sig",newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("image"): d[r["image"]]=r
        except Exception: pass
    return d


def write_rows(rows):
    with RESULT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for r in rows.values(): w.writerow({k:r.get(k,"") for k in FIELDS})


def one_image():
    p=Path(input("\nEnter image path: ").strip().strip('"'))
    if not p.exists(): print("[ERROR] Image not found"); return
    try:
        r=image_infer(p); rows=read_rows(); rows[str(p.resolve())]=r; write_rows(rows)
        print("\n"+"="*70); print("IMAGE RESULT"); print("="*70)
        print("YOLO:",r["yolo_status"],("confidence %.4f"%float(r["yolo_confidence"]) if r["yolo_confidence"] else ""))
        print("Classifier input:",r["classification_source"])
        print("Species:",r["scientific_name"]); print("Common:",r["common_name"] or "Not recorded")
        print("Confidence: %.2f%%"%r["confidence_percent"]); print("Category:",r["confidence_category"])
        print("Venom:",r["venom_status"]); print("Risk:",r["risk"]); print("Decision:",r["decision"])
        print("Top 5:")
        for x in json.loads(r["top5"]): print(" ",x["scientific_name"],"%.2f%%"%(x["confidence"]*100))
        print("="*70)
    except Exception as e: print("[FAILED]",e)


def folder():
    folder=Path(input("\nEnter image folder path: ").strip().strip('"'))
    if not folder.is_dir(): print("[ERROR] Folder not found"); return
    images=sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    rows=read_rows(); new=skip=fail=0
    for i,p in enumerate(images,1):
        key=str(p.resolve()); old=rows.get(key)
        if old and old.get("image_sha256")==sha256(p): skip+=1; continue
        print(f"[{i}/{len(images)}] {p}")
        try: rows[key]=image_infer(p); new+=1; write_rows(rows)
        except Exception as e: rows[key]={"image":key,"error":str(e)}; fail+=1; write_rows(rows); print("  FAILED:",e)
    report={"timestamp":datetime.now().isoformat(timespec="seconds"),"total":len(images),"processed_new":new,"skipped_existing":skip,"failed":fail,"results":str(RESULT_CSV),"status":"COMPLETE" if fail==0 else "COMPLETE_WITH_FAILURES"}
    REPORT_JSON.write_text(json.dumps(report,indent=2),encoding="utf-8")
    with REPORT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["field","value"]); [w.writerow(x) for x in report.items()]
    print("\nFOLDER COMPLETE | processed:",new,"skipped:",skip,"failed:",fail)


def load_phase15():
    global P15
    p=ROOT/"phase15_complete.py"
    if not p.exists(): return
    try:
        s=importlib.util.spec_from_file_location("vg_p15",str(p)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
        if hasattr(m,"rank_description"): P15=m; print("[PASS] Phase 15 description engine loaded")
    except Exception as e: print("[WARNING] Phase 15 unavailable:",e)


def description():
    if P15 is None: print("Phase 15 is unavailable."); return
    q=input("\nDescribe the unknown snake: ").strip()
    if not q: return
    try:
        candidates,features,confident=P15.rank_description(q)
        print("\n"+"="*70); print("UNKNOWN SNAKE DESCRIPTION"); print("="*70)
        print("Description:",q)
        if features: print("Characteristics:",", ".join(x[1].replace("_"," ") for x in features))
        print("Status:","LEADING CANDIDATE" if confident else "UNCERTAIN - RANKED CANDIDATES")
        for i,c in enumerate(candidates,1):
            common=(" — "+c["common_name"]) if c.get("common_name") else ""
            print(f"{i}. {c['scientific_name']}{common} ({c['ranking_confidence']*100:.2f}%)")
        print("Do not approach or handle the snake.")
    except Exception as e: print("[FAILED]",e)


def main():
    print("="*70); print("VENOMGUARD PHASE 17 - UNIFIED INFERENCE"); print("="*70)
    try:
        load_classes(); print("[PASS] 313 classes verified")
        kp=load_kb(); print("[PASS] Phase 14 loaded:",kp)
        load_yolo(); load_eff(); load_phase15()
    except Exception as e:
        print("\nPHASE 17 STARTUP FAILED:\n",e); sys.exit(1)
    print("\nMODELS READY | YOLO miss -> EfficientNet full-image fallback | threshold",YOLO_CONF)
    while True:
        print("\n1. Analyze one image\n2. Process image folder (resume-safe)\n3. Unknown snake text description\n4. Verify model hashes\n5. Exit")
        c=input("\nSelect: ").strip()
        if c=="1": one_image()
        elif c=="2": folder()
        elif c=="3": description()
        elif c=="4":
            print("\nYOLO SHA256:",sha256(YOLO_MODEL)); print("EfficientNet SHA256:",sha256(EFFICIENTNET_MODEL)); print("Device:",DEVICE)
        elif c=="5": break
        else: print("Select 1-5")
    print("\nPHASE 17 BUILD COMPLETE | STATUS: COMPLETE")

if __name__=="__main__": main()
