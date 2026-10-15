from pathlib import Path
import cv2
import torch
from ultralytics import YOLOWorld

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

IMAGE_DIR = PROJECT_DIR / "yolo_dataset" / "images" / "train"
MODEL_PATH = PROJECT_DIR / "models" / "yolov8s-world.pt"

model = YOLOWorld(str(MODEL_PATH))

prompts = [
    ["snake"],
    ["reptile"],
    ["a snake"],
    ["snake animal"]
]

device = 0 if torch.cuda.is_available() else "cpu"

images = sorted(IMAGE_DIR.glob("*"))[:5]

print("Images tested:", len(images))
print("Device:", device)
print()

for image_path in images:
    print("=" * 70)
    print("IMAGE:", image_path.name)

    image = cv2.imread(str(image_path))

    if image is None:
        print("IMAGE READ FAILED")
        continue

    print("SIZE:", image.shape)

    for prompt in prompts:
        model.set_classes(prompt)

        results = model.predict(
            source=image,
            conf=0.01,
            iou=0.50,
            imgsz=640,
            device=device,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print(prompt, "-> NO DETECTION")
            continue

        confs = boxes.conf.cpu().tolist()

        print(
            prompt,
            "->",
            len(confs),
            "detections",
            "confidence:",
            [round(x, 3) for x in confs]
        )

    print()