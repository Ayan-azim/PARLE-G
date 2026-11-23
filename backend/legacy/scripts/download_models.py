
from pathlib import Path
import shutil
import urllib.request
from ultralytics import YOLO
import easyocr

MODELS_DIR = Path("models")
EASYOCR_DIR = MODELS_DIR / "easyocr"
MODELS_DIR.mkdir(exist_ok=True)
EASYOCR_DIR.mkdir(exist_ok=True)

# 1. Download YOLOv8 COCO weights
for name in ["yolov8n.pt", "yolov8s.pt"]:
    target = MODELS_DIR / name
    if not target.exists():
        print(f"Downloading {name}...")
        YOLO(name)  # downloads to current dir
        shutil.move(name, target)

# 2. Download pretrained YOLOv8 License Plate detector (Hugging Face)
lp_path = MODELS_DIR / "lp_detector.pt"
if not lp_path.exists():
    print("Downloading license plate detector...")
    url = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
    urllib.request.urlretrieve(url, lp_path)

# 3. Download EasyOCR English weights into models/easyocr/
print("Caching EasyOCR English weights...")
reader = easyocr.Reader(
    ["en"],
    gpu=False,  # set True if using CUDA
    model_storage_directory=str(EASYOCR_DIR),
    download_enabled=True,
)

print("All models downloaded and cached in models/!")