import os
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

os.makedirs("weights", exist_ok=True)

# 1. Vehicle detector (YOLO11 Small — good balance of speed and accuracy)
print("Downloading vehicle detector...")
v_model = YOLO("yolo11s.pt")
if os.path.exists("yolo11s.pt"):
    shutil.move("yolo11s.pt", "weights/yolo11s.pt")

# 2. Dedicated Indian license plate detector
print("Downloading Indian plate detector...")
plate_path = hf_hub_download(
    repo_id="subash1652007/indian-plate-ocr",
    filename="yolo_best.pt",
    local_dir="weights"
)
os.replace(plate_path, "weights/indian_plate_best.pt")

print("Done! Saved models in weights/:")
print(os.listdir("weights"))