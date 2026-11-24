import os
import shutil
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV_WORK = os.path.join(ROOT, "weights", "openvino_scratch")
os.makedirs(OV_WORK, exist_ok=True)

# 1. Vehicle detector: rectangular 16:9 static FP32 export
veh_copy = os.path.join(OV_WORK, "yolo11s.pt")
shutil.copy2(os.path.join(ROOT, "weights", "yolo11s.pt"), veh_copy)
print("Exporting yolo11s.pt to OpenVINO FP32 (736x1280)...")
veh_ov = YOLO(veh_copy).export(format="openvino", imgsz=(736, 1280), dynamic=False)
print("Saved:", veh_ov)

# 2. Plate detector: dynamic FP32 export (handles varying bumper crop aspect ratios)
plate_copy = os.path.join(OV_WORK, "indian_plate_best.pt")
shutil.copy2(os.path.join(ROOT, "weights", "indian_plate_best.pt"), plate_copy)
print("Exporting indian_plate_best.pt to OpenVINO FP32 (dynamic)...")
plate_ov = YOLO(plate_copy).export(format="openvino", imgsz=640, dynamic=True)
print("Saved:", plate_ov)