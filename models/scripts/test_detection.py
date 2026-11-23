import os
import sys
import argparse
import cv2
import torch
from ultralytics import YOLO

# COCO class IDs we care about
VEHICLE_CLASSES = {
    2: "car",
    3: "two-wheeler",
    5: "bus",
    7: "truck",
}

def main():
    parser = argparse.ArgumentParser(description="Step 2: Visual test for Two-Stage Vehicle + Plate Detection")
    parser.add_argument("video", help="Path to your CCTV video clip (e.g., data/clip1.mp4)")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame for fast testing (default: 5)")
    parser.add_argument("--max-frames", type=int, default=60, help="Stop after N processed frames (default: 60)")
    parser.add_argument("--sahi", action="store_true", help="Enable SAHI slicing (slower, use if cars are very tiny)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    # Load offline models from Step 1
    vehicle_model = YOLO("weights/yolo11s.pt")
    plate_model = YOLO("weights/indian_plate_best.pt")

    # Optional SAHI wrapper (only loaded if --sahi is passed)
    sahi_model = None
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="weights/yolo11s.pt",
            confidence_threshold=0.30,
            device=device,
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    os.makedirs("runs/debug_plates", exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path = "runs/debug_detect.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, fps / args.stride), (w, h))

    frame_idx = 0
    processed = 0
    total_vehicles = 0
    total_plates = 0

    while cap.isOpened() and processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        # --- STAGE 1A: Detect Vehicles ---
        vehicle_boxes = []
        if args.sahi:
            res = get_sliced_prediction(
                frame, sahi_model,
                slice_height=640, slice_width=640,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                verbose=0
            )
            for p in res.object_prediction_list:
                if p.category.id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, p.bbox.to_xyxy())
                    vehicle_boxes.append((x1, y1, x2, y2, VEHICLE_CLASSES[p.category.id], float(p.score.value)))
        else:
            # High-res 1280px pass (much sharper than default 640px, still fast)
            res = vehicle_model.predict(
                frame, imgsz=1280, conf=0.30,
                classes=list(VEHICLE_CLASSES.keys()),
                device=device, verbose=False
            )[0]
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                vehicle_boxes.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id], conf))

        # --- STAGE 1B: Zoom into each vehicle to find its plate ---
        vis = frame.copy()
        for v_i, (x1, y1, x2, y2, v_type, v_conf) in enumerate(vehicle_boxes):
            total_vehicles += 1
            # Draw GREEN box for vehicle
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{v_type} {v_conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crop lower 70% of vehicle (with small safety bounds)
            vh = y2 - y1
            roi_y1 = max(0, y1 + int(vh * 0.30))
            roi_y2 = min(h, y2)
            roi_x1 = max(0, x1)
            roi_x2 = min(w, x2)
            vehicle_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if vehicle_roi.size == 0:
                continue

            p_res = plate_model.predict(vehicle_roi, imgsz=640, conf=0.20, device=device, verbose=False)[0]
            if len(p_res.boxes) > 0:
                # Pick highest-confidence plate inside this vehicle
                best_box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                px1, py1, px2, py2 = map(int, best_box.xyxy[0])
                p_conf = float(best_box.conf[0])

                # Convert back to full-frame coordinates
                fx1, fy1 = roi_x1 + px1, roi_y1 + py1
                fx2, fy2 = roi_x1 + px2, roi_y1 + py2

                total_plates += 1
                # Draw RED box for number plate
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                cv2.putText(vis, f"plate {p_conf:.2f}", (fx1, max(20, fy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                # Save first 20 plate crops so we can inspect them visually
                if total_plates <= 20:
                    crop = frame[max(0, fy1 - 4):min(h, fy2 + 4), max(0, fx1 - 4):min(w, fx2 + 4)]
                    if crop.size > 0:
                        cv2.imwrite(f"runs/debug_plates/f{frame_idx:04d}_v{v_i}_c{p_conf:.2f}.jpg", crop)

        writer.write(vis)
        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone! Processed {processed} frames.")
    print(f"Total vehicle detections: {total_vehicles}")
    print(f"Total plate detections:   {total_plates}")
    print(f"Check video: {out_path}")
    print(f"Check crops: runs/debug_plates/")

if __name__ == "__main__":
    main()