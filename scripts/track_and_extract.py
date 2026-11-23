import os
import sys
import json
import argparse
from collections import Counter, defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure project root (D:\parle g) is on sys.path so `vision.colour` imports cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.colour import estimate_colour, aggregate_estimates, to_attribute_dict

VEHICLE_CLASSES = {
    2: "car",
    3: "two-wheeler",
    5: "bus",
    7: "truck",
}


def sharpness_score(bgr_img):
    """Higher score = sharper plate crop."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_plate(crop):
    """Upscale tiny plates and boost contrast via CIELAB CLAHE."""
    h, w = crop.shape[:2]
    if w < 160:
        scale = 160.0 / max(1, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Step 3: ByteTrack + Plate Enhancement + CIELAB Colour")
    parser.add_argument("video", help="Path to CCTV clip (e.g., data/clips/clip1.mp4)")
    parser.add_argument("--out-dir", default="runs/step3_tracks", help="Output folder")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame (default: 2)")
    parser.add_argument("--top-k", type=int, default=12, help="Keep top K sharpest plate crops per track")
    parser.add_argument("--no-plate-ref", action="store_true", help="Disable white-plate exposure reference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Step 3 on: {device.upper()}")

    vehicle_model = YOLO("weights/yolo11s.pt")
    plate_model = YOLO("weights/indian_plate_best.pt")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open {args.video}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    video_out = os.path.join(args.out_dir, "tracked_debug.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, fps / args.stride), (w, h))

    tracks = defaultdict(lambda: {
        "types": Counter(),
        "colour_items": [],   # list of (ColourEstimate, box_area_px, penalty)
        "plates": [],
        "saved_colour_dbg": False,
    })

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        # --- STAGE 1A: Vehicle Detection + ByteTrack ---
        results = vehicle_model.track(
            frame,
            imgsz=1280,
            conf=0.35,
            iou=0.5,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            persist=True,
            device=device,
            verbose=False,
        )[0]

        vis = frame.copy()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy().astype(float)

            for (x1, y1, x2, y2), tid, cls_id, v_conf in zip(boxes, track_ids, cls_ids, confs):
                tid = int(tid)
                vx1, vy1 = max(0, int(x1)), max(0, int(y1))
                vx2, vy2 = min(w, int(x2)), min(h, int(y2))
                v_type = VEHICLE_CLASSES.get(int(cls_id), "car")

                v_crop = frame[vy1:vy2, vx1:vx2]
                if v_crop.size == 0:
                    continue

                tracks[tid]["types"][v_type] += 1

                # --- STAGE 1B: Plate Detection FIRST (so colour.py can mask & reference it) ---
                vh = vy2 - vy1
                roi_y1 = vy1 + int(vh * 0.30)
                roi_y2 = vy2
                bumper_roi = frame[roi_y1:roi_y2, vx1:vx2]

                plate_box_in_crop = None
                if bumper_roi.size > 0:
                    p_res = plate_model.predict(bumper_roi, imgsz=640, conf=0.20, device=device, verbose=False)[0]
                    if len(p_res.boxes) > 0:
                        best_box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                        px1, py1, px2, py2 = map(int, best_box.xyxy[0])
                        p_conf = float(best_box.conf[0])

                        fx1, fy1 = int(vx1 + px1), int(roi_y1 + py1)
                        fx2, fy2 = int(vx1 + px2), int(roi_y1 + py2)

                        # Plate coordinates relative to v_crop for colour.py
                        plate_box_in_crop = (fx1 - vx1, fy1 - vy1, fx2 - vx1, fy2 - vy1)

                        pad_x = max(2, int((fx2 - fx1) * 0.05))
                        pad_y = max(2, int((fy2 - fy1) * 0.08))
                        raw_crop = frame[max(0, fy1 - pad_y):min(h, fy2 + pad_y),
                                         max(0, fx1 - pad_x):min(w, fx2 + pad_x)]

                        if raw_crop.size > 0 and raw_crop.shape[1] >= 20 and raw_crop.shape[0] >= 8:
                            sharp = sharpness_score(raw_crop)
                            enh_crop = enhance_plate(raw_crop)

                            track_dir = os.path.join(args.out_dir, f"track_{tid}")
                            os.makedirs(track_dir, exist_ok=True)
                            raw_path = os.path.join(track_dir, f"f{frame_idx:04d}_raw.jpg")
                            enh_path = os.path.join(track_dir, f"f{frame_idx:04d}_enh.jpg")
                            cv2.imwrite(raw_path, raw_crop)
                            cv2.imwrite(enh_path, enh_crop)

                            tracks[tid]["plates"].append({
                                "frame": int(frame_idx),
                                "plate_bbox": [fx1, fy1, fx2, fy2],
                                "plate_conf": round(p_conf, 3),
                                "sharpness": round(sharp, 2),
                                "raw_path": raw_path,
                                "enh_path": enh_path,
                            })

                            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                            cv2.putText(vis, f"p:{p_conf:.2f} s:{int(sharp)}", (fx1, max(20, fy1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # --- STAGE 1C: CIELAB Colour Estimation ---
                dbg_path = None
                if plate_box_in_crop is not None and not tracks[tid]["saved_colour_dbg"]:
                    track_dir = os.path.join(args.out_dir, f"track_{tid}")
                    os.makedirs(track_dir, exist_ok=True)
                    dbg_path = os.path.join(track_dir, f"f{frame_idx:04d}_colour_debug.jpg")
                    tracks[tid]["saved_colour_dbg"] = True

                c_est = estimate_colour(
                    v_crop,
                    vehicle_type=v_type,
                    plate_box_in_crop=plate_box_in_crop,
                    use_plate_reference=(not args.no_plate_ref),
                    debug_path=dbg_path,
                )

                # Down-weight boxes cut off by the frame border
                touches_border = (vx1 <= 5 or vy1 <= 5 or vx2 >= w - 5 or vy2 >= h - 5)
                penalty = 0.3 if touches_border else 1.0
                area_px = float((vx2 - vx1) * (vy2 - vy1))
                tracks[tid]["colour_items"].append((c_est, area_px, penalty))

                # Draw tracked vehicle
                cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID {tid} | {c_est.value} ({c_est.confidence:.2f}) {v_type}",
                            (vx1, max(20, vy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    # Aggregate colour & keep top-K sharpest plates per track
    summary = {}
    for tid, data in tracks.items():
        sorted_plates = sorted(data["plates"], key=lambda x: x["sharpness"], reverse=True)[:args.top_k]
        if not sorted_plates:
            continue

        agg_colour = aggregate_estimates(data["colour_items"])
        colour_attr = to_attribute_dict(agg_colour)

        summary[str(tid)] = {
            "track_id": int(tid),
            "type": data["types"].most_common(1)[0][0],
            "colour": agg_colour.value,
            "colour_confidence": agg_colour.confidence,
            "colour_reliability": agg_colour.reliability,
            "colour_probs": agg_colour.distribution,
            "colour_attribute": colour_attr,
            "total_frames_with_plate": len(data["plates"]),
            "top_plates": sorted_plates,
        }

    meta_path = os.path.join(args.out_dir, "tracks_meta.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2, default=int)

    print(f"\nDone! Tracked {len(summary)} vehicle(s) with readable plate crops.")
    for tid, info in summary.items():
        top3 = sorted(info["colour_probs"].items(), key=lambda kv: -kv[1])[:3]
        probs_str = ", ".join(f"{k}:{v:.2f}" for k, v in top3)
        print(
            f"  Track {tid}: {info['colour']} {info['type']} "
            f"(conf={info['colour_confidence']:.2f}, rel={info['colour_reliability']:.2f} | {probs_str}) "
            f"| {info['total_frames_with_plate']} plate frames"
        )
    print(f"Debug video: {video_out}")
    print(f"Metadata:    {meta_path}")


if __name__ == "__main__":
    main()