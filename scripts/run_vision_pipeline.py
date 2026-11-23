import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# Ensure project root is on sys.path so `vision.colour` imports cleanly
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
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_plate(crop):
    h, w = crop.shape[:2]
    if w < 160:
        scale = 160.0 / max(1, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def run_fast_plate_ocr(recognizer, img_path):
    out = recognizer.run(img_path, return_confidence=True)
    if isinstance(out, list) and len(out) > 0 and hasattr(out[0], "plate"):
        raw_text = str(out[0].plate)
        probs = [float(p) for p in out[0].char_probs]
    elif isinstance(out, tuple) and len(out) == 2:
        plates, confs = out
        raw_text = str(plates[0])
        probs = [float(p) for p in confs[0]]
    else:
        return "", []

    clean_chars, clean_probs = [], []
    for ch, p in zip(raw_text, probs):
        if ch == "_" or not ch.isalnum():
            continue
        clean_chars.append(ch.upper())
        clean_probs.append(round(p, 4))
    return "".join(clean_chars), clean_probs


def mask_uncertain(text, probs, threshold=0.55):
    return "".join(ch if p >= threshold else "?" for ch, p in zip(text, probs))


def build_consensus_plate(reads, conf_thresh=0.55):
    """
    Combine multi-frame reads by picking the dominant plate length among
    strong reads and taking confidence-weighted votes per character slot.
    Uncertain slots stay '?' (never invented).
    """
    valid = [r for r in reads if len(r["raw_text"]) >= 4 and r["mean_char_prob"] >= 0.40]
    if not valid:
        return "", []

    len_weights = defaultdict(float)
    for r in valid:
        len_weights[len(r["raw_text"])] += r["mean_char_prob"]
    target_len = max(len_weights, key=len_weights.get)

    same_len = [r for r in valid if len(r["raw_text"]) == target_len]
    consensus_chars = []
    consensus_probs = []

    for pos in range(target_len):
        slot_votes = defaultdict(float)
        total_w = 0.0
        for r in same_len:
            ch = r["raw_text"][pos]
            p = r["char_probs"][pos]
            slot_votes[ch] += p
            total_w += 1.0

        best_ch = max(slot_votes, key=slot_votes.get)
        avg_prob = slot_votes[best_ch] / max(1.0, total_w)
        consensus_chars.append(best_ch if avg_prob >= conf_thresh else "?")
        consensus_probs.append(round(avg_prob, 4))

    return "".join(consensus_chars), consensus_probs


def merge_duplicate_tracks(vehicle_records):
    """Merge split ByteTrack IDs that share the same high-confidence plate."""
    vehicle_records.sort(key=lambda v: v["total_frames_with_plate"], reverse=True)
    merged = []

    for rec in vehicle_records:
        plate = rec["consensus_plate"].replace("?", "")
        matched_parent = None

        if len(plate) >= 6:
            for parent in merged:
                p_plate = parent["consensus_plate"].replace("?", "")
                if len(p_plate) >= 6:
                    sim = SequenceMatcher(None, plate, p_plate).ratio()
                    if sim >= 0.88:
                        matched_parent = parent
                        break

        if matched_parent is not None:
            matched_parent["merged_track_ids"].append(rec["track_id"])
            matched_parent["total_frames_with_plate"] += rec["total_frames_with_plate"]
            matched_parent["frame_reads"].extend(rec["frame_reads"])
            matched_parent["frame_reads"].sort(key=lambda r: r["mean_char_prob"], reverse=True)
        else:
            rec["merged_track_ids"] = [rec["track_id"]]
            merged.append(rec)

    return merged


def main():
    parser = argparse.ArgumentParser(description="End-to-End Offline Vision Pipeline -> vision.json")
    parser.add_argument("video", help="Path to CCTV video (e.g., data/clips/clip1.mp4)")
    parser.add_argument("--out-dir", default="runs/clip1_new", help="Output directory")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame (default: 2)")
    parser.add_argument("--top-k", type=int, default=12, help="Top K sharpest plate crops per track")
    parser.add_argument("--min-frames", type=int, default=3, help="Drop noisy tracks with fewer plate frames")
    parser.add_argument("--conf-thresh", type=float, default=0.55, help="Char prob threshold below which -> '?'")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Loading offline models on {device.upper()}...")
    vehicle_model = YOLO("weights/yolo11s.pt")
    plate_model = YOLO("weights/indian_plate_best.pt")
    recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open {args.video}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    video_out = os.path.join(args.out_dir, "annotated.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, fps / args.stride), (w, h))

    tracks = defaultdict(lambda: {
        "types": Counter(),
        "colour_items": [],
        "plates": [],
    })

    print(f"[2/3] Tracking vehicles & extracting plates from {args.video}...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

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

            for (x1, y1, x2, y2), tid, cls_id in zip(boxes, track_ids, cls_ids):
                tid = int(tid)
                vx1, vy1 = max(0, int(x1)), max(0, int(y1))
                vx2, vy2 = min(w, int(x2)), min(h, int(y2))
                v_type = VEHICLE_CLASSES.get(int(cls_id), "car")

                v_crop = frame[vy1:vy2, vx1:vx2]
                if v_crop.size == 0:
                    continue

                tracks[tid]["types"][v_type] += 1

                # Plate detection inside lower 70% of vehicle
                vh = vy2 - vy1
                roi_y1 = vy1 + int(vh * 0.30)
                bumper_roi = frame[roi_y1:vy2, vx1:vx2]
                plate_box_in_crop = None

                if bumper_roi.size > 0:
                    p_res = plate_model.predict(bumper_roi, imgsz=640, conf=0.20, device=device, verbose=False)[0]
                    if len(p_res.boxes) > 0:
                        best_box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                        px1, py1, px2, py2 = map(int, best_box.xyxy[0])
                        p_conf = float(best_box.conf[0])

                        fx1, fy1 = int(vx1 + px1), int(roi_y1 + py1)
                        fx2, fy2 = int(vx1 + px2), int(roi_y1 + py2)
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

                # CIELAB Colour Estimation
                c_est = estimate_colour(
                    v_crop,
                    vehicle_type=v_type,
                    plate_box_in_crop=plate_box_in_crop,
                    use_plate_reference=True,
                )
                touches_border = (vx1 <= 5 or vy1 <= 5 or vx2 >= w - 5 or vy2 >= h - 5)
                penalty = 0.3 if touches_border else 1.0
                area_px = float((vx2 - vx1) * (vy2 - vy1))
                tracks[tid]["colour_items"].append((c_est, area_px, penalty))

                cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID {tid} | {c_est.value} {v_type}", (vx1, max(20, vy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    print("[3/3] Running multi-frame OCR & building vision.json...")
    vehicle_records = []
    for tid, data in tracks.items():
        if len(data["plates"]) < args.min_frames:
            continue

        top_plates = sorted(data["plates"], key=lambda x: x["sharpness"], reverse=True)[:args.top_k]
        frame_reads = []
        for item in top_plates:
            enh_text, enh_probs = run_fast_plate_ocr(recognizer, item["enh_path"])
            raw_text, raw_probs = run_fast_plate_ocr(recognizer, item["raw_path"])

            enh_mean = float(np.mean(enh_probs)) if enh_probs else 0.0
            raw_mean = float(np.mean(raw_probs)) if raw_probs else 0.0

            if enh_mean >= raw_mean:
                best_text, best_probs, src, best_img = enh_text, enh_probs, "enh", item["enh_path"]
            else:
                best_text, best_probs, src, best_img = raw_text, raw_probs, "raw", item["raw_path"]

            frame_reads.append({
                "frame": item["frame"],
                "plate_bbox": item["plate_bbox"],
                "plate_conf": item["plate_conf"],
                "sharpness": item["sharpness"],
                "source": src,
                "raw_text": best_text,
                "masked_text": mask_uncertain(best_text, best_probs, args.conf_thresh),
                "char_probs": best_probs,
                "mean_char_prob": round(max(enh_mean, raw_mean), 4),
                "image_path": best_img,
            })

        consensus_plate, consensus_probs = build_consensus_plate(frame_reads, args.conf_thresh)
        agg_colour = aggregate_estimates(data["colour_items"])
        type_top, type_count = data["types"].most_common(1)[0]
        type_total = sum(data["types"].values())

        vehicle_records.append({
            "track_id": int(tid),
            "total_frames_with_plate": len(data["plates"]),
            "consensus_plate": consensus_plate,
            "consensus_char_probs": consensus_probs,
            "colour": to_attribute_dict(agg_colour),
            "colour_distribution": agg_colour.distribution,
            "type": {
                "value": type_top,
                "confidence": round(type_count / max(1, type_total), 4),
                "method": "YOLO11s ByteTrack majority vote",
            },
            "frame_reads": frame_reads,
        })

    merged_vehicles = merge_duplicate_tracks(vehicle_records)

    vision_payload = {
        "video_path": args.video,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "vehicles": merged_vehicles,
    }

    out_json = os.path.join(args.out_dir, "vision.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vision_payload, f, indent=2, default=int)

    print(f"\nDone! Saved {len(merged_vehicles)} unique vehicle(s) to {out_json}")
    for v in merged_vehicles:
        ids_str = ",".join(map(str, v["merged_track_ids"]))
        print(
            f"  Track [{ids_str}] -> plate: {v['consensus_plate'] or '(unreadable)':12s} | "
            f"{v['colour']['value']} ({v['colour']['confidence']:.2f}) {v['type']['value']} | "
            f"{v['total_frames_with_plate']} plate frames"
        )


if __name__ == "__main__":
    main()