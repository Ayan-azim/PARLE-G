import os
import sys
import json
import time
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


def plate_belongs_to_vehicle(plate_box, owner_idx, all_vehicle_boxes, frame_w, frame_h):
    """
    Prevents a truck/bus bounding box from stealing an adjacent car's plate,
    and rejects partial plates clipped against the frame border.
    """
    fx1, fy1, fx2, fy2 = plate_box
    if fx1 <= 4 or fy1 <= 4 or fx2 >= frame_w - 4 or fy2 >= frame_h - 4:
        return False

    cx, cy = 0.5 * (fx1 + fx2), 0.5 * (fy1 + fy2)
    ox1, oy1, ox2, oy2 = all_vehicle_boxes[owner_idx]
    owner_area = (ox2 - ox1) * (oy2 - oy1)

    for j, (bx1, by1, bx2, by2) in enumerate(all_vehicle_boxes):
        if j == owner_idx:
            continue
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            other_area = (bx2 - bx1) * (by2 - by1)
            if other_area < owner_area * 0.90:
                return False
    return True


def filter_foreign_plate_reads(frame_reads):
    """
    Clusters readable frames of ONE track by plate text similarity.
    Keeps the strongest cluster (total readable character evidence) and marks
    any foreign outlier plate as skipped with weight 0.0.
    """
    usable_idx = [
        i for i, r in enumerate(frame_reads)
        if r["used"] and len(r["raw_text"]) >= 3
    ]
    if len(usable_idx) <= 1:
        return

    clusters = []
    for idx in usable_idx:
        txt = frame_reads[idx]["raw_text"]
        placed = False
        for cl in clusters:
            for member_idx in cl:
                mtxt = frame_reads[member_idx]["raw_text"]
                sim = SequenceMatcher(None, txt, mtxt).ratio()
                if sim >= 0.50 or (len(txt) >= 4 and (txt in mtxt or mtxt in txt)):
                    cl.append(idx)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([idx])

    if len(clusters) <= 1:
        return

    best_cluster = set(
        max(clusters, key=lambda cl: sum(len(frame_reads[i]["raw_text"]) for i in cl))
    )

    for idx in usable_idx:
        if idx not in best_cluster:
            frame_reads[idx]["used"] = False
            frame_reads[idx]["weight"] = 0.0
            frame_reads[idx]["skip_reason"] = (
                "inconsistent plate reading: does not match primary plate of this track (adjacent vehicle)"
            )


def assign_frame_weights(frame_reads):
    """
    Computes normalized weights strictly across final USED frames,
    capping sharpness at 400.0 so zoomed-in partial fragments don't dominate.
    Skipped frames always get weight = 0.0.
    """
    used_idx = [i for i, r in enumerate(frame_reads) if r["used"]]
    capped = {i: min(max(1.0, float(frame_reads[i]["sharpness"])), 400.0) for i in used_idx}
    total = sum(capped.values())

    for i, r in enumerate(frame_reads):
        if i in capped and total > 0:
            r["weight"] = round(capped[i] / total, 4)
        else:
            r["weight"] = 0.0


def build_consensus_plate(reads, conf_thresh=0.55):
    """
    Combine multi-frame USED reads by picking the dominant plate length among
    strong reads and taking weight- & confidence-weighted votes per character slot.
    Uncertain slots stay '?' (never invented).
    """
    valid = [
        r for r in reads
        if r.get("used", True) and len(r["raw_text"]) >= 4 and r["mean_char_prob"] >= 0.40
    ]
    if not valid:
        return "", []

    len_weights = defaultdict(float)
    for r in valid:
        len_weights[len(r["raw_text"])] += r["mean_char_prob"] * max(0.05, r.get("weight", 1.0))
    target_len = max(len_weights, key=len_weights.get)

    same_len = [r for r in valid if len(r["raw_text"]) == target_len]
    consensus_chars = []
    consensus_probs = []

    for pos in range(target_len):
        slot_votes = defaultdict(float)
        total_w = 0.0
        for r in same_len:
            w = max(0.05, r.get("weight", 1.0))
            ch = r["raw_text"][pos]
            p = r["char_probs"][pos]
            slot_votes[ch] += p * w
            total_w += w

        best_ch = max(slot_votes, key=slot_votes.get)
        avg_prob = slot_votes[best_ch] / max(1e-6, total_w)
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
            filter_foreign_plate_reads(matched_parent["frame_reads"])
            assign_frame_weights(matched_parent["frame_reads"])
            matched_parent["frame_reads"].sort(key=lambda r: r["frame"])
        else:
            rec["merged_track_ids"] = [rec["track_id"]]
            merged.append(rec)

    return merged


def resolve_ov_device(requested="auto"):
    """Picks 'xpu' (Intel Arc iGPU) when available, otherwise 'cpu'."""
    if requested in ("xpu", "cpu"):
        return requested
    try:
        import openvino as ov
        if any(d.startswith("GPU") for d in ov.Core().available_devices):
            return "xpu"
    except Exception:
        pass
    return "cpu"


def load_detectors(backend, requested_device, pt_veh_imgsz, pt_plate_imgsz):
    """
    Loads OpenVINO FP32 models on Intel Arc iGPU ('xpu') or CPU when backend=='ov',
    with automatic fallback to PyTorch .pt weights if OpenVINO folders are missing.
    """
    pt_fallback_dev = "cuda" if torch.cuda.is_available() else "cpu"
    ov_dev = resolve_ov_device(requested_device)

    veh_ov_path = os.path.join(PROJECT_ROOT, "weights", "openvino_scratch", "yolo11s_openvino_model")
    veh_pt_path = os.path.join(PROJECT_ROOT, "weights", "yolo11s.pt")

    plate_xpu_path = os.path.join(PROJECT_ROOT, "weights", "openvino_xpu", "indian_plate_best_openvino_model")
    plate_dyn_path = os.path.join(PROJECT_ROOT, "weights", "openvino_scratch", "indian_plate_best_openvino_model")
    plate_pt_path = os.path.join(PROJECT_ROOT, "weights", "indian_plate_best.pt")

    # 1. Vehicle detector
    if backend == "ov" and os.path.isdir(veh_ov_path):
        try:
            veh_model = YOLO(veh_ov_path, task="detect")
            veh_dev, veh_imgsz = ov_dev, (736, 1280)
            print(f"  Vehicle detector: OpenVINO FP32 ({veh_dev.upper()} @ 736x1280)")
        except Exception as exc:
            print(f"  [WARN] OpenVINO vehicle load failed ({exc}); using PyTorch.")
            veh_model = YOLO(veh_pt_path)
            veh_dev, veh_imgsz = pt_fallback_dev, pt_veh_imgsz
    else:
        veh_model = YOLO(veh_pt_path)
        veh_dev, veh_imgsz = pt_fallback_dev, pt_veh_imgsz
        print(f"  Vehicle detector: PyTorch ({veh_dev.upper()} @ {veh_imgsz}px)")

    # 2. Plate detector (prefer static 480x480 for xpu/cpu speed)
    plate_ov_choice = plate_xpu_path if os.path.isdir(plate_xpu_path) else plate_dyn_path
    if backend == "ov" and os.path.isdir(plate_ov_choice):
        try:
            plate_model = YOLO(plate_ov_choice, task="detect")
            plate_dev, plate_imgsz = ov_dev, 480
            print(f"  Plate detector  : OpenVINO FP32 ({plate_dev.upper()} @ 480px)")
        except Exception as exc:
            print(f"  [WARN] OpenVINO plate load failed ({exc}); using PyTorch.")
            plate_model = YOLO(plate_pt_path)
            plate_dev, plate_imgsz = pt_fallback_dev, pt_plate_imgsz
    else:
        plate_model = YOLO(plate_pt_path)
        plate_dev, plate_imgsz = pt_fallback_dev, pt_plate_imgsz
        print(f"  Plate detector  : PyTorch ({plate_dev.upper()} @ {plate_imgsz}px)")

    return veh_model, veh_dev, veh_imgsz, plate_model, plate_dev, plate_imgsz


def main():
    parser = argparse.ArgumentParser(description="Fast Offline Vision Pipeline (OpenVINO iGPU/CPU) -> vision.json")
    parser.add_argument("video", help="Path to CCTV video (e.g., data/clips/demo6.mp4)")
    parser.add_argument("--out-dir", default="runs/clip1_new", help="Output directory")
    parser.add_argument("--backend", choices=["ov", "pt"], default="ov", help="Detector backend (default: ov)")
    parser.add_argument("--device", choices=["auto", "xpu", "cpu"], default="auto", help="OpenVINO device (default: auto)")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Frames/sec to analyse (default: 5.0)")
    parser.add_argument("--stride", type=int, default=0, help="Manual frame stride override (0 = auto from --target-fps)")
    parser.add_argument("--veh-imgsz", type=int, default=960, help="PyTorch vehicle YOLO size (fallback only)")
    parser.add_argument("--plate-imgsz", type=int, default=480, help="PyTorch plate YOLO size (fallback only)")
    parser.add_argument("--top-k", type=int, default=12, help="Max plate frames evaluated per track")
    parser.add_argument("--min-frames", type=int, default=2, help="Drop noisy tracks with fewer plate frames")
    parser.add_argument("--conf-thresh", type=float, default=0.55, help="Char prob threshold below which -> '?'")
    parser.add_argument("--min-sharpness", type=float, default=60.0, help="Laplacian sharpness cutoff")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Optional max clip duration; 0 = full clip")
    parser.add_argument("--save-video", action="store_true", help="Also encode annotated.mp4 (slower)")
    args = parser.parse_args()

    t_start_all = time.perf_counter()
    print("[1/3] Loading offline models...")
    vehicle_model, veh_device, veh_imgsz, plate_model, plate_device, plate_imgsz = load_detectors(
        args.backend, args.device, args.veh_imgsz, args.plate_imgsz
    )
    recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open {args.video}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame_idx = int(args.max_seconds * fps) if args.max_seconds > 0 else None

    stride = args.stride if args.stride > 0 else max(1, int(round(fps / max(1.0, args.target_fps))))
    est_sampled = (min(total_frames, max_frame_idx or total_frames) // stride) if total_frames > 0 else 0

    writer = None
    if args.save_video:
        video_out = os.path.join(args.out_dir, "annotated.mp4")
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), max(4.0, fps / stride), (w, h))

    tracks = defaultdict(lambda: {
        "seen": 0,
        "sharp_plates": 0,
        "last_colour": "unknown",
        "types": Counter(),
        "colour_items": [],
        "plates": [],
    })

    print(
        f"[2/3] Video: {w}x{h} @ {fps:.1f}fps ({total_frames} frames) | "
        f"stride={stride} (~{est_sampled} sampled frames)..."
    )

    frame_idx = 0
    sampled_count = 0
    while cap.isOpened():
        if frame_idx % stride != 0:
            if not cap.grab():
                break
            frame_idx += 1
            if max_frame_idx is not None and frame_idx > max_frame_idx:
                break
            continue

        ret, frame = cap.read()
        if not ret:
            break
        if max_frame_idx is not None and frame_idx > max_frame_idx:
            break

        sampled_count += 1
        results = vehicle_model.track(
            frame,
            imgsz=veh_imgsz,
            conf=0.35,
            iou=0.5,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            persist=True,
            device=veh_device,
            verbose=False,
        )[0]

        vis = frame.copy() if writer is not None else None

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            clamped_boxes = [
                (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
                for (x1, y1, x2, y2) in boxes
            ]

            for idx_v, ((vx1, vy1, vx2, vy2), tid, cls_id) in enumerate(zip(clamped_boxes, track_ids, cls_ids)):
                tid = int(tid)
                vw, vh = vx2 - vx1, vy2 - vy1
                if vw <= 0 or vh <= 0:
                    continue

                v_type = VEHICLE_CLASSES.get(int(cls_id), "car")
                t = tracks[tid]
                t["seen"] += 1
                t["types"][v_type] += 1

                # Skip tiny distant vehicles whose plates are too small to read
                if vw < 85 or vh < 65:
                    continue

                v_crop = frame[vy1:vy2, vx1:vx2]
                roi_y1 = vy1 + int(vh * 0.30)
                bumper_roi = frame[roi_y1:vy2, vx1:vx2]
                plate_box_in_crop = None

                # Plate YOLO: stop once track already has 12 sharp plates
                if bumper_roi.size > 0 and t["sharp_plates"] < 12:
                    p_res = plate_model.predict(
                        bumper_roi, imgsz=plate_imgsz, conf=0.20, device=plate_device, verbose=False
                    )[0]
                    if len(p_res.boxes) > 0:
                        sorted_boxes = sorted(p_res.boxes, key=lambda b: float(b.conf[0]), reverse=True)
                        for best_box in sorted_boxes:
                            px1, py1, px2, py2 = map(int, best_box.xyxy[0])
                            p_conf = float(best_box.conf[0])

                            fx1, fy1 = int(vx1 + px1), int(roi_y1 + py1)
                            fx2, fy2 = int(vx1 + px2), int(roi_y1 + py2)

                            if not plate_belongs_to_vehicle((fx1, fy1, fx2, fy2), idx_v, clamped_boxes, w, h):
                                continue

                            plate_box_in_crop = (fx1 - vx1, fy1 - vy1, fx2 - vx1, fy2 - vy1)

                            pad_x = max(2, int((fx2 - fx1) * 0.05))
                            pad_y = max(2, int((fy2 - fy1) * 0.08))
                            raw_crop = frame[max(0, fy1 - pad_y):min(h, fy2 + pad_y),
                                             max(0, fx1 - pad_x):min(w, fx2 + pad_x)]

                            if raw_crop.size > 0 and raw_crop.shape[1] >= 20 and raw_crop.shape[0] >= 8:
                                sharp = sharpness_score(raw_crop)
                                if sharp >= 120.0:
                                    t["sharp_plates"] += 1

                                t["plates"].append({
                                    "frame": int(frame_idx),
                                    "time_s": round(frame_idx / fps, 2),
                                    "plate_bbox": [fx1, fy1, fx2, fy2],
                                    "plate_conf": round(p_conf, 3),
                                    "sharpness": round(sharp, 2),
                                    "raw_crop": raw_crop.copy(),
                                    "vehicle_crop": v_crop.copy(),
                                })
                                if vis is not None:
                                    cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                                break

                # CIELAB Colour Estimation: capped at 8 samples per track
                if len(t["colour_items"]) < 8 and (t["seen"] % 2 == 1):
                    c_est = estimate_colour(
                        v_crop,
                        vehicle_type=v_type,
                        plate_box_in_crop=plate_box_in_crop,
                        use_plate_reference=True,
                    )
                    touches_border = (vx1 <= 5 or vy1 <= 5 or vx2 >= w - 5 or vy2 >= h - 5)
                    penalty = 0.3 if touches_border else 1.0
                    area_px = float(vw * vh)
                    t["colour_items"].append((c_est, area_px, penalty))
                    if c_est.value != "unknown":
                        t["last_colour"] = c_est.value

                if vis is not None:
                    cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID {tid} | {t['last_colour']} {v_type}", (vx1, max(20, vy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if writer is not None:
            writer.write(vis)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"[3/3] Processed {sampled_count} frames. Running multi-frame OCR & building vision.json...")
    vehicle_records = []
    for tid, data in tracks.items():
        if len(data["plates"]) < args.min_frames:
            continue

        chronological = sorted(data["plates"], key=lambda x: x["frame"])
        if len(chronological) > args.top_k:
            idx_pick = np.linspace(0, len(chronological) - 1, args.top_k, dtype=int)
            selected_plates = [chronological[i] for i in idx_pick]
        else:
            selected_plates = chronological

        track_dir = os.path.join(args.out_dir, f"track_{tid}")
        os.makedirs(track_dir, exist_ok=True)

        frame_reads = []
        for item in selected_plates:
            fidx = item["frame"]
            raw_path = os.path.join(track_dir, f"f{fidx:04d}_raw.jpg")
            enh_path = os.path.join(track_dir, f"f{fidx:04d}_enh.jpg")
            vehicle_path = os.path.join(track_dir, f"f{fidx:04d}_vehicle.jpg")

            enh_crop = enhance_plate(item["raw_crop"])
            cv2.imwrite(raw_path, item["raw_crop"])
            cv2.imwrite(enh_path, enh_crop)
            cv2.imwrite(vehicle_path, item["vehicle_crop"])

            enh_text, enh_probs = run_fast_plate_ocr(recognizer, enh_path)
            enh_mean = float(np.mean(enh_probs)) if enh_probs else 0.0

            if enh_mean >= 0.85 and len(enh_text) >= 6:
                best_text, best_probs, src, best_img, best_mean = enh_text, enh_probs, "enh", enh_path, enh_mean
            else:
                raw_text, raw_probs = run_fast_plate_ocr(recognizer, raw_path)
                raw_mean = float(np.mean(raw_probs)) if raw_probs else 0.0
                if enh_mean >= raw_mean:
                    best_text, best_probs, src, best_img, best_mean = enh_text, enh_probs, "enh", enh_path, enh_mean
                else:
                    best_text, best_probs, src, best_img, best_mean = raw_text, raw_probs, "raw", raw_path, raw_mean

            used = True
            skip_reason = None
            if item["sharpness"] < args.min_sharpness:
                used = False
                skip_reason = f"motion blur: sharpness {item['sharpness']:.1f} < {args.min_sharpness:.0f}"
            elif len(best_text) == 0:
                used = False
                skip_reason = "OCR returned no readable characters"

            frame_reads.append({
                "frame": fidx,
                "time_s": item["time_s"],
                "plate_bbox": item["plate_bbox"],
                "plate_conf": item["plate_conf"],
                "sharpness": item["sharpness"],
                "weight": 0.0,
                "used": used,
                "skip_reason": skip_reason,
                "source": src,
                "raw_text": best_text,
                "masked_text": mask_uncertain(best_text, best_probs, args.conf_thresh),
                "char_probs": best_probs,
                "mean_char_prob": round(best_mean, 4),
                "image_path": best_img,
                "vehicle_path": vehicle_path,
            })

        if not any(r["used"] for r in frame_reads):
            readable = [r for r in frame_reads if len(r["raw_text"]) > 0]
            readable.sort(key=lambda r: r["sharpness"], reverse=True)
            for r in readable[:3]:
                r["used"] = True
                r["skip_reason"] = None

        filter_foreign_plate_reads(frame_reads)
        assign_frame_weights(frame_reads)

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
    for v in merged_vehicles:
        v["consensus_plate"], v["consensus_char_probs"] = build_consensus_plate(v["frame_reads"], args.conf_thresh)

    elapsed_s = round(time.perf_counter() - t_start_all, 2)
    vision_payload = {
        "video_path": args.video,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "elapsed_s": elapsed_s,
        "vehicles": merged_vehicles,
    }

    out_json = os.path.join(args.out_dir, "vision.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vision_payload, f, indent=2, default=int)

    print(f"\nDone in {elapsed_s:.2f}s! Saved {len(merged_vehicles)} unique vehicle(s) to {out_json}")
    for v in merged_vehicles:
        ids_str = ",".join(map(str, v["merged_track_ids"]))
        used_n = sum(1 for r in v["frame_reads"] if r["used"])
        print(
            f"  Track [{ids_str}] -> plate: {v['consensus_plate'] or '(unreadable)':12s} | "
            f"{v['colour']['value']} ({v['colour']['confidence']:.2f}) {v['type']['value']} | "
            f"{used_n}/{len(v['frame_reads'])} frames used"
        )


if __name__ == "__main__":
    main()