"""
Model & Pipeline Latency Checker (PyTorch vs OpenVINO).

Usage:
    python scripts/benchmark_latency.py data/clips/clip1.mp4 --backend ov
    python scripts/benchmark_latency.py data/clips/clip1.mp4 --backend pt
    python scripts/benchmark_latency.py data/clips/clip1.mp4 --backend both
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.colour import estimate_colour

VEHICLE_CLASSES = {2: "car", 3: "two-wheeler", 5: "bus", 7: "truck"}


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


def summarize(times_s):
    if not times_s:
        return {"calls": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "total_s": 0.0}
    arr = np.array(times_s, dtype=np.float64)
    return {
        "calls": int(len(arr)),
        "avg_ms": round(float(arr.mean() * 1000.0), 2),
        "min_ms": round(float(arr.min() * 1000.0), 2),
        "max_ms": round(float(arr.max() * 1000.0), 2),
        "total_s": round(float(arr.sum()), 3),
    }


def load_detectors(backend):
    """Loads either PyTorch (.pt) or OpenVINO exported folders."""
    if backend == "ov":
        ov_dir = os.path.join(PROJECT_ROOT, "weights", "openvino_scratch")
        veh_path = os.path.join(ov_dir, "yolo11s_openvino_model")
        plate_path = os.path.join(PROJECT_ROOT, "weights", "openvino_xpu", "indian_plate_best_openvino_model")
        device = "xpu"

        t0 = time.perf_counter()
        vehicle_model = YOLO(veh_path, task="detect")
        t_load_veh = time.perf_counter() - t0

        t0 = time.perf_counter()
        plate_model = YOLO(plate_path, task="detect")
        t_load_plate = time.perf_counter() - t0

        # Always force CPU for OpenVINO to avoid unpredictable AUTO/iGPU stalls
        device = "xpu"
        veh_imgsz = (736, 1280)
    else:
        t0 = time.perf_counter()
        vehicle_model = YOLO(os.path.join(PROJECT_ROOT, "weights", "yolo11s.pt"))
        t_load_veh = time.perf_counter() - t0

        t0 = time.perf_counter()
        plate_model = YOLO(os.path.join(PROJECT_ROOT, "weights", "indian_plate_best.pt"))
        t_load_plate = time.perf_counter() - t0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        veh_imgsz = 1280

    return vehicle_model, plate_model, t_load_veh, t_load_plate, device, veh_imgsz


def benchmark_components(video_path, backend="ov", stride=2, max_frames=40):
    vehicle_model, plate_model, t_load_veh, t_load_plate, device, veh_imgsz = load_detectors(backend)

    t0 = time.perf_counter()
    recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
    t_load_ocr = time.perf_counter() - t0

    print(
        f"\n[Benchmark | Backend: {backend.upper()}] Device: {device.upper()} | "
        f"Video: {video_path} | Stride: {stride} | Max sampled frames: {max_frames}"
    )

    load_times = {
        "yolo11s_load_s": round(t_load_veh, 3),
        "plate_yolo_load_s": round(t_load_plate, 3),
        "fast_plate_ocr_load_s": round(t_load_ocr, 3),
        "total_load_s": round(t_load_veh + t_load_plate + t_load_ocr, 3),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_veh_list = []
    t_plate_list = []
    t_colour_list = []
    t_ocr_list = []

    tmp_file = os.path.join(tempfile.gettempdir(), "_bench_plate_tmp.jpg")
    frame_idx = 0
    sampled = 0
    loop_start = time.perf_counter()

    while cap.isOpened() and sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        sampled += 1

        # Stage A: Vehicle YOLO + ByteTrack
        t0 = time.perf_counter()
        results = vehicle_model.track(
            frame,
            imgsz=veh_imgsz,
            conf=0.35,
            iou=0.5,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            persist=True,
            device=device,
            verbose=False,
        )[0]
        t_veh_list.append(time.perf_counter() - t0)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cls_id in zip(boxes, cls_ids):
                vx1, vy1 = max(0, int(x1)), max(0, int(y1))
                vx2, vy2 = min(w, int(x2)), min(h, int(y2))
                v_crop = frame[vy1:vy2, vx1:vx2]
                if v_crop.size == 0:
                    continue

                v_type = VEHICLE_CLASSES.get(int(cls_id), "car")
                vh = vy2 - vy1
                roi_y1 = vy1 + int(vh * 0.30)
                bumper_roi = frame[roi_y1:vy2, vx1:vx2]
                plate_box_in_crop = None

                # Stage B: Indian Plate YOLO
                if bumper_roi.size > 0:
                    t0 = time.perf_counter()
                    p_res = plate_model.predict(
                        bumper_roi, imgsz=480, conf=0.20, device=device, verbose=False
                    )[0]
                    t_plate_list.append(time.perf_counter() - t0)

                    if len(p_res.boxes) > 0:
                        best_box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                        px1, py1, px2, py2 = map(int, best_box.xyxy[0])
                        fx1, fy1 = vx1 + px1, roi_y1 + py1
                        fx2, fy2 = vx1 + px2, roi_y1 + py2
                        plate_box_in_crop = (fx1 - vx1, fy1 - vy1, fx2 - vx1, fy2 - vy1)

                        raw_crop = frame[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                        if raw_crop.size > 0 and raw_crop.shape[1] >= 20 and raw_crop.shape[0] >= 8:
                            # Stage D: Plate Enhancement + fast-plate-ocr
                            t0 = time.perf_counter()
                            enh = enhance_plate(raw_crop)
                            cv2.imwrite(tmp_file, enh)
                            recognizer.run(tmp_file, return_confidence=True)
                            t_ocr_list.append(time.perf_counter() - t0)

                # Stage C: CIELAB Colour Estimation
                t0 = time.perf_counter()
                estimate_colour(
                    v_crop,
                    vehicle_type=v_type,
                    plate_box_in_crop=plate_box_in_crop,
                    use_plate_reference=True,
                )
                t_colour_list.append(time.perf_counter() - t0)

        frame_idx += 1

    cap.release()
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    total_loop_s = round(time.perf_counter() - loop_start, 3)

    stats = {
        f"YOLO11s + ByteTrack ({backend.upper()})": summarize(t_veh_list),
        f"Indian Plate YOLO ({backend.upper()})": summarize(t_plate_list),
        "CIELAB Colour Estimation": summarize(t_colour_list),
        "Enhance + fast-plate-ocr": summarize(t_ocr_list),
    }

    total_model_s = sum(s["total_s"] for s in stats.values()) or 1e-6

    print("\n--- 1. Model Load Delay ---")
    for k, v in load_times.items():
        print(f"  {k:24s}: {v:.3f} s")

    print(f"\n--- 2. Inference Delay ({sampled} sampled frames, total wall time {total_loop_s:.2f}s) ---")
    print(f"  {'Stage':32s} | {'Calls':>5s} | {'Avg (ms)':>9s} | {'Max (ms)':>9s} | {'Total (s)':>9s} | {'Share':>6s}")
    print("  " + "-" * 82)
    for name, s in stats.items():
        share = (s["total_s"] / total_model_s) * 100.0
        print(
            f"  {name:32s} | {s['calls']:5d} | {s['avg_ms']:9.2f} | "
            f"{s['max_ms']:9.2f} | {s['total_s']:9.3f} | {share:5.1f}%"
        )

    eff_fps = round(sampled / total_loop_s, 2) if total_loop_s > 0 else 0.0
    print(f"\n  Effective processing speed: {eff_fps} sampled frames/sec")

    return {
        "backend": backend,
        "device": device,
        "video": video_path,
        "sampled_frames": sampled,
        "total_loop_s": total_loop_s,
        "effective_fps": eff_fps,
        "load_times_s": load_times,
        "stages": stats,
    }


def benchmark_e2e(video_path, registry_path, out_dir):
    print("\n--- 3. Full End-to-End Subprocess Timing ---")
    py = sys.executable

    t0 = time.perf_counter()
    subprocess.run(
        [py, os.path.join(PROJECT_ROOT, "scripts", "run_vision_pipeline.py"), video_path, "--out-dir", out_dir],
        check=True,
    )
    t_stage1 = round(time.perf_counter() - t0, 3)

    vision_json = os.path.join(out_dir, "vision.json")
    t0 = time.perf_counter()
    subprocess.run(
        [py, os.path.join(PROJECT_ROOT, "scripts", "match_registry.py"), vision_json, "--registry", registry_path],
        check=True,
    )
    t_stage2 = round(time.perf_counter() - t0, 3)

    total_e2e = round(t_stage1 + t_stage2, 3)
    print(f"\n  Stage 1 (run_vision_pipeline.py): {t_stage1:.2f} s")
    print(f"  Stage 2 (match_registry.py)     : {t_stage2:.2f} s")
    print(f"  Total End-to-End Delay          : {total_e2e:.2f} s")
    return {"stage1_vision_s": t_stage1, "stage2_matching_s": t_stage2, "total_e2e_s": total_e2e}


def main():
    parser = argparse.ArgumentParser(description="Benchmark model & pipeline latency (PyTorch vs OpenVINO)")
    parser.add_argument("video", help="Path to CCTV clip (e.g., data/clips/clip1.mp4)")
    parser.add_argument("--backend", choices=["pt", "ov", "both"], default="ov", help="Model backend to benchmark")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=40, help="Max sampled frames to profile")
    parser.add_argument("--e2e", action="store_true", help="Also time full run_vision_pipeline.py + match_registry.py")
    parser.add_argument("--registry", default=os.path.join(PROJECT_ROOT, "data", "registry.csv"))
    parser.add_argument("--out-dir", default="runs/benchmark")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}

    if args.backend in ("pt", "both"):
        report["pt"] = benchmark_components(args.video, backend="pt", stride=args.stride, max_frames=args.max_frames)
    if args.backend in ("ov", "both"):
        report["ov"] = benchmark_components(args.video, backend="ov", stride=args.stride, max_frames=args.max_frames)

    if args.backend == "both":
        pt_t = report["pt"]["total_loop_s"]
        ov_t = report["ov"]["total_loop_s"]
        speedup = round(pt_t / ov_t, 2) if ov_t > 0 else 0.0
        print(f"\n================ SUMMARY ================")
        print(f"  PyTorch total  : {pt_t:.2f} s")
        print(f"  OpenVINO total : {ov_t:.2f} s")
        print(f"  Overall speedup: {speedup}x")

    if args.e2e:
        report["e2e"] = benchmark_e2e(args.video, args.registry, args.out_dir)

    report_path = os.path.join(args.out_dir, f"latency_report_{args.backend}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved latency report -> {report_path}")


if __name__ == "__main__":
    main()