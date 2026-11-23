"""Adapter from the models/ multi-vehicle pipeline to the shared app schema."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from schema import CharReading, FrameReading, Settings, VisionOutput


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PIPELINE = MODELS / "scripts" / "run_vision_pipeline.py"


def run_vision_batch(video_path: str, settings: Settings, run_dir: str) -> list[VisionOutput]:
    """Run the new detector/OCR pipeline and adapt every track for Person B."""
    output_dir = Path(run_dir) / "model_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(Path(video_path).resolve()))
    source_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
    source_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
    cap.release()
    source_fps = source_fps if source_fps and source_fps > 0 else 25.0
    duration_s = source_frames / source_fps if source_frames and source_frames > 0 else 0.0
    stride = max(1, round(source_fps / max(1.0, settings.sample_fps)))
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), str(Path(video_path).resolve()),
         "--out-dir", str(output_dir.resolve()), "--stride", str(stride),
         "--max-seconds", str(settings.max_clip_seconds)],
        cwd=str(MODELS), capture_output=True, text=True,
    )
    if proc.returncode:
        raise RuntimeError("New model pipeline failed: " + (proc.stderr or proc.stdout)[-3000:])

    payload_path = output_dir / "vision.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    fps = float(payload.get("fps") or 25.0)
    frame_size = (int(payload.get("width", 0)), int(payload.get("height", 0)))
    tracks = payload.get("vehicles", [])
    results: list[VisionOutput] = []
    for vehicle in tracks:
        track_id = int(vehicle["track_id"])
        read_items = vehicle.get("frame_reads", [])
        sharp_total = sum(max(0.0, float(r.get("sharpness", 0))) for r in read_items)
        frames = []
        for read in read_items:
            raw = str(read.get("raw_text") or "").upper()
            probs = read.get("char_probs") or []
            chars = [CharReading(char=ch, conf=max(0.0, min(1.0, float(probs[i])) if i < len(probs) else 0.5))
                     for i, ch in enumerate(raw) if ch.isalnum()]
            frame_idx = int(read.get("frame", 0))
            source_image = Path(str(read.get("image_path") or ""))
            if not source_image.is_absolute():
                source_image = (MODELS / source_image).resolve()
            rel_image = None
            if source_image.is_file():
                dest = output_dir / "evidence" / f"track_{track_id}" / f"frame_{frame_idx:06d}{source_image.suffix or '.jpg'}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if source_image.resolve() != dest.resolve():
                    shutil.copy2(source_image, dest)
                rel_image = dest.resolve().relative_to(Path(run_dir).resolve()).as_posix()
            sharpness = max(0.0, float(read.get("sharpness", 0)))
            frame_box = read.get("plate_bbox")
            frames.append(FrameReading(
                frame_idx=frame_idx, timestamp_s=frame_idx / fps, sharpness=sharpness,
                quality_weight=sharpness / sharp_total if sharp_total else 0.0,
                used=bool(chars), skip_reason=None if chars else "OCR returned no readable characters",
                plate_box=tuple(map(float, frame_box)) if frame_box else None,
                plate_crop_path=rel_image, raw_text="".join(c.char for c in chars), chars=chars,
            ))
        frames.sort(key=lambda f: f.frame_idx)
        vehicle_crop_rel = None
        for read in read_items:
            source_vehicle = Path(str(read.get("vehicle_path") or ""))
            if not source_vehicle.is_absolute():
                source_vehicle = (MODELS / source_vehicle).resolve()
            if source_vehicle.is_file():
                vehicle_dest = output_dir / "evidence" / f"track_{track_id}" / "vehicle.jpg"
                vehicle_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_vehicle, vehicle_dest)
                vehicle_crop_rel = vehicle_dest.resolve().relative_to(Path(run_dir).resolve()).as_posix()
                break
        colour = vehicle.get("colour") or {}
        vehicle_type = vehicle.get("type") or {}
        results.append(VisionOutput(
            clip_name=Path(video_path).name, duration_s=duration_s, fps=fps,
            frame_size=frame_size, sampled_fps=fps / max(1, stride),
            track_id=track_id, target_selection_reason="tracked vehicle from multi-vehicle model pipeline",
            vehicle_class=str(vehicle_type.get("value") or "unknown"),
            vehicle_crop_path=vehicle_crop_rel, frames=frames,
            notes=[f"Model pipeline detected {len(tracks)} vehicle track(s)."],
            vehicle_id=track_id, vehicle_count=len(tracks),
            model_colour=colour, model_vehicle_type=vehicle_type,
        ))

    if not results:
        results.append(VisionOutput(
            clip_name=Path(video_path).name, duration_s=duration_s, fps=fps,
            frame_size=frame_size, sampled_fps=fps / max(1, stride), track_id=-1,
            target_selection_reason="no vehicle track produced plate readings", vehicle_class="unknown",
            frames=[], notes=["No vehicle with usable plate reads was detected."], vehicle_id=None,
            vehicle_count=0,
        ))
    return results
