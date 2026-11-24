"""Adapter from the models/ multi-vehicle pipeline to the shared app schema."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from schema import CharReading, FrameReading, Settings, VisionOutput


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PIPELINE = MODELS / "scripts" / "run_vision_pipeline.py"


def _frame_quality_score(chars: list[CharReading], sharpness: float, plate_box) -> float:
    """Score a usable frame so crisp, high-confidence, full plates outweigh blurry/noisy ones."""
    if not chars:
        return 0.0
    mean_conf = sum(c.conf for c in chars) / len(chars)
    # Strongly separate clear reads (0.95^3 = 0.86) from blurry guesses (0.50^3 = 0.13)
    conf_factor = max(0.05, mean_conf) ** 3
    # Full 9-10 char plates outrank partial blurry fragments
    len_factor = (min(len(chars), 10) / 10.0) ** 1.5
    # Cap sharpness at 400 so noisy/high-contrast blurry borders cannot outscore clean plates
    sharp_factor = 0.5 + 0.5 * (min(max(0.0, sharpness), 400.0) / 400.0)
    # Larger plate crops (closer vehicle) carry more pixel detail
    area_factor = 1.0
    if plate_box and len(plate_box) == 4:
        bw = max(1.0, float(plate_box[2]) - float(plate_box[0]))
        bh = max(1.0, float(plate_box[3]) - float(plate_box[1]))
        area_factor = min(2.0, max(0.35, math.sqrt(bw * bh) / 60.0))
    return conf_factor * len_factor * sharp_factor * area_factor


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

        # First pass: parse chars, usability, and readability score for each frame
        prepared = []
        for read in read_items:
            raw = str(read.get("raw_text") or "").upper()
            probs = read.get("char_probs") or []
            chars = [
                CharReading(
                    char=ch,
                    conf=max(0.0, min(1.0, float(probs[i])) if i < len(probs) else 0.5),
                )
                for i, ch in enumerate(raw) if ch.isalnum()
            ]
            sharpness = round(max(0.0, float(read.get("sharpness", 0))), 2)
            frame_box = read.get("plate_bbox")
            used = bool(chars) and bool(read.get("used", True))
            skip_reason = None
            if not used:
                skip_reason = (
                    str(read.get("skip_reason"))
                    if read.get("skip_reason")
                    else "OCR returned no readable characters"
                )
            score = _frame_quality_score(chars, sharpness, frame_box) if used else 0.0
            prepared.append((read, chars, sharpness, frame_box, used, skip_reason, score))

        score_total = sum(item[6] for item in prepared)
        frames = []
        best_vehicle_source = None
        best_vehicle_score = -1.0

        for read, chars, sharpness, frame_box, used, skip_reason, score in prepared:
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

            norm_weight = round(score / score_total, 4) if (used and score_total > 0) else 0.0
            norm_weight = min(1.0, max(0.0, norm_weight))

            frames.append(FrameReading(
                frame_idx=frame_idx, timestamp_s=round(frame_idx / fps, 2), sharpness=sharpness,
                quality_weight=norm_weight,
                used=used, skip_reason=skip_reason,
                plate_box=tuple(map(float, frame_box)) if frame_box else None,
                plate_crop_path=rel_image, raw_text="".join(c.char for c in chars), chars=chars,
            ))

            # Track the best-scoring frame's vehicle crop for a cleaner thumbnail
            source_vehicle = Path(str(read.get("vehicle_path") or ""))
            if not source_vehicle.is_absolute():
                source_vehicle = (MODELS / source_vehicle).resolve()
            if source_vehicle.is_file() and score > best_vehicle_score:
                best_vehicle_score = score
                best_vehicle_source = source_vehicle

        frames.sort(key=lambda f: f.frame_idx)
        vehicle_crop_rel = None
        if best_vehicle_source is not None:
            vehicle_dest = output_dir / "evidence" / f"track_{track_id}" / "vehicle.jpg"
            vehicle_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_vehicle_source, vehicle_dest)
            vehicle_crop_rel = vehicle_dest.resolve().relative_to(Path(run_dir).resolve()).as_posix()

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