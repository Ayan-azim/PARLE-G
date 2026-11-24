# TraceVision

CCTV vehicle and number plate recognition app for the hackathon project.

## Python

Use **Python 3.11.9** for both the backend and model pipeline. The required version is recorded in `.python-version`.

For setup and app usage, see [backend/README.md](backend/README.md).

# Forensic CCTV Vehicle Identification & Traceability System

A **100% offline, zero-hallucination** computer vision and forensic matching pipeline for degraded, motion-blurred CCTV footage. Instead of guessing unreadable characters or sending sensitive footage to cloud Vision LLMs, the system extracts calibrated per-character probabilities, preserves uncertain positions as literal `?` wildcards, and produces an auditable Top-5 candidate shortlist—or honestly reports **No confident match** when evidence is insufficient.

---

## Why This Exists

* **Real-World CCTV Is Degraded:** Distant, low-light, and motion-blurred frames cause standard single-pass ANPR systems to fail.
* **Privacy & Offline Requirement:** Police and forensic footage cannot be uploaded to external cloud APIs.
* **Zero Hallucination:** Cloud VLMs frequently autocomplete blurry digits, risking false identification. Every character in this system is backed by frame-level crops, sharpness scores, readability weights, and OCR confidence.

---

## Key Features

* **Two-Stage Cascaded Vision Pipeline:** Tracks vehicles with `YOLO11s` + `ByteTrack` at adaptive 5.0 FPS sampling (`cap.grab()`), then runs a dedicated **Indian License Plate YOLO** detector on the lower 70% bumper ROI at native resolution.
* **Adjacent-Vehicle Plate Protection:** Spatial bounding-box ownership checks and post-OCR similarity clustering prevent large vehicles (trucks/buses) from stealing a passing car's plate.
* **Quality-Gated Multi-Frame OCR Fusion:** Uses offline ONNX `fast-plate-ocr` (`cct-s-v2-global-model`) with CIELAB CLAHE enhancement. Frames are weighted by readability (`OCR confidence³ × readable length × plate crop area × capped sharpness`), and skipped frames receive `0.0` weight.
* **Confusable-Aware Indian Plate Alignment:** Handles common Indian OCR confusions (`B↔8`, `0↔O`, `1↔I/L`, `5↔S`, `2↔Z`, `6↔G`) during alignment and trimming without truncating valid 10-character plates or altering raw evidence.
* **Perceptual CIELAB Colour Estimation:** Masks windshields, tyres, glare, and plates to output a soft colour distribution resilient to shadows and night lighting.
* **Intel Arc iGPU Acceleration (OpenVINO FP32):** Runs full FP32 OpenVINO models on Intel integrated GPU (`device="xpu"`) with zero INT8 accuracy loss and automatic fallback to CPU/PyTorch.
* **Audit-Ready Streamlit UI:** Displays every used and skipped plate crop, exact timestamps, sharpness scores, weights, fused character confidence tiles, and explainable candidate rankings.

---

## System Architecture

```text
CCTV Video
   │
   ▼
[1. Vision & Tracking]  (models/scripts/run_vision_pipeline.py)
   ├── YOLO11s + ByteTrack (OpenVINO FP32 on Arc iGPU / CPU, 5.0 FPS adaptive stride)
   ├── Indian License Plate YOLO (Static 480×480 OpenVINO FP32 on bumper ROI)
   ├── CIELAB Colour Estimator (vision/colour.py, capped at 8 samples/track)
   └── fast-plate-ocr + CIELAB CLAHE Enhancement (per-character probabilities)
   │
   ▼
[2. Schema Adapter & Weighting]  (backend/vision/new_model_pipeline.py)
   └── Pydantic v2 contract (schema.py) + readability-based frame weighting
   │
   ▼
[3. Forensic Fusion & Matching]  (backend/matching/fusion.py, align.py)
   ├── Confusable-aware edit-distance alignment across frames
   ├── Weighted per-column voting (uncertain slots remain '?')
   └── Probabilistic registry matching & explainable Top-5 shortlist
   │
   ▼
[4. Forensic Evidence UI]  (backend/app/main.py)
   └── Interactive Streamlit dashboard with full crop & confidence audit trail