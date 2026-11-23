"""Creates clearly-labelled MOCK sample images so the demo viewer has something to show.
Run:  python tools/make_sample_images.py     (needs opencv-python-headless and numpy only)
These are illustrations drawn from the mock JSON text. They are NOT real CCTV evidence."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOCKS = {"demo_001": "mock_result.json", "demo_002_nomatch": "mock_no_match.json"}


def plate(text, w, h, sigma):
    S = 3
    img = np.full((h * S, w * S, 3), (235, 238, 240), np.uint8)
    cv2.rectangle(img, (4, 4), (w * S - 5, h * S - 5), (60, 60, 60), 3)
    scale = min(1.6, w * S / (len(text) * 26))
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)[0]
    cv2.putText(img, text, ((w * S - size[0]) // 2, (h * S + size[1]) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (30, 30, 30), 3, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    noise = np.random.default_rng(7).normal(0, 6, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.putText(img, "MOCK SAMPLE", (8, h * S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    return img


for run_id, fname in MOCKS.items():
    d = json.load(open(ROOT / "app/mock" / fname, encoding="utf-8"))
    out = ROOT / "app/mock/sample_images" / run_id
    out.mkdir(parents=True, exist_ok=True)
    v = d["vision"]
    car = np.full((360, 640, 3), (226, 230, 236), np.uint8)
    cv2.rectangle(car, (120, 120), (520, 280), (150, 158, 172), -1)
    cv2.circle(car, (200, 285), 32, (60, 60, 70), -1); cv2.circle(car, (440, 285), 32, (60, 60, 70), -1)
    cv2.putText(car, f"MOCK SAMPLE: {v['vehicle_class']} crop", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / Path(v["vehicle_crop_path"]).name), car)
    for f in v["frames"]:
        b = f["plate_box"]; w, h = int(b[2] - b[0]), int(b[3] - b[1])
        sigma = max(0.6, 5.0 - f["sharpness"] / 30)
        cv2.imwrite(str(out / Path(f["plate_crop_path"]).name), plate(f["raw_text"], w, h, sigma))
        if f.get("enhanced_crop_path"):
            cv2.imwrite(str(out / Path(f["enhanced_crop_path"]).name), plate(f["raw_text"], w, h, sigma * 0.6))
    print("wrote", out)
