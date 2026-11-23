"""Writes the VEHICLE TRACKER logo (SVG) and favicon (PNG). Run: python tools/make_brand_assets.py"""
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "app" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logo.svg").write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="96" height="96"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#047857"/><stop offset="1" stop-color="#34D399"/></linearGradient></defs><rect width="48" height="48" rx="13" fill="url(#g)"/><path d="M24 8c-6.6 0-12 5.2-12 11.7 0 8.2 12 18.3 12 18.3s12-10.1 12-18.3C36 13.2 30.6 8 24 8z" fill="none" stroke="#fff" stroke-width="2.6"/><circle cx="24" cy="20" r="5.2" fill="none" stroke="#fff" stroke-width="2.4"/><circle cx="24" cy="20" r="1.9" fill="#fff"/><path d="M9 42h9M30 42h9" stroke="#fff" stroke-opacity=".75" stroke-width="2.4" stroke-linecap="round"/></svg>''', encoding="utf-8")

S = 256
yy, xx = np.mgrid[0:S, 0:S]
t = (xx + yy) / (2 * S)
img = np.zeros((S, S, 4), np.uint8)
img[..., 0] = 229 * (1 - t) + 212 * t   # B
img[..., 1] = 70 * (1 - t) + 182 * t    # G
img[..., 2] = 79 * (1 - t) + 6 * t      # R
img[..., 3] = 255
mask = np.zeros((S, S), np.uint8)
cv2.rectangle(mask, (0, 0), (S - 1, S - 1), 255, -1)
r = 56
for cx, cy in ((r, r), (S - r, r), (r, S - r), (S - r, S - r)):
    pass
mask = np.zeros((S, S), np.uint8)
cv2.rectangle(mask, (r, 0), (S - r, S), 255, -1); cv2.rectangle(mask, (0, r), (S, S - r), 255, -1)
for cx, cy in ((r, r), (S - r, r), (r, S - r), (S - r, S - r)):
    cv2.circle(mask, (cx, cy), r, 255, -1)
img[..., 3] = mask
W = (255, 255, 255, 255)
pts = np.array([[128, 205], [70, 120], [72, 90], [90, 58], [128, 44], [166, 58], [184, 90], [186, 120]], np.int32)
cv2.polylines(img, [pts], True, W, 10, cv2.LINE_AA)
cv2.circle(img, (128, 108), 28, W, 9, cv2.LINE_AA); cv2.circle(img, (128, 108), 10, W, -1, cv2.LINE_AA)
cv2.line(img, (52, 226), (98, 226), W, 10, cv2.LINE_AA); cv2.line(img, (158, 226), (204, 226), W, 10, cv2.LINE_AA)
cv2.imwrite(str(OUT / "favicon.png"), cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))
print("wrote", OUT)
