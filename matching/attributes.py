"""Colour and body-type attributes from A's vehicle crop and detections (Person B)."""
from __future__ import annotations

import os
from statistics import median
from typing import Optional

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from schema import AttrAlt, Attribute, Attributes, UserHints, VisionOutput

PALETTE = {
    "white": (240, 240, 240), "silver": (192, 192, 196), "grey": (128, 128, 128),
    "black": (20, 20, 20), "red": (190, 20, 30), "blue": (25, 60, 160), "green": (30, 110, 50),
    "yellow": (240, 200, 30), "orange": (240, 120, 20), "brown": (110, 70, 40), "beige": (200, 185, 150),
}
_PAL_LAB = {n: rgb2lab(np.array([[rgb]], dtype=float) / 255.0)[0, 0] for n, rgb in PALETTE.items()}


def colour_from_crop(bgr: np.ndarray) -> Attribute:
    """Central body region (avoids windows above and wheels/ground below) -> Lab -> k-means -> palette."""
    h, w = bgr.shape[:2]
    roi = bgr[int(0.40 * h): int(0.80 * h), int(0.15 * w): int(0.85 * w)]
    if roi.size == 0:
        return Attribute(value="unknown", confidence=0.0, method="vehicle crop too small")
    roi = cv2.resize(roi, (48, 32))
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 1, 3) / 255.0
    lab = rgb2lab(rgb).reshape(-1, 3)
    km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=3)
    k = int(counts.argmax())
    centre, share = km.cluster_centers_[k], counts[k] / counts.sum()
    dist = {n: float(np.linalg.norm(centre - l)) for n, l in _PAL_LAB.items()}
    order = sorted(dist, key=dist.get)
    d1, d2 = dist[order[0]], dist[order[1]]
    ratio = d2 / (d1 + d2 + 1e-9)                       # 0.5 = ambiguous, 1.0 = clear
    conf = float(max(0.0, min(1.0, share * (0.3 + 0.7 * (2 * ratio - 1)))))
    rgb_c = np.clip(lab2rgb(centre.reshape(1, 1, 3))[0, 0], 0, 1)
    hexv = "#%02X%02X%02X" % tuple(int(round(c * 255)) for c in rgb_c)
    alts = [AttrAlt(value=n, confidence=float(min(conf, conf * (d1 + 1) / (dist[n] + 1)))) for n in order[1:3]]
    return Attribute(value=order[0], confidence=conf, alternatives=alts, hex=hexv,
                     method=f"k-means (k=3) on central body region in Lab space; dominant cluster {share:.0%} of pixels")


def body_type(vehicle_class: str, aspect_ratios: list[float]) -> Attribute:
    """Detector class + aspect-ratio rules. Deliberately low confidence for car sub-types."""
    vc = vehicle_class.lower()
    if vc in ("motorcycle", "bus", "truck"):
        return Attribute(value=vc, confidence=0.80, method=f"detector class '{vc}'")
    if not aspect_ratios:
        return Attribute(value="unknown", confidence=0.0, method="no vehicle boxes available")
    ar = float(median(aspect_ratios))
    if ar >= 2.0:
        val, alts = "sedan", ["hatchback", "SUV"]
    elif ar <= 1.15:
        val, alts = "SUV", ["hatchback", "sedan"]
    else:
        val, alts = "hatchback", ["sedan", "SUV"]
    return Attribute(value=val, confidence=0.45,
                     alternatives=[AttrAlt(value=a, confidence=0.25) for a in alts],
                     method=f"rule-based on detector class 'car' and median box width/height = {ar:.2f} (weak; tune on real clips)")


def extract_attributes(vision: VisionOutput, run_dir: str, hints: Optional[UserHints] = None) -> Attributes:
    colour = Attribute(value="unknown", confidence=0.0, method="no vehicle crop available")
    if vision.vehicle_crop_path:
        img = cv2.imread(os.path.join(run_dir, vision.vehicle_crop_path))
        if img is not None:
            colour = colour_from_crop(img)
    ars = [(f.vehicle_box[2] - f.vehicle_box[0]) / max(1e-6, f.vehicle_box[3] - f.vehicle_box[1])
           for f in vision.frames if f.used and f.vehicle_box]
    btype = body_type(vision.vehicle_class, ars)
    if hints and hints.colour:
        colour = Attribute(value=hints.colour.strip().lower(), confidence=1.0, method="investigator hint")
    if hints and hints.body_type:
        btype = Attribute(value=hints.body_type.strip(), confidence=1.0, method="investigator hint")
    return Attributes(colour=colour, body_type=btype)
