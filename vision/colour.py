"""
colour.py - robust vehicle colour estimation for CCTV crops.

    from vision.colour import estimate_colour, aggregate_estimates, to_attribute_dict, colour_similarity

Design (why it differs from the HSV + per-crop gray-world version):
  * NO per-crop white balance. Gray-world on a vehicle crop removes the car's own colour
    (a red car is "balanced" towards grey). An optional SCENE gain from whole frames is provided instead.
  * CIELAB (perceptual lightness L, chroma C, hue angle h) instead of HSV thresholds.
  * Samples the painted lower body only (below the windows, above the wheels) and masks the plate,
    glare, and the crop edges; the box usually contains road and neighbouring vehicles.
  * Does NOT throw away the darkest half of the pixels (that turned black/dark cars into grey/silver).
  * Returns a full DISTRIBUTION over colours plus a RELIABILITY score, so the matcher can shrink a weak
    reading towards "no information" instead of letting it decide a ranking.
  * Aggregates across frames, down-weighting small, edge-cut and overlapped boxes.
  * Neutral colours (white/silver/grey/black) are SOFT: a borderline pixel is shared between neighbours, so
    exposure ambiguity shows up as an honest 50/50 instead of a confident wrong answer.
  * Optional use_plate_reference=True: the (white) plate fixes exposure and colour cast. Ignored for yellow
    or dark plates. On synthetic crops it raised accuracy a lot; verify on YOUR labelled crops.

All thresholds are constants near the top: tune them on 10-20 labelled crops of YOUR footage
(see the __main__ block, which saves a debug overlay showing exactly which pixels were used).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PALETTE = ["white", "silver", "grey", "black", "red", "orange", "yellow", "green", "blue"]
_IDX = {c: i for i, c in enumerate(PALETTE)}

# How easily one colour is mistaken for another (also usable by the registry matcher)
_SIMILAR = {
    frozenset(("white", "silver")): 0.5,
    frozenset(("silver", "grey")): 0.5,
    frozenset(("grey", "black")): 0.4,
    frozenset(("white", "grey")): 0.15,
    frozenset(("red", "orange")): 0.3,
    frozenset(("orange", "yellow")): 0.3,
    frozenset(("blue", "black")): 0.15,
}

# ---- tunable thresholds (CIELAB: L 0..100, chroma C, hue h in degrees) -----------------------------
NEUTRAL_C = 13.0     # chroma below this = achromatic (white/silver/grey/black)
DARK_L = 26.0        # below this lightness a pixel is black unless strongly coloured
DARK_C = 28.0        # chroma needed for a DARK pixel to count as coloured (dark blue/green/red)
WHITE_L = 72.0       # achromatic and lighter than this = white
SILVER_L = 52.0      # achromatic between SILVER_L and WHITE_L = silver; between DARK_L and SILVER_L = grey
GLARE_L = 98.5       # specular highlights are ignored
GLASS_ALLOWANCE = 0.12   # share of "black" pixels explained by windows/tyres/shadow when the car is not black
HUE_BINS = [         # (lo, hi, name) on the Lab hue circle; red wraps around 0
    (48.0, 75.0, "orange"),
    (75.0, 112.0, "yellow"),
    (112.0, 200.0, "green"),
    (200.0, 330.0, "blue"),
]                    # everything else (>=330 or <48) = red

# region of interest as fractions of the vehicle box: (y0, y1, x0, x1), and a reliability factor
ROI = {
    "car": (0.48, 0.84, 0.10, 0.90, 1.0),
    "bus": (0.40, 0.84, 0.08, 0.92, 0.9),
    "truck": (0.40, 0.84, 0.08, 0.92, 0.9),
    "two-wheeler": (0.30, 0.65, 0.25, 0.75, 0.5),   # tank/body is small; rider clothing contaminates
}


def colour_similarity(a: str, b: str) -> float:
    """1.0 same, up to 0.5 for commonly confused colours, else 0. Reusable by the matcher."""
    return 1.0 if a == b else _SIMILAR.get(frozenset((a, b)), 0.0)


def _similarity_matrix() -> np.ndarray:
    n = len(PALETTE)
    m = np.eye(n, dtype=np.float64)
    for i, a in enumerate(PALETTE):
        for j, b in enumerate(PALETTE):
            if i != j:
                m[i, j] = colour_similarity(a, b)
    return m / m.sum(axis=1, keepdims=True)


_SIM = _similarity_matrix()


@dataclass
class ColourEstimate:
    distribution: Dict[str, float]
    value: str
    confidence: float      # top probability x reliability (what to show a human)
    reliability: float     # 0..1 trust in this reading regardless of which colour won
    hex: Optional[str] = None
    n_pixels: int = 0
    notes: List[str] = field(default_factory=list)


def scene_gain_from_frames(frames: Sequence[np.ndarray], clamp: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """OPTIONAL. Per-channel gain (BGR) from whole frames (gray-world over the entire scene, clamped).
    Use only if footage has a strong colour cast, and check it helps on your clips; a wrong gain hurts."""
    means = np.mean([f.reshape(-1, 3).mean(axis=0) for f in frames], axis=0)
    gain = means.mean() / np.maximum(means, 1.0)
    return np.clip(gain, *clamp).astype(np.float32)


# lightness prototypes for the neutral colours; a pixel between two prototypes is shared between them,
# so a borderline white/silver pixel gives roughly 50/50 instead of a hard (and over-confident) cliff
_NEUTRAL_CENTRES = [("black", 12.0), ("grey", 40.0), ("silver", 62.0), ("white", 85.0)]


def _pixel_memberships(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(H, W, K) soft membership of every pixel in every PALETTE colour (rows sum to 1)."""
    C = np.hypot(a, b)
    h = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    hue_label = np.full(L.shape, _IDX["red"], dtype=np.int64)          # default: red / magenta range
    for lo, hi, name in HUE_BINS:
        hue_label[(h >= lo) & (h < hi)] = _IDX[name]
    neutral = (C < NEUTRAL_C) | ((L < DARK_L) & (C < DARK_C))

    mem = np.zeros(L.shape + (len(PALETTE),), dtype=np.float32)
    np.put_along_axis(mem, hue_label[..., None], 1.0, axis=-1)          # coloured pixels: one-hot by hue
    mem[neutral] = 0.0
    centres = np.array([c for _, c in _NEUTRAL_CENTRES], dtype=np.float32)
    for k in range(len(_NEUTRAL_CENTRES) - 1):
        lo, hi = centres[k], centres[k + 1]
        t = np.clip((L - lo) / (hi - lo), 0.0, 1.0)
        in_seg = neutral & (L >= lo) & (L < hi)
        mem[..., _IDX[_NEUTRAL_CENTRES[k][0]]] += np.where(in_seg, 1.0 - t, 0.0)
        mem[..., _IDX[_NEUTRAL_CENTRES[k + 1][0]]] += np.where(in_seg, t, 0.0)
    mem[..., _IDX["black"]] += np.where(neutral & (L < centres[0]), 1.0, 0.0)
    mem[..., _IDX["white"]] += np.where(neutral & (L >= centres[-1]), 1.0, 0.0)
    return mem


def _plate_reference(L, A, B, plate_box):
    """Brightest part of the plate = (nearly) white reference. Returns (ref_L, ref_a, ref_b) or None.
    Rejected when the plate is dark, tiny or strongly coloured (yellow commercial plates)."""
    px1, py1, px2, py2 = [int(v) for v in plate_box]
    H, W = L.shape
    px1, py1, px2, py2 = max(0, px1), max(0, py1), min(W, px2), min(H, py2)
    if (px2 - px1) * (py2 - py1) < 60:
        return None
    pl, pa, pb = L[py1:py2, px1:px2].ravel(), A[py1:py2, px1:px2].ravel(), B[py1:py2, px1:px2].ravel()
    bright = pl >= np.percentile(pl, 60)
    ref_L, ref_a, ref_b = float(np.percentile(pl, 85)), float(np.median(pa[bright])), float(np.median(pb[bright]))
    if ref_L < 45.0 or np.hypot(ref_a, ref_b) > 28.0:
        return None
    return ref_L, ref_a, ref_b


def estimate_colour(
    v_crop: np.ndarray,
    vehicle_type: str = "car",
    plate_box_in_crop: Optional[Tuple[float, float, float, float]] = None,
    scene_gain: Optional[np.ndarray] = None,
    use_plate_reference: bool = False,
    debug_path: Optional[str] = None,
) -> ColourEstimate:
    """v_crop: BGR image of the vehicle box. plate_box_in_crop: (x1,y1,x2,y2) of its plate in crop pixels.
    use_plate_reference: treat the (white) plate as an exposure and colour-cast reference. Helps with dim,
    over-exposed and tinted footage; ignored for dark or strongly coloured (yellow) plates. A/B test it on
    labelled crops of your own footage before relying on it."""
    H, W = v_crop.shape[:2]
    if H < 24 or W < 24:
        return ColourEstimate({}, "unknown", 0.0, 0.0, None, 0, ["vehicle crop too small"])

    y0f, y1f, x0f, x1f, type_rel = ROI.get(vehicle_type, ROI["car"])
    img = v_crop.astype(np.float32)
    if scene_gain is not None:
        img = np.clip(img * scene_gain.reshape(1, 1, 3), 0, 255)

    # mask: True = usable pixel
    mask = np.zeros((H, W), dtype=bool)
    ry0, ry1, rx0, rx1 = int(H * y0f), int(H * y1f), int(W * x0f), int(W * x1f)
    mask[ry0:ry1, rx0:rx1] = True
    if plate_box_in_crop is not None:
        px1, py1, px2, py2 = plate_box_in_crop
        pw, ph = (px2 - px1) * 0.25, (py2 - py1) * 0.35
        mask[max(0, int(py1 - ph)):int(py2 + ph), max(0, int(px1 - pw)):int(px2 + pw)] = False

    lab = cv2.cvtColor((img / 255.0).astype(np.float32), cv2.COLOR_BGR2LAB)   # L 0..100, a/b ~ -127..127
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    notes: List[str] = []
    if use_plate_reference and plate_box_in_crop is not None:
        ref = _plate_reference(L, A, B, plate_box_in_crop)
        if ref is not None:
            ref_L, ref_a, ref_b = ref
            L = np.clip(L * float(np.clip(88.0 / ref_L, 0.7, 1.5)), 0.0, 100.0)
            A, B = A - ref_a, B - ref_b
            notes.append("plate used as exposure/cast reference")

    roi_px = max(1, (ry1 - ry0) * (rx1 - rx0))
    glare = mask & (L > GLARE_L)
    if glare.sum() > 0.5 * roi_px:                 # mostly bright: over-exposed paint, not specular glare
        notes.append("over-exposed region")
        glare = np.zeros_like(glare)
    mask &= ~glare
    n_valid = int(mask.sum())
    if n_valid < 120:
        return ColourEstimate({}, "unknown", 0.0, 0.0, None, n_valid, notes + ["too few usable pixels"])

    mem = _pixel_memberships(L, A, B)
    counts = mem[mask].sum(axis=0).astype(np.float64)
    share = counts / counts.sum()

    # windows/tyres/shadow can fall inside the region: do not let them decide for a non-black car
    black = _IDX["black"]
    if share[black] < 0.6:
        share[black] = max(0.0, share[black] - GLASS_ALLOWANCE)
        if share.sum() <= 0:
            return ColourEstimate({}, "unknown", 0.0, 0.0, None, n_valid, notes + ["nothing left after masking"])
        share /= share.sum()

    top = int(np.argmax(share))
    glare_frac = float(glare.sum()) / roi_px
    r_pix = min(1.0, n_valid / 4000.0)
    r_size = float(np.clip((min(H, W) - 40) / 120.0, 0.2, 1.0))
    r_purity = 0.5 + 0.5 * float(share[top])
    r_glare = 1.0 - min(0.5, glare_frac)
    reliability = float(np.clip(r_pix * r_size * r_purity * r_glare * type_rel, 0.0, 1.0))

    lam = 0.10 + 0.25 * (1.0 - reliability)          # smooth towards commonly confused colours
    dist_vec = (1 - lam) * share + lam * (share @ _SIM)
    dist_vec /= dist_vec.sum()
    distribution = {PALETTE[i]: round(float(p), 4) for i, p in enumerate(dist_vec) if p >= 0.005}
    value = max(distribution, key=distribution.get)

    sel = mask & (np.argmax(mem, axis=-1) == _IDX[value])
    hex_val = None
    if sel.any():
        b_, g_, r_ = img[sel].mean(axis=0)
        hex_val = "#{:02X}{:02X}{:02X}".format(int(r_), int(g_), int(b_))

    if glare_frac > 0.15:
        notes.append("strong glare in sampled region")
    if min(H, W) < 80:
        notes.append("small vehicle crop")

    if debug_path:
        dbg = v_crop.copy()
        tint = dbg.copy()
        tint[mask] = (0, 255, 0)
        dbg = cv2.addWeighted(dbg, 0.6, tint, 0.4, 0)
        cv2.rectangle(dbg, (rx0, ry0), (rx1, ry1), (255, 0, 0), 2)
        cv2.putText(dbg, f"{value} {distribution[value]:.2f} rel={reliability:.2f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(debug_path, dbg)

    return ColourEstimate(distribution, value, round(distribution[value] * reliability, 4),
                          round(reliability, 4), hex_val, n_valid, notes)


def aggregate_estimates(items: Sequence[Tuple[ColourEstimate, float, float]]) -> ColourEstimate:
    """items: (estimate, box_area_px, penalty) per frame of ONE track. penalty in (0,1]: use ~0.3 when the box
    touches the frame border or overlaps another vehicle by more than ~10% of its area, else 1.0."""
    usable = [(e, a, p) for e, a, p in items if e.distribution and e.reliability > 0]
    if not usable:
        return ColourEstimate({}, "unknown", 0.0, 0.0, None, 0, ["no usable frames"])
    weights = np.array([e.reliability * np.sqrt(max(a, 1.0)) * p for e, a, p in usable], dtype=np.float64)
    weights /= weights.sum()
    agg = np.zeros(len(PALETTE))
    for w, (e, _, _) in zip(weights, usable):
        for name, prob in e.distribution.items():
            agg[_IDX[name]] += w * prob
    agg /= agg.sum()
    distribution = {PALETTE[i]: round(float(p), 4) for i, p in enumerate(agg) if p >= 0.005}
    value = max(distribution, key=distribution.get)
    reliability = float(sum(w * e.reliability for w, (e, _, _) in zip(weights, usable)))
    hexes = [e.hex for e, _, _ in usable if e.hex and e.value == value]
    return ColourEstimate(distribution, value, round(distribution[value] * reliability, 4),
                          round(reliability, 4), hexes[0] if hexes else None,
                          int(sum(e.n_pixels for e, _, _ in usable)), [f"aggregated over {len(usable)} frames"])


def to_attribute_dict(est: ColourEstimate) -> dict:
    """Fields for schema.Attribute(**d): value, confidence, alternatives, hex, method."""
    alts = [{"value": k, "confidence": round(v, 3)} for k, v in
            sorted(est.distribution.items(), key=lambda kv: -kv[1]) if k != est.value][:3]
    return {
        "value": est.value,
        "confidence": float(min(1.0, max(0.0, est.confidence))),
        "alternatives": alts,
        "hex": est.hex,
        "method": f"CIELAB pixel classification of the lower body (windows, plate, glare masked); "
                  f"reliability {est.reliability:.2f}",
    }


if __name__ == "__main__":
    # python -m vision.colour crop1.jpg [crop2.jpg ...] [--type car|two-wheeler|bus|truck]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    vt = "car"
    if "--type" in sys.argv:
        vt = sys.argv[sys.argv.index("--type") + 1]
        args = [a for a in args if a != vt]
    for path in args:
        img = cv2.imread(path)
        if img is None:
            print(f"cannot read {path}")
            continue
        out = path.rsplit(".", 1)[0] + "_colour_debug.jpg"
        e = estimate_colour(img, vt, debug_path=out)
        top3 = sorted(e.distribution.items(), key=lambda kv: -kv[1])[:3]
        print(f"{path}: {e.value} conf={e.confidence:.2f} reliability={e.reliability:.2f} top3={top3}  debug={out}")