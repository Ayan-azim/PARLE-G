"""Fake Person A's output so Person B (and C) can test end to end before the real pipeline exists.

Run from the repo root:
    python scripts/simulate_vision.py                      # all three scenarios
    python scripts/simulate_vision.py --scenario blurry    # one scenario
    python scripts/simulate_vision.py --min-support 0.4 --min-share 0.55   # tune fusion thresholds

Writes runs/sim_<scenario>/vehicle_crop.jpg and runs/sim_<scenario>/result.json
"""
from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from matching.analyse import analyse  # noqa: E402
from matching.attributes import PALETTE  # noqa: E402
from matching.registry import load_registry  # noqa: E402
from schema import (CharReading, FrameReading, Settings, UserHints, VisionOutput,  # noqa: E402
                    save_result)

CONFUSABLE_SWAP = {"B": "8", "8": "B", "0": "O", "O": "0", "5": "S", "S": "5", "1": "I", "I": "1", "2": "Z", "Z": "2"}
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"

SCENARIOS = {
    # clear plate: mostly sharp frames -> fully known plate, confident match
    "clear": dict(n_clean=5, n_blur=1, n_drop=1, n_skip=1, clean=(0.85, 0.97), blur=(0.30, 0.50), swap=0.30, source="unique"),
    # blurry plate: weak, noisy frames -> '?' marks; truth has a near-duplicate in the registry
    "blurry": dict(n_clean=2, n_blur=4, n_drop=1, n_skip=1, clean=(0.55, 0.75), blur=(0.25, 0.45), swap=0.50, source="neighbour"),
    # clear read of a plate that is NOT in the registry -> "No confident match" with weak leads
    "not_in_registry": dict(n_clean=5, n_blur=1, n_drop=0, n_skip=0, clean=(0.85, 0.97), blur=(0.30, 0.50), swap=0.30, source="absent"),
}

BOX_BY_TYPE = {  # (x1, y1, x2, y2); width/height drives the rule-based body type
    "sedan": (100, 100, 540, 300),       # 2.2
    "hatchback": (100, 100, 420, 300),   # 1.6
    "suv": (100, 100, 300, 300),         # 1.0
}


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else 99


def random_plate(rng: random.Random) -> str:
    return "CG%02d%s%04d" % (rng.randint(1, 30), "".join(rng.choice(LETTERS) for _ in range(2)), rng.randint(1, 9999))


def pick_truth(source: str, registry, rng: random.Random):
    """Returns (plate, colour, type) of the 'real' vehicle."""
    plates = [e.plate for e in registry]
    if source == "absent":
        while True:
            p = random_plate(rng)
            if all(hamming(p, q) >= 3 for q in plates):
                return p, "white", "sedan"
    pool = list(registry)
    rng.shuffle(pool)
    for e in pool:
        nearest = min((hamming(e.plate, q) for q in plates if q != e.plate), default=99)
        if (source == "unique" and nearest >= 3) or (source == "neighbour" and nearest == 1):
            return e.plate, e.colour, e.type
    return pool[0].plate, pool[0].colour, pool[0].type


def write_vehicle_crop(path: str, colour: str, rng: np.random.Generator) -> None:
    rgb = PALETTE.get(colour.lower(), PALETTE["grey"])
    img = np.full((300, 500, 3), rgb[::-1], dtype=np.uint8)            # BGR body
    cv2.rectangle(img, (60, 0), (440, 110), (50, 40, 35), -1)          # windows (top, avoided by the colour ROI)
    cv2.circle(img, (130, 285), 40, (15, 15, 15), -1)                  # wheels (bottom, avoided too)
    cv2.circle(img, (370, 285), 40, (15, 15, 15), -1)
    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def noisy_reading(truth: str, kind: str, cfg: dict, rng: random.Random):
    lo, hi = cfg["clean"] if kind in ("clean", "drop") else cfg["blur"]
    chars = list(truth)
    if kind == "blur":
        for i, c in enumerate(chars):
            if rng.random() < cfg["swap"]:
                if c in CONFUSABLE_SWAP and rng.random() < 0.7:
                    chars[i] = CONFUSABLE_SWAP[c]
                else:
                    chars[i] = rng.choice(LETTERS) if c.isalpha() else rng.choice("0123456789")
    word_conf = round(rng.uniform(lo, hi), 3)
    confs = [word_conf] * len(chars)          # word-level confidence, like A's real output
    if kind == "drop":
        j = rng.randrange(len(chars))
        del chars[j], confs[j]
    return chars, confs


def build_vision(scenario: str, cfg: dict, registry, rng: random.Random, nrng: np.random.Generator, run_dir: str):
    truth, colour, typ = pick_truth(cfg["source"], registry, rng)
    write_vehicle_crop(os.path.join(run_dir, "vehicle_crop.jpg"), colour, nrng)
    box = BOX_BY_TYPE.get(typ.lower(), BOX_BY_TYPE["hatchback"])
    vclass = typ.lower() if typ.lower() in ("truck", "bus", "motorcycle") else "car"

    kinds = ["clean"] * cfg["n_clean"] + ["blur"] * cfg["n_blur"] + ["drop"] * cfg["n_drop"] + ["skip"] * cfg["n_skip"]
    rng.shuffle(kinds)

    raw = []
    for i, kind in enumerate(kinds):
        idx = i * 6
        if kind == "skip":
            raw.append(dict(frame_idx=idx, timestamp_s=round(idx / 30, 3), sharpness=round(rng.uniform(15, 40), 1),
                            used=False, skip_reason="sharpness below min_frame_sharpness", vehicle_box=box,
                            raw_text="", chars=[]))
            continue
        chars, confs = noisy_reading(truth, kind, cfg, rng)
        if rng.random() < 0.3:                    # junk from the 'IND' strip
            chars, confs = list('IND') + chars, [confs[0]] * 3 + confs
        sharp = rng.uniform(40, 90) if kind == "blur" else rng.uniform(120, 220)
        raw.append(dict(frame_idx=idx, timestamp_s=round(idx / 30, 3), sharpness=round(sharp, 1), used=True,
                        vehicle_box=box, plate_box=(box[0] + 90, box[3] - 60, box[0] + 190, box[3] - 30),
                        raw_text="".join(chars), chars=[CharReading(char=c, conf=p) for c, p in zip(chars, confs)]))

    total = sum(r["sharpness"] for r in raw if r["used"]) or 1.0
    frames = [FrameReading(quality_weight=(r["sharpness"] / total if r["used"] else 0.0), **r) for r in raw]

    vision = VisionOutput(
        clip_name=f"sim_{scenario}.mp4", duration_s=6.0, fps=30.0, frame_size=(640, 360), sampled_fps=5.0,
        track_id=1, target_selection_reason="simulated: single vehicle", vehicle_class=vclass,
        vehicle_crop_path="vehicle_crop.jpg", frames=frames,
        notes=[f"SIMULATED data. True plate was {truth} ({colour} {typ})."],
    )
    return vision, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    ap.add_argument("--registry", default=os.path.join(ROOT, "data", "registry.csv"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-support", type=float, default=None)
    ap.add_argument("--min-share", type=float, default=None)
    args = ap.parse_args()

    settings = Settings()
    if args.min_support is not None:
        settings.min_support = args.min_support
    if args.min_share is not None:
        settings.min_share = args.min_share

    registry = load_registry(args.registry)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)
        run_dir = os.path.join(ROOT, "runs", f"sim_{name}")
        os.makedirs(run_dir, exist_ok=True)
        vision, truth = build_vision(name, SCENARIOS[name], registry, rng, nrng, run_dir)
        result = analyse(vision, args.registry, settings, UserHints(), run_dir)
        save_result(result, os.path.join(run_dir, "result.json"))

        print(f"\n=== {name} ===")
        print(f"true plate : {truth}")
        print(f"fused plate: {result.fusion.pattern}  ({result.fusion.known_count}/{result.fusion.length} known)")
        print(f"attributes : {result.attributes.colour.value} ({result.attributes.colour.confidence:.2f}), "
              f"{result.attributes.body_type.value} ({result.attributes.body_type.confidence:.2f})")
        print(f"verdict    : {result.verdict.headline}  | {result.verdict.reason}")
        for c in (result.candidates or result.weak_leads)[:3]:
            print(f"  #{c.rank} {c.plate}  conf={c.confidence:.2f}  final={c.breakdown.final_score:.2f}")
        print(f"saved      : runs/sim_{name}/result.json")


if __name__ == "__main__":
    main()
