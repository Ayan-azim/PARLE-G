"""Run Person B's analysis on a REAL vision.json from Person A.

    python scripts/run_on_vision.py runs/test_run_01/vision.json
Writes result.json next to vision.json and prints a summary.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from matching.analyse import analyse  # noqa: E402
from schema import Settings, UserHints, VisionOutput, save_result  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("vision_json")
ap.add_argument("--registry", default=os.path.join(ROOT, "data", "registry.csv"))
ap.add_argument("--min-support", type=float, default=None)
ap.add_argument("--min-share", type=float, default=None)
a = ap.parse_args()

settings = Settings()
if a.min_support is not None:
    settings.min_support = a.min_support
if a.min_share is not None:
    settings.min_share = a.min_share

run_dir = os.path.dirname(os.path.abspath(a.vision_json))
with open(a.vision_json, encoding="utf-8") as f:
    vision = VisionOutput.model_validate_json(f.read())
r = analyse(vision, a.registry, settings, UserHints(), run_dir)
save_result(r, os.path.join(run_dir, "result.json"))

print("per-frame readings:")
for fr in vision.frames:
    print(f"  frame {fr.frame_idx:>4} used={fr.used!s:<5} w={fr.quality_weight:.2f} raw='{fr.raw_text}'")
print(f"fused    : {r.fusion.pattern} ({r.fusion.known_count}/{r.fusion.length} known)")
print(f"notes    : {r.fusion.format_note}")
print(f"colour   : {r.attributes.colour.value} ({r.attributes.colour.confidence:.2f}); type: {r.attributes.body_type.value}")
print(f"verdict  : {r.verdict.headline} | {r.verdict.reason}")
for c in (r.candidates or r.weak_leads)[:3]:
    print(f"  #{c.rank} {c.plate} conf={c.confidence:.2f}")
print("saved    :", os.path.join(run_dir, "result.json"))
