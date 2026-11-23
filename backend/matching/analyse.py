"""Person B's single entry point: analyse(vision, registry_csv, settings, hints, run_dir) -> AnalysisResult."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List

from schema import AnalysisResult, Settings, UserHints, VisionOutput
from matching.attributes import extract_attributes
from matching.fusion import fuse_plate
from matching.registry import RegistryEntry, load_registry, normalise_plate
from matching.scoring import match


def _apply_plate_hint(registry: List[RegistryEntry], hint: str | None, warnings: List[str]) -> List[RegistryEntry]:
    pat = normalise_plate(hint, keep_unknown=True)
    if not pat:
        return registry
    kept = [e for e in registry if len(e.plate) == len(pat) and all(h == "?" or h == c for h, c in zip(pat, e.plate))]
    if not kept:
        warnings.append(f"Investigator's partial plate {pat} matches no registry entry; search was NOT constrained.")
        return registry
    warnings.append(f"Registry search constrained by investigator's partial plate {pat}: {len(kept)} of {len(registry)} entries kept.")
    return kept


def analyse(vision: VisionOutput, registry_csv: str, settings: Settings, hints: UserHints, run_dir: str) -> AnalysisResult:
    t0 = time.perf_counter()
    warnings: List[str] = list(vision.notes)
    if vision.track_id == -1 or not vision.frames:
        warnings.append("No vehicle was detected in this clip.")
    registry = load_registry(registry_csv)
    if not registry:
        warnings.append("Registry is empty.")
    registry = _apply_plate_hint(registry, hints.partial_plate, warnings)

    fused = fuse_plate(vision.frames, settings)
    t1 = time.perf_counter()
    attrs = extract_attributes(vision, run_dir, hints)
    t2 = time.perf_counter()
    verdict, candidates, weak = match(fused, attrs, registry, settings)
    t3 = time.perf_counter()

    if fused.length == 0:
        warnings.append("No usable plate readings in any frame.")
    return AnalysisResult(
        run_id=os.path.basename(os.path.normpath(run_dir)),
        created_at=datetime.now(timezone.utc).isoformat(),
        settings=settings, hints=hints, vision=vision, fusion=fused, attributes=attrs,
        verdict=verdict, candidates=candidates, weak_leads=weak, warnings=warnings,
        timings_s={"fusion": t1 - t0, "attributes": t2 - t1, "matching": t3 - t2},
    )
