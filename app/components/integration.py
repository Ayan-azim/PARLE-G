"""LIVE PIPELINE INTEGRATION POINTS (kept apart from the mock loader).

Person A and Person B: edit ONLY the two import lines below when your modules are ready.
This file never contains vision or matching logic.
"""
from __future__ import annotations

from pathlib import Path

from schema import AnalysisResult, Settings, UserHints

VISION_IMPORT = "vision.pipeline"      # exposes run_vision(video_path, settings, run_dir) -> VisionOutput
ANALYSE_IMPORT = "matching.analyse"    # exposes analyse(vision, registry_csv, settings, hints, run_dir) -> AnalysisResult


class PipelineNotReady(RuntimeError):
    pass


def _get(module: str, func: str):
    import importlib
    try:
        return getattr(importlib.import_module(module), func)
    except (ImportError, AttributeError) as e:
        raise PipelineNotReady(f"Live pipeline not available: could not load {module}.{func} ({e}). "
                               "Turn on 'Use mock data' or edit app/components/integration.py.") from e


def run_vision_step(video_path: Path, settings: Settings, run_dir: Path):
    return _get(VISION_IMPORT, "run_vision")(str(video_path), settings, str(run_dir))  # Person A


def analyse_step(vision, registry_csv: Path, settings: Settings, hints: UserHints, run_dir: Path) -> AnalysisResult:
    result = _get(ANALYSE_IMPORT, "analyse")(vision, str(registry_csv), settings, hints, str(run_dir))  # Person B
    return AnalysisResult.model_validate(result.model_dump() if hasattr(result, "model_dump") else result)


def get_vision(video_path: Path, settings: Settings, run_dir: Path):
    """Returns (VisionOutput, from_cache). Reuses runs/<run_id>/vision.json when it exists and is valid."""
    from schema import VisionOutput
    cache = Path(run_dir) / "vision.json"
    if cache.is_file():
        try:
            return VisionOutput.model_validate_json(cache.read_text(encoding="utf-8")), True
        except Exception:
            pass  # stale or invalid cache: recompute
    vision = run_vision_step(video_path, settings, run_dir)
    try:
        cache.write_text(vision.model_dump_json(indent=2), encoding="utf-8")
    except OSError:
        pass
    return vision, False
