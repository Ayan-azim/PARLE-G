"""LIVE PIPELINE INTEGRATION POINTS (kept apart from the mock loader).

Person A and Person B: edit ONLY the two import lines below when your modules are ready.
This file never contains vision or matching logic.
"""
from __future__ import annotations

from pathlib import Path
import sys

from schema import AnalysisResult, Settings, UserHints

VISION_IMPORT = "vision.new_model_pipeline"  # returns one VisionOutput per detected vehicle
ANALYSE_IMPORT = "matching.analyse"    # must expose analyse(vision, registry_csv, settings, hints, run_dir) -> AnalysisResult


def _ensure_pipeline_path() -> None:
    """Make the consolidated backend package importable from any launch path."""
    backend_root = Path(__file__).resolve().parents[2]
    root_str = str(backend_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


class PipelineNotReady(RuntimeError):
    pass


def _get(module: str, func: str):
    import importlib
    try:
        _ensure_pipeline_path()
        return getattr(importlib.import_module(module), func)
    except (ImportError, AttributeError) as e:
        raise PipelineNotReady(f"Live pipeline not available: could not load {module}.{func} ({e}). "
                               "Turn on 'Use mock data' or edit app/components/integration.py.") from e


def run_vision_step(video_path: Path, settings: Settings, run_dir: Path):
    return _get(VISION_IMPORT, "run_vision_batch")(str(video_path), settings, str(run_dir))


def analyse_step(vision, registry_csv: Path, settings: Settings, hints: UserHints, run_dir: Path) -> AnalysisResult:
    result = _get(ANALYSE_IMPORT, "analyse")(vision, str(registry_csv), settings, hints, str(run_dir))  # Person B
    return AnalysisResult.model_validate(result.model_dump() if hasattr(result, "model_dump") else result)
