"""
schema.py - the shared contract between Person A (vision), Person B (matching)
and Person C (UI / report).

RULES
  1. Everybody imports from here. Nobody redefines these structures.
  2. If you need a new field, add it here as Optional with a default, tell the team,
     and never rename or remove an existing field after hour 3.
  3. Unknown plate characters are ALWAYS the literal string "?" (UNKNOWN). Never guess.
  4. All confidences and scores are floats in [0, 1].
  5. All file paths are relative to the run folder: runs/<run_id>/...

DATA FLOW
  A: run_vision(video_path, settings)            -> VisionOutput
  B: analyse(vision, registry, settings, hints)  -> AnalysisResult   (embeds the VisionOutput)
  C: renders AnalysisResult in the UI and builds the report from it.
  C develops against mock/mock_result.json and mock/mock_no_match.json from hour 1.

Requires: pydantic>=2
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Protocol, Tuple

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
UNKNOWN = "?"

# (x1, y1, x2, y2) in pixels of the ORIGINAL video frame
Box = Tuple[float, float, float, float]

Score = Field(ge=0.0, le=1.0)


class _Base(BaseModel):
    # forbid typos: an unexpected key raises an error instead of silently disappearing
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------
# Settings and optional investigator hints (UI -> pipeline)
# ----------------------------------------------------------------------------
class Settings(_Base):
    sample_fps: float = 5.0                 # frames per second analysed
    max_clip_seconds: float = 30.0
    min_frame_sharpness: float = 60.0       # Laplacian variance; below -> frame skipped
    # fusion rules (B)
    min_support: float = 0.50               # a position is KNOWN only if weighted support >= this ...
    min_share: float = 0.60                 # ... AND the top char holds >= this share of the votes
    # matching rules (B)
    min_known_chars: int = 3                # fewer known chars -> "no confident match"
    no_match_threshold: float = 0.55        # best final_score below this -> "no confident match"
    min_margin: float = 0.10                # best final_score minus second best must be >= this
    weight_plate: float = 0.6
    weight_colour: float = 0.2
    weight_type: float = 0.2
    top_k: int = 5


class UserHints(_Base):
    """Optional input typed by the investigator. Empty means 'use only what the video shows'."""
    partial_plate: Optional[str] = None     # e.g. "CG04??1234" ('?' or '*' for unknown)
    colour: Optional[str] = None
    body_type: Optional[str] = None


# ----------------------------------------------------------------------------
# PERSON A output: what the vision pipeline saw
# ----------------------------------------------------------------------------
class CharReading(_Base):
    """OCR's best guess for ONE plate position in ONE frame, with its confidence."""
    char: str                               # one character (OCR's top guess, even if weak)
    conf: float = Score


class FrameReading(_Base):
    frame_idx: int
    timestamp_s: float
    sharpness: float                        # Laplacian variance of the plate crop (higher = sharper)
    quality_weight: float = Field(ge=0.0, le=1.0)  # normalised weight used in fusion; 0 if skipped
    used: bool                              # False when the frame was rejected
    skip_reason: Optional[str] = None
    vehicle_box: Optional[Box] = None
    plate_box: Optional[Box] = None
    plate_crop_path: Optional[str] = None
    enhanced_crop_path: Optional[str] = None
    raw_text: str = ""                      # concatenated top guesses, length == len(chars)
    chars: List[CharReading] = Field(default_factory=list)


class VisionOutput(_Base):
    clip_name: str
    duration_s: float
    fps: float
    frame_size: Tuple[int, int]             # (width, height)
    sampled_fps: float
    track_id: int                           # tracker id of the target vehicle
    target_selection_reason: str            # e.g. "longest track, largest vehicle box"
    vehicle_class: str                      # raw detector class: car, truck, bus, motorcycle
    vehicle_crop_path: Optional[str] = None
    frames: List[FrameReading]
    notes: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# PERSON B output: fusion, attributes, matching
# ----------------------------------------------------------------------------
class FusedChar(_Base):
    """Result of voting across frames for ONE plate position.

    votes[c]  = sum over used frames of  quality_weight * conf   for character c
    support   = votes[top] (since quality weights sum to 1; this is the weighted evidence for the top char)
    share     = votes[top] / sum(votes.values())
    status is "known" only if support >= settings.min_support and share >= settings.min_share.
    Otherwise char == UNKNOWN and status == "unknown". NEVER fill in an unknown character.
    """
    position: int                           # 0-based
    char: str                               # the character, or "?"
    status: Literal["known", "unknown"]
    confidence: float = Score               # = support
    share: float = Score
    top_candidate: Optional[str] = None     # leading char even when unknown (for the UI tooltip only)
    votes: Dict[str, float] = Field(default_factory=dict)


class FusedPlate(_Base):
    pattern: str                            # e.g. "CG04?B1?34"
    length: int
    known_count: int
    chars: List[FusedChar]
    frames_used: List[int]                  # frame_idx values
    format_note: Optional[str] = None       # e.g. "matches Indian format SS NN XX NNNN"


class AttrAlt(_Base):
    value: str
    confidence: float = Score


class Attribute(_Base):
    value: str
    confidence: float = Score
    alternatives: List[AttrAlt] = Field(default_factory=list)
    hex: Optional[str] = None               # colour swatch, e.g. "#E8E8EA"
    method: Optional[str] = None            # short human-readable "how we got this"


class Attributes(_Base):
    colour: Attribute
    body_type: Attribute
    make_model: Optional[Attribute] = None  # bonus feature; None when not run


class ScoreBreakdown(_Base):
    plate_score: float = Score
    colour_score: float = Score
    type_score: float = Score
    final_score: float = Score              # weight_plate*plate + weight_colour*colour + weight_type*type
    evidence_factor: float = Score          # 0.65 + 0.35 * known_count / plate_length
    margin_factor: float = Score            # top: min(1, gap_to_second/0.15); others: max(0, 1 - gap_to_top/0.40)


class CharMatch(_Base):
    position: int
    observed: str                           # fused char or "?"
    registry: str                           # char in the registry plate
    status: Literal["match", "confusable", "mismatch", "unknown"]


class Candidate(_Base):
    rank: int
    plate: str
    make: Optional[str] = None
    model: Optional[str] = None
    registry_colour: str
    registry_type: str
    confidence: float = Score               # = final_score * evidence_factor * margin_factor
    breakdown: ScoreBreakdown
    char_matches: List[CharMatch]
    reasons: List[str]                      # human-readable, shown verbatim in the UI and report


class RuleCheck(_Base):
    name: str                               # e.g. "minimum known characters"
    passed: bool
    detail: str


class Verdict(_Base):
    status: Literal["match", "no_confident_match"]
    headline: str                           # shown in the big banner
    top_confidence: Optional[float] = None
    reason: str
    rule_checks: List[RuleCheck]


class AnalysisResult(_Base):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str                         # ISO 8601
    settings: Settings
    hints: UserHints = Field(default_factory=UserHints)
    vision: VisionOutput
    fusion: FusedPlate
    attributes: Attributes
    verdict: Verdict
    candidates: List[Candidate]             # top-k, ranked; EMPTY when verdict is no_confident_match
    weak_leads: List[Candidate] = Field(default_factory=list)  # nearest entries, shown only as "weak leads"
    warnings: List[str] = Field(default_factory=list)
    timings_s: Dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Function contracts (stubs the pipeline must keep matching)
# ----------------------------------------------------------------------------
class RunVision(Protocol):
    def __call__(self, video_path: str, settings: Settings, run_dir: str) -> VisionOutput:
        """Person A. Writes crops into run_dir and returns relative paths."""


class Analyse(Protocol):
    def __call__(
        self,
        vision: VisionOutput,
        registry_csv: str,
        settings: Settings,
        hints: UserHints,
        run_dir: str,
    ) -> AnalysisResult:
        """Person B. Fuses frames, extracts attributes, matches and ranks."""


# ----------------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------------
def load_result(path: str) -> AnalysisResult:
    with open(path, "r", encoding="utf-8") as f:
        return AnalysisResult.model_validate_json(f.read())


def save_result(result: AnalysisResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:] or ["mock/mock_result.json", "mock/mock_no_match.json"]:
        r = load_result(p)
        print(f"OK  {p}: plate={r.fusion.pattern} verdict={r.verdict.status} candidates={len(r.candidates)}")
