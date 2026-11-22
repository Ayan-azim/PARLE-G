"""Multi-frame plate fusion (Person B).

Position-wise weighted voting. Never invents characters: a position is KNOWN only if
support >= settings.min_support AND share >= settings.min_share, otherwise it is "?".

Indian format (SS NN X{1,3} NNNN) is used as a SOFT prior only when choosing between
competing candidates at a position (a format-valid char needs only 60% of the votes of
an invalid one to win). It never creates a character that OCR did not read.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from schema import UNKNOWN, FrameReading, FusedChar, FusedPlate, Settings

INDIAN_LENGTHS = (9, 10, 11)
INVALID_CLASS_PENALTY = 0.6


def expected_class(pos: int, length: int) -> Optional[str]:
    """'L' letter, 'D' digit, None if the length doesn't fit the Indian format."""
    if length not in INDIAN_LENGTHS:
        return None
    if pos < 2:
        return "L"
    if pos < 4:
        return "D"
    if pos >= length - 4:
        return "D"
    return "L"


def _valid(ch: str, cls: Optional[str]) -> bool:
    if cls is None:
        return True
    return ch.isalpha() if cls == "L" else ch.isdigit()


def fuse_plate(frames: List[FrameReading], settings: Settings) -> FusedPlate:
    usable = [f for f in frames if f.used and f.quality_weight > 0 and f.chars]
    if not usable:
        return FusedPlate(pattern="", length=0, known_count=0, chars=[], frames_used=[],
                          format_note="no usable frames")

    # 1. vote on plate length, drop frames that disagree
    len_weight: Dict[int, float] = defaultdict(float)
    len_count: Dict[int, int] = defaultdict(int)
    for f in usable:
        len_weight[len(f.chars)] += f.quality_weight
        len_count[len(f.chars)] += 1
    length = max(len_weight, key=lambda n: (len_weight[n], len_count[n]))
    kept = [f for f in usable if len(f.chars) == length]
    dropped = len(usable) - len(kept)

    # 2. renormalise quality weights over the kept frames so they sum to 1
    total_w = sum(f.quality_weight for f in kept)
    weights = [f.quality_weight / total_w for f in kept]

    # 3. position-wise voting
    fused: List[FusedChar] = []
    violations = 0
    for pos in range(length):
        votes: Dict[str, float] = defaultdict(float)
        for f, w in zip(kept, weights):
            c = f.chars[pos]
            votes[c.char.upper()] += w * c.conf
        total_votes = sum(votes.values())
        if total_votes <= 0:
            fused.append(FusedChar(position=pos, char=UNKNOWN, status="unknown", confidence=0.0, share=0.0, votes={}))
            continue
        cls = expected_class(pos, length)
        top = max(votes, key=lambda ch: votes[ch] * (1.0 if _valid(ch, cls) else INVALID_CLASS_PENALTY))
        support = min(1.0, votes[top])
        share = min(1.0, votes[top] / total_votes)
        known = support >= settings.min_support and share >= settings.min_share
        if known and not _valid(top, cls):
            violations += 1
        fused.append(FusedChar(
            position=pos,
            char=top if known else UNKNOWN,
            status="known" if known else "unknown",
            confidence=support,
            share=share,
            top_candidate=top,
            votes={k: round(v, 6) for k, v in votes.items()},
        ))

    pattern = "".join(c.char for c in fused)
    notes = []
    if length in INDIAN_LENGTHS:
        notes.append("length fits Indian format SS NN X NNNN")
        if violations:
            notes.append(f"{violations} known char(s) outside the expected letter/digit class (kept as read; confusable matching applies)")
    else:
        notes.append("length does not fit Indian format; format prior not applied")
    if dropped:
        notes.append(f"{dropped} frame(s) dropped for disagreeing on plate length")
    return FusedPlate(
        pattern=pattern, length=length, known_count=sum(c.status == "known" for c in fused),
        chars=fused, frames_used=[f.frame_idx for f in kept], format_note="; ".join(notes),
    )
