"""Multi-frame plate fusion (Person B).

Real OCR output varies in length per frame and may carry junk (e.g. the "IND" strip), so we
cannot vote by raw index. Steps:
  1. clean each reading: strip a leading "IND", trim to the longest span that fits the Indian
     plate regex (state 2 letters, RTO 1-2 digits, series 0-3 letters, number 4 digits);
  2. pick a reference reading (weighted medoid by edit distance, preferring regex-valid ones);
  3. align every reading to the reference (confusable-aware edit distance); grossly different
     readings are rejected; characters inserted relative to the reference are dropped;
  4. vote per reference column: votes[c] += weight * conf (weights renormalised to sum to 1).
A column is KNOWN only if support >= min_support AND share >= min_share, else "?".
A frame that has no character for a column adds no votes, so support honestly drops.
The plate format is only a soft prior when choosing between competing candidates.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from schema import UNKNOWN, CharReading, FrameReading, FusedChar, FusedPlate, Settings
from matching.align import align

PLATE_RE = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}")
PLATE_PARTS = re.compile(r"([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})")
MIN_TRIM_LEN = 8
REJECT_DISTANCE = 0.5          # normalised edit distance above which a frame is rejected
INVALID_CLASS_PENALTY = 0.6


@dataclass
class _Reading:
    frame_idx: int
    chars: List[CharReading]
    text: str
    weight: float
    trimmed: bool


def _prepare(f: FrameReading) -> _Reading:
    chars = list(f.chars)
    text = "".join(c.char.upper() for c in chars)
    trimmed = False
    if text.startswith("IND") and len(text) - 3 >= MIN_TRIM_LEN:
        chars, text, trimmed = chars[3:], text[3:], True
    spans = [m for m in PLATE_RE.finditer(text) if m.end() - m.start() >= MIN_TRIM_LEN]
    if spans:
        m = max(spans, key=lambda m: m.end() - m.start())
        if (m.start(), m.end()) != (0, len(text)):
            chars, text, trimmed = chars[m.start():m.end()], text[m.start():m.end()], True
    return _Reading(f.frame_idx, chars, text, f.quality_weight, trimmed)


def _classes(ref_text: str) -> Optional[str]:
    m = PLATE_PARTS.fullmatch(ref_text)
    if not m:
        return None
    return "L" * len(m[1]) + "D" * len(m[2]) + "L" * len(m[3]) + "D" * len(m[4])


def _valid(ch: str, cls: Optional[str]) -> bool:
    if cls is None:
        return True
    return ch.isalpha() if cls == "L" else ch.isdigit()


def fuse_plate(frames: List[FrameReading], settings: Settings) -> FusedPlate:
    usable = [f for f in frames if f.used and f.quality_weight > 0 and f.chars]
    if not usable:
        return FusedPlate(pattern="", length=0, known_count=0, chars=[], frames_used=[],
                          format_note="no usable frames")
    rs = [_prepare(f) for f in usable]

    def dist(a: _Reading, b: _Reading) -> float:
        cost, _ = align(a.text, b.text)
        return cost / max(len(a.text), len(b.text), 1)

    pool = [r for r in rs if PLATE_RE.fullmatch(r.text)] or rs
    ref = min(pool, key=lambda c: sum(o.weight * dist(c, o) for o in rs))

    kept = []
    rejected = 0
    for r in rs:
        cost, pairs = align(ref.text, r.text)
        if r is ref or cost / max(len(ref.text), len(r.text)) <= REJECT_DISTANCE:
            kept.append((r, pairs))
        else:
            rejected += 1
    total_w = sum(r.weight for r, _ in kept)

    length = len(ref.text)
    cols: List[Dict[str, float]] = [defaultdict(float) for _ in range(length)]
    for r, pairs in kept:
        w = r.weight / total_w
        for i, j in pairs:
            if i is not None and j is not None:
                c = r.chars[j]
                cols[i][c.char.upper()] += w * c.conf

    cls_all = _classes(ref.text)
    fused: List[FusedChar] = []
    violations = 0
    for pos, votes in enumerate(cols):
        total_votes = sum(votes.values())
        if total_votes <= 0:
            fused.append(FusedChar(position=pos, char=UNKNOWN, status="unknown", confidence=0.0, share=0.0, votes={}))
            continue
        cls = cls_all[pos] if cls_all else None
        top = max(votes, key=lambda ch: votes[ch] * (1.0 if _valid(ch, cls) else INVALID_CLASS_PENALTY))
        support = min(1.0, votes[top])
        share = min(1.0, votes[top] / total_votes)
        known = support >= settings.min_support and share >= settings.min_share
        if known and not _valid(top, cls):
            violations += 1
        fused.append(FusedChar(
            position=pos, char=top if known else UNKNOWN, status="known" if known else "unknown",
            confidence=support, share=share, top_candidate=top,
            votes={k: round(v, 6) for k, v in votes.items()},
        ))

    notes = ["readings aligned by confusable-aware edit distance to a reference frame"]
    notes.append("reference fits Indian format (state, RTO, series, number)" if cls_all
                 else "no reading fits the Indian plate format; format prior not applied")
    if violations:
        notes.append(f"{violations} known char(s) outside the expected letter/digit class (kept as read)")
    trimmed = sum(r.trimmed for r, _ in kept)
    if trimmed:
        notes.append(f"{trimmed} frame(s) had junk characters trimmed (e.g. leading 'IND')")
    if rejected:
        notes.append(f"{rejected} frame(s) rejected as too different from the reference reading")
    return FusedPlate(
        pattern="".join(c.char for c in fused), length=length, known_count=sum(c.status == "known" for c in fused),
        chars=fused, frames_used=[r.frame_idx for r, _ in kept], format_note="; ".join(notes),
    )
