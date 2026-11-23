"""Cross-video vehicle matching (Person B).

Input: one VisionOutput per tracked vehicle per clip (what A produces today: one VisionOutput = one
vehicle in one clip). Output: identities = groups of sightings we believe are the same vehicle, the
evidence for each link, and a pooled plate + registry match per identity.

Honesty rules (same spirit as the single-clip pipeline):
  * No link without READABLE PLATE evidence. Colour/type alone never join two sightings.
  * Contradicting plates (2+ hard mismatches) can never be merged, even through a third sighting.
  * Two vehicles visible in the same clip at the same time are never the same vehicle.
  * Only "likely_same" links merge sightings. "possible_same" is reported as a lead, not merged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from schema import Attribute, Attributes, FusedPlate, Settings, VisionOutput
from matching.align import CONFUSABLE, align
from matching.fusion import fuse_plate
from matching.registry import RegistryEntry, load_registry
from matching.scoring import colour_score, match, type_score

MIN_OVERLAP_POSSIBLE = 3          # known-in-both plate positions needed for any link at all
MIN_OVERLAP_LIKELY = 6            # ... needed for "likely_same"
SINGLE_MISMATCH_MIN_OVERLAP = 5   # one differing char is only discussed if enough else agrees
CONTRADICTION_MISMATCHES = 2      # this many hard mismatches => different plates
SAME_CLIP_OVERLAP_S = 0.3         # seconds two tracks of one clip may overlap (tracker fragments)
TYPE_VETO_CONF = 0.7              # detector-class types (motorcycle/bus/truck) are reliable enough to veto
COLOUR_CLASH_CONF = 0.5


@dataclass
class Observation:
    obs_id: str
    clip_name: str
    track_id: int
    vision: VisionOutput
    run_dir: str
    fusion: FusedPlate
    attributes: Attributes
    t_start: float
    t_end: float


@dataclass
class Link:
    a: str
    b: str
    verdict: str                  # likely_same | possible_same | different | insufficient_evidence
    score: float
    overlap: int
    reasons: List[str] = field(default_factory=list)


@dataclass
class Identity:
    identity_id: str
    observation_ids: List[str]
    clips: List[str]
    fusion: FusedPlate            # plate pooled over every sighting
    attributes: Attributes        # pooled colour / body type
    verdict: Any                  # schema.Verdict from the registry match on the pooled plate
    candidates: list
    weak_leads: list
    member_patterns: Dict[str, str]   # obs_id -> its single-sighting pattern (shows what pooling gained)


@dataclass
class CrossClipResult:
    identities: List[Identity]
    links: List[Link]
    possible_links: List[Link]
    warnings: List[str]


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def make_observation(vision: VisionOutput, run_dir: str, settings: Settings, attributes: Attributes,
                     obs_id: Optional[str] = None) -> Optional[Observation]:
    if not vision.frames or vision.track_id == -1:
        return None
    ts = [f.timestamp_s for f in vision.frames]
    return Observation(
        obs_id=obs_id or f"{vision.clip_name}#t{vision.track_id}", clip_name=vision.clip_name,
        track_id=vision.track_id, vision=vision, run_dir=run_dir,
        fusion=fuse_plate(vision.frames, settings), attributes=attributes,
        t_start=min(ts), t_end=max(ts),
    )


def plate_agreement(fa: FusedPlate, fb: FusedPlate) -> dict:
    """Align two fused plates and compare positions that are KNOWN in both."""
    _, pairs = align(fa.pattern, fb.pattern)
    m = c = x = 0
    num = den = 0.0
    bad_pos: List[int] = []
    for i, j in pairs:
        if i is None or j is None:
            continue
        ca, cb = fa.chars[i], fb.chars[j]
        if ca.status != "known" or cb.status != "known":
            continue
        w = min(ca.confidence, cb.confidence)
        if ca.char == cb.char:
            m += 1; num += w
        elif frozenset((ca.char, cb.char)) in CONFUSABLE:
            c += 1; num += 0.6 * w
        else:
            x += 1; num -= w; bad_pos.append(ca.position + 1)
        den += w
    return {"overlap": m + c + x, "match": m, "confusable": c, "mismatch": x, "bad_positions": bad_pos,
            "score": max(0.0, min(1.0, num / den)) if den > 0 else 0.0}


def _time_overlap(a: Observation, b: Observation) -> float:
    return max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))


def compare(a: Observation, b: Observation) -> Link:
    def link(verdict, score, overlap, reasons):
        return Link(a.obs_id, b.obs_id, verdict, round(score, 4), overlap, reasons)

    if a.clip_name == b.clip_name:
        ov = _time_overlap(a, b)
        if ov > SAME_CLIP_OVERLAP_S:
            return link("different", 0.0, 0, [f"Both are visible in {a.clip_name} at the same time "
                                               f"({ov:.1f}s overlap), so they are different vehicles"])
    ta, tb = a.attributes.body_type, b.attributes.body_type
    if ta.confidence >= TYPE_VETO_CONF and tb.confidence >= TYPE_VETO_CONF and type_score(ta.value, tb.value) == 0:
        return link("different", 0.0, 0, [f"Vehicle type differs: {ta.value} vs {tb.value}"])

    ag = plate_agreement(a.fusion, b.fusion)
    ov = ag["overlap"]
    reasons = [f"{ov} plate position(s) readable in both: {ag['match']} match, {ag['confusable']} confusable, "
               f"{ag['mismatch']} mismatch"]
    score = ag["score"] * min(1.0, ov / 8.0)

    if ov < MIN_OVERLAP_POSSIBLE:
        reasons.append("Too few plate characters readable in both sightings; colour and type alone are not "
                       "enough to link them")
        return link("insufficient_evidence", 0.0, ov, reasons)
    if ag["mismatch"] >= CONTRADICTION_MISMATCHES:
        reasons.append("Plates contradict each other at position(s) " + ", ".join(map(str, ag["bad_positions"])))
        return link("different", 0.0, ov, reasons)
    if ag["mismatch"] == 1:
        if ov < SINGLE_MISMATCH_MIN_OVERLAP:
            reasons.append("One character differs and too little else is readable to decide")
            return link("insufficient_evidence", 0.0, ov, reasons)
        reasons.append(f"One character differs (position {ag['bad_positions'][0]}): could be a different vehicle "
                       f"with a near-identical plate")
        return link("possible_same", score, ov, reasons)

    verdict = "likely_same" if ov >= MIN_OVERLAP_LIKELY else "possible_same"
    if verdict == "possible_same":
        reasons.append(f"Plates agree but only {ov} positions are readable in both (need {MIN_OVERLAP_LIKELY})")

    ca, cb = a.attributes.colour, b.attributes.colour
    cs = colour_score(ca.value, cb.value)
    if cs == 1.0:
        reasons.append(f"Colour {ca.value} = {cb.value}")
    elif cs == 0.0 and ca.confidence >= COLOUR_CLASH_CONF and cb.confidence >= COLOUR_CLASH_CONF:
        reasons.append(f"Colour estimates disagree ({ca.value} vs {cb.value}); treated as only a possible link")
        verdict = "possible_same"
    else:
        reasons.append(f"Colour {ca.value} ~ {cb.value} (uncertain)")
    return link(verdict, score, ov, reasons)


def _pool_attribute(attrs: List[Attribute], what: str) -> Attribute:
    totals: Dict[str, float] = defaultdict(float)
    for a in attrs:
        if a.value != "unknown":
            totals[a.value] += a.confidence
    if not totals:
        return Attribute(value="unknown", confidence=0.0, method=f"{what}: no usable sightings")
    best = max(totals, key=totals.get)
    top = max((a for a in attrs if a.value == best), key=lambda a: a.confidence)
    return Attribute(value=best, confidence=min(1.0, totals[best] / len(attrs)), hex=getattr(top, "hex", None),
                     method=f"{what} pooled over {len(attrs)} sighting(s), confidence-weighted vote")


def _pool_fusion(group: List[Observation], settings: Settings) -> FusedPlate:
    frames, weights = [], []
    for o in group:
        for f in o.vision.frames:
            if f.used and f.chars:
                frames.append(f)
                weights.append(max(float(f.sharpness), 0.0))     # sharper frames count more, across clips too
    return fuse_plate(frames, settings, weights)


def build_identities(observations: List[Observation], registry: List[RegistryEntry],
                     settings: Settings) -> CrossClipResult:
    links: Dict[Tuple[str, str], Link] = {}
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            l = compare(observations[i], observations[j])
            links[tuple(sorted((l.a, l.b)))] = l

    members: Dict[int, List[str]] = {k: [o.obs_id] for k, o in enumerate(observations)}
    group_of = {o.obs_id: k for k, o in enumerate(observations)}
    warnings: List[str] = []
    for l in sorted((l for l in links.values() if l.verdict == "likely_same"), key=lambda l: (-l.score, -l.overlap)):
        ga, gb = group_of[l.a], group_of[l.b]
        if ga == gb:
            continue
        clash = [(x, y) for x in members[ga] for y in members[gb]
                 if links[tuple(sorted((x, y)))].verdict == "different"]
        if clash:
            warnings.append(f"{l.a} and {l.b} look alike but were kept apart because {clash[0][0]} and "
                            f"{clash[0][1]} contradict each other.")
            continue
        for oid in members[gb]:
            group_of[oid] = ga
        members[ga].extend(members.pop(gb))

    by_id = {o.obs_id: o for o in observations}
    identities: List[Identity] = []
    order = sorted(members.values(), key=lambda ids: min(observations.index(by_id[i]) for i in ids))
    for n, ids in enumerate(order, start=1):
        group = [by_id[i] for i in sorted(ids, key=lambda i: observations.index(by_id[i]))]
        fused = _pool_fusion(group, settings)
        attrs = Attributes(colour=_pool_attribute([o.attributes.colour for o in group], "colour"),
                           body_type=_pool_attribute([o.attributes.body_type for o in group], "body type"))
        verdict, cands, weak = match(fused, attrs, registry, settings)
        identities.append(Identity(
            identity_id=f"V{n}", observation_ids=[o.obs_id for o in group],
            clips=sorted({o.clip_name for o in group}), fusion=fused, attributes=attrs,
            verdict=verdict, candidates=cands, weak_leads=weak,
            member_patterns={o.obs_id: o.fusion.pattern for o in group},
        ))

    possible = [l for l in links.values() if l.verdict == "possible_same" and group_of[l.a] != group_of[l.b]]
    return CrossClipResult(identities=identities, links=list(links.values()), possible_links=possible,
                           warnings=warnings)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def analyse_clips(items: List[Tuple[VisionOutput, str]], registry_csv: str, settings: Settings) -> CrossClipResult:
    """items: [(VisionOutput of one tracked vehicle, run_dir that its crop paths are relative to), ...]."""
    from matching.attributes import extract_attributes      # needs cv2 / scikit-*; kept lazy

    registry = load_registry(registry_csv)
    observations: List[Observation] = []
    warnings: List[str] = []
    seen: Dict[str, int] = defaultdict(int)
    for vision, run_dir in items:
        attrs = extract_attributes(vision, run_dir)
        base = f"{vision.clip_name}#t{vision.track_id}"
        seen[base] += 1
        obs_id = base if seen[base] == 1 else f"{base}~{seen[base]}"
        o = make_observation(vision, run_dir, settings, attrs, obs_id)
        if o is None:
            warnings.append(f"{vision.clip_name}: no vehicle detected; skipped.")
        else:
            observations.append(o)
    res = build_identities(observations, registry, settings)
    res.warnings = warnings + res.warnings
    return res


def to_jsonable(x: Any) -> Any:
    """Turn a CrossClipResult (dataclasses holding pydantic models) into plain JSON-ready data."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if is_dataclass(x):
        return {f.name: to_jsonable(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x
