"""Registry scoring, ranking, confidence and the no-match rule (Person B)."""
from __future__ import annotations

from typing import List, Tuple

from schema import (Attributes, Candidate, CharMatch, FusedPlate, RuleCheck, ScoreBreakdown,
                    Settings, Verdict)
from matching.align import CONFUSABLE, align
from matching.registry import RegistryEntry

CONFUSABLE_CREDIT = 0.6
NEAR_COLOURS = {"white", "silver", "grey"}
CAR_LIKE = {"hatchback", "sedan", "suv", "car", "muv"}


def _colour(c: str) -> str:
    c = c.strip().lower()
    return "grey" if c == "gray" else c


def colour_score(observed: str, registry: str) -> float:
    o, r = _colour(observed), _colour(registry)
    if o == r:
        return 1.0
    return 0.5 if (o in NEAR_COLOURS and r in NEAR_COLOURS) else 0.0


def type_score(observed: str, registry: str) -> float:
    o, r = observed.strip().lower(), registry.strip().lower()
    if o == r:
        return 1.0
    return 0.5 if (o in CAR_LIKE and r in CAR_LIKE) else 0.0


def compare_plate(fused: FusedPlate, plate: str) -> Tuple[float, List[CharMatch], int]:
    """Align the fused pattern ('?' = unknown) to the registry plate, then score KNOWN columns:
    plate_score = sum(sign*conf)/sum(conf), floored at 0. sign: +1 match, +0.6 confusable,
    -1 mismatch (a known char with no registry counterpart also counts as a mismatch).
    Unknown columns are ignored. Returns (score, matches, registry_chars_without_counterpart)."""
    _, pairs = align(fused.pattern, plate)
    matches: List[CharMatch] = []
    num = den = 0.0
    orphans = 0
    for i, j in pairs:
        if i is None:
            orphans += 1
            continue
        fc = fused.chars[i]
        reg = plate[j] if j is not None else ""
        if fc.status == "unknown":
            status = "unknown"
        elif not reg:
            status = "mismatch"
        elif fc.char == reg:
            status = "match"
        elif frozenset((fc.char, reg)) in CONFUSABLE:
            status = "confusable"
        else:
            status = "mismatch"
        matches.append(CharMatch(position=fc.position, observed=fc.char, registry=reg or "-", status=status))
        if status != "unknown":
            sign = {"match": 1.0, "confusable": CONFUSABLE_CREDIT, "mismatch": -1.0}[status]
            num += sign * fc.confidence
            den += fc.confidence
    if den == 0:
        return 0.0, matches, orphans
    return max(0.0, min(1.0, num / den)), matches, orphans


def _reasons(fused, matches, orphans, cs, ts, obs_colour, obs_type, entry) -> List[str]:
    known = [m for m in matches if m.status != "unknown"]
    exact = sum(m.status == "match" for m in known)
    conf = [m for m in known if m.status == "confusable"]
    bad = [m for m in known if m.status == "mismatch"]
    out = []
    out.append(f"{exact + len(conf)} of {len(known)} known characters match")
    for m in conf:
        out.append(f"Confusable pair at position {m.position + 1}: read {m.observed}, registry {m.registry}")
    if bad:
        out.append("Mismatch at position " + ", ".join(str(m.position + 1) for m in bad))
    if orphans:
        out.append(f"{orphans} registry character(s) had no counterpart in the reading")
    unk = len(matches) - len(known)
    if unk:
        out.append(f"{unk} unreadable position(s) ignored, neither helping nor hurting")
    out.append(f"Colour {obs_colour} {'=' if cs == 1 else '≈' if cs == 0.5 else '≠'} {entry.colour}")
    out.append(f"Type {obs_type} {'=' if ts == 1 else '≈' if ts == 0.5 else '≠'} {entry.type}")
    return out


def _confidences(finals: List[float], known: int, length: int):
    """Returns (evidence_factor, [margin_factor per ranked item], gap_to_second)."""
    ev = 0.65 + 0.35 * (known / length) if length else 0.65
    if not finals:
        return ev, [], 0.0
    gap = finals[0] - finals[1] if len(finals) > 1 else finals[0]
    lead = min(1.0, gap / 0.15)
    # runner-ups inherit the leader's margin factor, scaled by how far behind they are, so a
    # lower-ranked candidate can never show higher confidence than the one above it
    margins = [lead] + [lead * max(0.0, 1.0 - (finals[0] - f) / 0.40) for f in finals[1:]]
    return min(1.0, ev), margins, gap


def match(fused: FusedPlate, attributes: Attributes, registry: List[RegistryEntry],
          settings: Settings) -> Tuple[Verdict, List[Candidate], List[Candidate]]:
    scored = []
    for e in registry:
        ps, cm, orph = compare_plate(fused, e.plate)
        cs = colour_score(attributes.colour.value, e.colour)
        ts = type_score(attributes.body_type.value, e.type)
        final = settings.weight_plate * ps + settings.weight_colour * cs + settings.weight_type * ts
        scored.append((min(1.0, max(0.0, final)), ps, cs, ts, cm, e, orph))
    scored.sort(key=lambda t: (-t[0], -t[1], t[5].plate))
    ranked = scored[: settings.top_k]

    finals = [t[0] for t in scored]
    ev, margins, gap = _confidences(finals, fused.known_count, fused.length)

    cands: List[Candidate] = []
    for i, (final, ps, cs, ts, cm, e, orph) in enumerate(ranked):
        mf = margins[i]
        cands.append(Candidate(
            rank=i + 1, plate=e.plate, make=e.make or None, model=e.model or None,
            registry_colour=e.colour, registry_type=e.type,
            confidence=max(0.0, min(1.0, final * ev * mf)),
            breakdown=ScoreBreakdown(plate_score=ps, colour_score=cs, type_score=ts, final_score=final,
                                     evidence_factor=ev, margin_factor=mf),
            char_matches=cm,
            reasons=_reasons(fused, cm, orph, cs, ts, attributes.colour.value, attributes.body_type.value, e),
        ))

    best = finals[0] if finals else 0.0
    checks = [
        RuleCheck(name="minimum known characters", passed=fused.known_count >= settings.min_known_chars,
                  detail=f"{fused.known_count} of {fused.length} characters readable (need >= {settings.min_known_chars})"),
        RuleCheck(name="best score threshold", passed=best >= settings.no_match_threshold,
                  detail=f"best final score {best:.0%} (need >= {settings.no_match_threshold:.0%})"),
        RuleCheck(name="margin over runner-up", passed=bool(finals) and gap >= settings.min_margin,
                  detail=f"gap to second {gap:.1%} (need >= {settings.min_margin:.1%})"),
    ]
    if all(c.passed for c in checks):
        conf = cands[0].confidence
        verdict = Verdict(status="match", headline=f"Top match: {round(conf * 100)}% confidence",
                          top_confidence=conf, rule_checks=checks,
                          reason=f"{cands[0].plate} scored {best:.0%} and leads the runner-up by {gap:.1%}; "
                                 f"{fused.known_count} of {fused.length} plate characters readable.")
        return verdict, cands, []
    failed = [c.detail for c in checks if not c.passed]
    reason = f"{fused.known_count} of {fused.length} plate characters readable; best candidate scored {best:.0%}"
    if failed:
        reason += ". Failed: " + "; ".join(failed)
    verdict = Verdict(status="no_confident_match", headline="No confident match", top_confidence=None,
                      reason=reason, rule_checks=checks)
    return verdict, [], cands
