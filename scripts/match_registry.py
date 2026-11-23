"""
Stage 2: From-Scratch Vehicle Matching, Ranking & No-Match Engine.

Usage:
    python scripts/match_registry.py runs/clip1_new/vision.json --registry data/registry.csv
"""

import argparse
import csv
import json
import os
import sys

# Ensure project root is on sys.path so `vision.colour` imports cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.colour import colour_similarity

# Common Indian CCTV OCR character confusions
CONFUSABLE_GROUPS = [
    {"0", "O", "D", "Q"},
    {"1", "I", "L", "T"},
    {"2", "Z"},
    {"5", "S"},
    {"6", "G"},
    {"8", "B"},
    {"B", "D"},
    {"U", "V"},
    {"4", "A"},
    {"7", "T"},
]

# Compatibility between YOLO coarse classes and registry body types
TYPE_COMPAT = {
    "car": {"hatchback": 1.0, "sedan": 1.0, "suv": 1.0, "car": 1.0, "auto-rickshaw": 0.3, "truck": 0.1},
    "two-wheeler": {"two-wheeler": 1.0, "motorcycle": 1.0, "auto-rickshaw": 0.2},
    "bus": {"bus": 1.0, "truck": 0.35},
    "truck": {"truck": 1.0, "bus": 0.35, "suv": 0.25},
}


def are_confusable(a: str, b: str) -> bool:
    if a == b:
        return False
    return any(a in grp and b in grp for grp in CONFUSABLE_GROUPS)


def align_and_score_plate(obs_plate: str, obs_probs: list, reg_plate: str):
    """
    Needleman-Wunsch alignment between observed partial plate (with '?') and
    a registry plate. Returns (plate_score, char_matches, summary_counts).
    """
    m, n = len(obs_plate), len(reg_plate)
    if m == 0 or n == 0:
        return 0.0, [], {"match": 0, "confusable": 0, "mismatch": 0, "unknown": 0, "gap": n}

    probs = list(obs_probs) if len(obs_probs) == m else [0.75] * m

    gap_pen = -1.2
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * gap_pen
        bt[i][0] = "U"
    for j in range(1, n + 1):
        dp[0][j] = j * gap_pen
        bt[0][j] = "L"

    for i in range(1, m + 1):
        oc = obs_plate[i - 1]
        for j in range(1, n + 1):
            rc = reg_plate[j - 1]
            if oc == "?":
                sub = 0.4
            elif oc == rc:
                sub = 2.0
            elif are_confusable(oc, rc):
                sub = 0.8
            else:
                sub = -1.5

            diag = dp[i - 1][j - 1] + sub
            up = dp[i - 1][j] + gap_pen
            left = dp[i][j - 1] + gap_pen

            if diag >= up and diag >= left:
                dp[i][j], bt[i][j] = diag, "D"
            elif up >= left:
                dp[i][j], bt[i][j] = up, "U"
            else:
                dp[i][j], bt[i][j] = left, "L"

    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "D":
            aligned.append((obs_plate[i - 1], probs[i - 1], reg_plate[j - 1]))
            i -= 1
            j -= 1
        elif move == "U":
            aligned.append((obs_plate[i - 1], probs[i - 1], "-"))
            i -= 1
        else:
            aligned.append(("-", 0.0, reg_plate[j - 1]))
            j -= 1
    aligned.reverse()

    num, den = 0.0, 0.0
    mismatch_weight = 0.0
    char_matches = []
    counts = {"match": 0, "confusable": 0, "mismatch": 0, "unknown": 0, "gap": 0}

    for pos, (oc, prob, rc) in enumerate(aligned):
        w = max(0.35, float(prob))
        if oc == "?":
            status = "unknown"
            counts["unknown"] += 1
        elif oc == "-" or rc == "-":
            status = "gap"
            counts["gap"] += 1
            den += 0.65
            mismatch_weight += 0.45
        elif oc == rc:
            status = "match"
            counts["match"] += 1
            num += w
            den += w
        elif are_confusable(oc, rc):
            status = "confusable"
            counts["confusable"] += 1
            num += 0.50 * w
            den += w
            mismatch_weight += 0.25 * w
        else:
            status = "mismatch"
            counts["mismatch"] += 1
            den += w
            mismatch_weight += w

        char_matches.append({
            "position": pos,
            "observed": oc,
            "registry": rc,
            "prob": round(float(prob), 3),
            "status": status,
        })

    base_score = (num / den) if den > 0 else 0.0
    # Penalize high-confidence character conflicts (e.g. '1' vs '3' at p=1.0)
    plate_score = base_score * (0.72 ** mismatch_weight)
    return round(plate_score, 4), char_matches, counts


def score_colour(colour_dist: dict, colour_conf: float, reg_colour: str) -> float:
    reg_c = reg_colour.strip().lower()
    if not colour_dist:
        return 0.5
    raw_sim = sum(prob * colour_similarity(obs_c.lower(), reg_c) for obs_c, prob in colour_dist.items())
    trust = min(1.0, max(0.0, float(colour_conf)))
    blended = (trust * raw_sim) + ((1.0 - trust) * 0.5)
    return round(blended, 4)


def score_type(obs_type: str, reg_type: str) -> float:
    ot = obs_type.strip().lower()
    rt = reg_type.strip().lower()
    if ot == rt:
        return 1.0
    return float(TYPE_COMPAT.get(ot, {}).get(rt, 0.0))


def build_reasons(counts, char_matches, obs_col, col_conf, reg_col, col_score, obs_type, reg_type, type_score):
    reasons = []
    known = counts["match"] + counts["confusable"] + counts["mismatch"]
    if counts["mismatch"] == 0 and counts["confusable"] == 0 and counts["gap"] == 0 and known > 0:
        reasons.append(f"All {counts['match']}/{known} recovered plate characters match exactly.")
    else:
        reasons.append(
            f"Plate alignment: {counts['match']} exact match(es), "
            f"{counts['confusable']} confusable, {counts['mismatch']} mismatch(es), {counts['gap']} gap(s)."
        )
        for cm in char_matches:
            if cm["status"] in ("mismatch", "confusable"):
                reasons.append(
                    f"Pos {cm['position'] + 1} ({cm['status']}): observed '{cm['observed']}' "
                    f"(p={cm['prob']:.2f}) vs registry '{cm['registry']}'."
                )

    if counts["unknown"] > 0:
        reasons.append(f"{counts['unknown']} unreadable '?' position(s) treated as wildcard (no penalty).")

    if col_conf < 0.40:
        reasons.append(f"Colour evidence weak (conf {col_conf:.2f}): observed {obs_col}, registered {reg_col} (soft weight).")
    elif col_score >= 0.75:
        reasons.append(f"Colour consistent: observed {obs_col}, registered {reg_col}.")
    elif col_score >= 0.40:
        reasons.append(f"Colour partial match: observed {obs_col}, registered {reg_col}.")
    else:
        reasons.append(f"Colour conflict: observed {obs_col}, registered {reg_col}.")

    if type_score >= 0.9:
        reasons.append(f"Vehicle type compatible: detected {obs_type}, registered {reg_type}.")
    elif type_score > 0.0:
        reasons.append(f"Vehicle type partially compatible: detected {obs_type}, registered {reg_type}.")
    else:
        reasons.append(f"Vehicle type mismatch: detected {obs_type}, registered {reg_type}.")

    return reasons


def match_vehicle(vehicle: dict, registry: list, top_k: int = 5,
                  min_known: int = 3, no_match_thresh: float = 0.60, min_margin: float = 0.08):
    obs_plate = vehicle.get("consensus_plate", "").upper()
    obs_probs = vehicle.get("consensus_char_probs", [])
    obs_col = vehicle.get("colour", {}).get("value", "unknown")
    col_conf = float(vehicle.get("colour", {}).get("confidence", 0.0))
    col_dist = vehicle.get("colour_distribution", {})
    obs_type = vehicle.get("type", {}).get("value", "car")

    known_count = sum(1 for c in obs_plate if c != "?")

    if known_count < min_known:
        return {
            "track_id": vehicle["track_id"],
            "merged_track_ids": vehicle.get("merged_track_ids", [vehicle["track_id"]]),
            "observed_plate": obs_plate,
            "observed_colour": obs_col,
            "observed_colour_conf": col_conf,
            "observed_type": obs_type,
            "total_frames_with_plate": vehicle.get("total_frames_with_plate", 0),
            "verdict": {
                "status": "no_confident_match",
                "headline": "No confident match",
                "top_confidence": 0.0,
                "reason": f"Only {known_count} readable character(s) recovered (minimum required: {min_known}).",
            },
            "candidates": [],
            "weak_leads": [],
            "frame_reads": vehicle.get("frame_reads", []),
        }

    # Weights: plate carries primary weight; colour & type break ties
    w_plate, w_col, w_type = 0.75, 0.15, 0.10
    scored = []

    for row in registry:
        reg_plate = row["plate"].strip().upper()
        reg_col = row["colour"].strip().lower()
        reg_type = row["type"].strip()

        p_score, char_matches, counts = align_and_score_plate(obs_plate, obs_probs, reg_plate)
        c_score = score_colour(col_dist, col_conf, reg_col)
        t_score = score_type(obs_type, reg_type)

        final_score = round(w_plate * p_score + w_col * c_score + w_type * t_score, 4)
        evidence_factor = round(0.65 + 0.35 * min(1.0, known_count / max(1, len(reg_plate))), 4)

        reasons = build_reasons(
            counts, char_matches, obs_col, col_conf, reg_col, c_score, obs_type, reg_type, t_score
        )

        scored.append({
            "plate": reg_plate,
            "make": row.get("make", ""),
            "model": row.get("model", ""),
            "registry_colour": reg_col,
            "registry_type": reg_type,
            "breakdown": {
                "plate_score": p_score,
                "colour_score": c_score,
                "type_score": t_score,
                "final_score": final_score,
                "evidence_factor": evidence_factor,
            },
            "char_matches": char_matches,
            "reasons": reasons,
        })

    scored.sort(key=lambda x: (x["breakdown"]["final_score"], x["breakdown"]["plate_score"]), reverse=True)
    top = scored[:top_k]

    best_score = top[0]["breakdown"]["final_score"]
    second_score = top[1]["breakdown"]["final_score"] if len(top) > 1 else 0.0
    margin = round(best_score - second_score, 4)

    # Monotonic confidence: #1 is always >= #2..#5
    top_margin_factor = min(1.0, 0.55 + 0.45 * (margin / 0.12)) if margin > 0 else 0.50
    top_conf = round(best_score * top[0]["breakdown"]["evidence_factor"] * top_margin_factor, 4)

    for rank, cand in enumerate(top, start=1):
        cand["rank"] = rank
        gap = best_score - cand["breakdown"]["final_score"]
        if rank == 1:
            cand["breakdown"]["margin_factor"] = round(top_margin_factor, 4)
            cand["confidence"] = top_conf
        else:
            rel_factor = max(0.05, 1.0 - (gap / 0.30))
            cand["breakdown"]["margin_factor"] = round(rel_factor, 4)
            cand["confidence"] = round(
                min(top_conf * 0.90, cand["breakdown"]["final_score"] * cand["breakdown"]["evidence_factor"] * rel_factor),
                4,
            )

    # Verdict decision (no-match handling)
    if best_score < no_match_thresh or top[0]["breakdown"]["plate_score"] < 0.55:
        verdict = {
            "status": "no_confident_match",
            "headline": "No confident match",
            "top_confidence": top[0]["confidence"],
            "reason": (
                f"Best candidate {top[0]['plate']} scored {best_score:.2f} "
                f"(plate {top[0]['breakdown']['plate_score']:.2f}), below confidence cutoff."
            ),
        }
        candidates, weak_leads = [], top
    elif margin < min_margin:
        verdict = {
            "status": "no_confident_match",
            "headline": "No confident match (ambiguous tie)",
            "top_confidence": top[0]["confidence"],
            "reason": f"Top two candidates ({top[0]['plate']} and {top[1]['plate']}) are too close (margin {margin:.2f} < {min_margin:.2f}).",
        }
        candidates, weak_leads = [], top
    else:
        verdict = {
            "status": "match",
            "headline": f"Top match: {top[0]['plate']} ({int(top[0]['confidence'] * 100)}% confidence)",
            "top_confidence": top[0]["confidence"],
            "reason": (
                f"{known_count} plate characters recovered; {top[0]['plate']} leads "
                f"{top[1]['plate'] if len(top) > 1 else 'others'} by {margin:.2f}."
            ),
        }
        candidates, weak_leads = top, []

    return {
        "track_id": vehicle["track_id"],
        "merged_track_ids": vehicle.get("merged_track_ids", [vehicle["track_id"]]),
        "observed_plate": obs_plate,
        "observed_colour": obs_col,
        "observed_colour_conf": col_conf,
        "observed_type": obs_type,
        "total_frames_with_plate": vehicle.get("total_frames_with_plate", 0),
        "verdict": verdict,
        "candidates": candidates,
        "weak_leads": weak_leads,
        "frame_reads": vehicle.get("frame_reads", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Match vision.json vehicles against registry CSV")
    parser.add_argument("vision_json", help="Path to vision.json from run_vision_pipeline.py")
    parser.add_argument("--registry", default="data/registry.csv", help="Path to vehicle registry CSV")
    parser.add_argument("--out", default=None, help="Output result JSON (default: matches.json next to vision.json)")
    args = parser.parse_args()

    with open(args.vision_json, "r", encoding="utf-8") as f:
        vision_data = json.load(f)

    with open(args.registry, "r", encoding="utf-8-sig") as f:
        registry = list(csv.DictReader(f))

    results = []
    for v in vision_data.get("vehicles", []):
        res = match_vehicle(v, registry)
        results.append(res)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.vision_json)), "matches.json")
    payload = {
        "video_path": vision_data.get("video_path"),
        "registry_path": args.registry,
        "registry_size": len(registry),
        "vehicle_results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMatched {len(results)} vehicle(s) against {len(registry)} registry entries -> {out_path}\n")
    for r in results:
        ids = ",".join(map(str, r["merged_track_ids"]))
        print(f"=== Track [{ids}] | Observed: '{r['observed_plate']}' | {r['observed_colour']} {r['observed_type']} ===")
        print(f"  Verdict: {r['verdict']['headline']} — {r['verdict']['reason']}")
        shown = r["candidates"] if r["candidates"] else r["weak_leads"][:3]
        label = "Top candidates" if r["candidates"] else "Weak leads"
        if shown:
            print(f"  {label}:")
            for c in shown:
                b = c["breakdown"]
                print(
                    f"    #{c['rank']} {c['plate']:12s} | {c['registry_colour']:6s} {c['registry_type']:11s} "
                    f"({c['make']} {c['model']}) | score={b['final_score']:.2f} "
                    f"(plate={b['plate_score']:.2f}, col={b['colour_score']:.2f}, type={b['type_score']:.2f}) "
                    f"| conf={c['confidence']:.2f}"
                )
        print()


if __name__ == "__main__":
    main()