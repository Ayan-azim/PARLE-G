import os
import re
import json
import argparse
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

VALID_CHARS = re.compile(r"[^A-Z0-9]")

def run_fast_plate_ocr(recognizer, img_path):
    """
    Runs fast-plate-ocr and returns (clean_text, char_probs_list).
    Handles both newer object outputs (pred.plate, pred.char_probs)
    and older tuple outputs ((plates, confs)).
    """
    out = recognizer.run(img_path, return_confidence=True)

    # Newer fast-plate-ocr (returns list of prediction objects)
    if isinstance(out, list) and len(out) > 0 and hasattr(out[0], "plate"):
        pred = out[0]
        raw_text = str(pred.plate)
        probs = [float(p) for p in pred.char_probs]
    # Older fast-plate-ocr (returns tuple: (plates_list, probs_array))
    elif isinstance(out, tuple) and len(out) == 2:
        plates, confs = out
        raw_text = str(plates[0])
        probs = [float(p) for p in confs[0]]
    else:
        return "", []

    # Strip padding underscores '_' while keeping character & prob aligned
    clean_chars = []
    clean_probs = []
    for ch, p in zip(raw_text, probs):
        if ch == "_" or not ch.isalnum():
            continue
        clean_chars.append(ch.upper())
        clean_probs.append(round(p, 4))

    return "".join(clean_chars), clean_probs

def mask_uncertain(text, probs, threshold=0.55):
    """Replace any character below threshold with '?' (never invent)."""
    masked = []
    for ch, p in zip(text, probs):
        masked.append(ch if p >= threshold else "?")
    return "".join(masked)

def main():
    parser = argparse.ArgumentParser(description="Step 4: Run fast-plate-ocr on Step 3 tracks")
    parser.add_argument("--meta", default="runs/step3_tracks/tracks_meta.json", help="Path to tracks_meta.json")
    parser.add_argument("--out", default="runs/step3_tracks/ocr_results.json", help="Output JSON path")
    parser.add_argument("--conf-thresh", type=float, default=0.55, help="Below this prob -> '?'")
    parser.add_argument("--min-frames", type=int, default=3, help="Skip tiny noisy tracks with fewer plate frames")
    args = parser.parse_args()

    with open(args.meta, "r") as f:
        tracks = json.load(f)

    print("Loading fast-plate-ocr (cct-s-v2-global-model)...")
    recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

    output = {}
    for tid, info in tracks.items():
        if info["total_frames_with_plate"] < args.min_frames:
            continue

        frame_reads = []
        for item in info["top_plates"]:
            # Run on both enhanced and raw crop, pick whichever has higher mean confidence
            enh_text, enh_probs = run_fast_plate_ocr(recognizer, item["enh_path"])
            raw_text, raw_probs = run_fast_plate_ocr(recognizer, item["raw_path"])

            enh_mean = float(np.mean(enh_probs)) if enh_probs else 0.0
            raw_mean = float(np.mean(raw_probs)) if raw_probs else 0.0

            if enh_mean >= raw_mean:
                best_text, best_probs, src = enh_text, enh_probs, "enh"
            else:
                best_text, best_probs, src = raw_text, raw_probs, "raw"

            masked = mask_uncertain(best_text, best_probs, args.conf_thresh)
            frame_reads.append({
                "frame": item["frame"],
                "sharpness": item["sharpness"],
                "plate_conf": item["plate_conf"],
                "source": src,
                "raw_text": best_text,
                "masked_text": masked,
                "char_probs": best_probs,
                "mean_char_prob": round(max(enh_mean, raw_mean), 4),
                "image_path": item["enh_path"] if src == "enh" else item["raw_path"],
            })

        output[tid] = {
            "track_id": info["track_id"],
            "type": info["type"],
            "colour": info["colour"],
            "total_frames_with_plate": info["total_frames_with_plate"],
            "reads": frame_reads,
        }

        # Print quick preview for this track
        top_preview = sorted(frame_reads, key=lambda r: r["mean_char_prob"], reverse=True)[:3]
        print(f"\nTrack {tid} ({info['colour']} {info['type']}, {info['total_frames_with_plate']} frames):")
        for r in top_preview:
            print(f"  f{r['frame']:04d} [{r['source']}] -> {r['masked_text']:12s} (raw: {r['raw_text']:12s} | mean_p: {r['mean_char_prob']:.2f})")

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=int)

    print(f"\nSaved OCR results to: {args.out}")

if __name__ == "__main__":
    main()