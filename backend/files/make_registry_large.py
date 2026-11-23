"""Grow the synthetic registry to N rows with varied Indian plate formats, states and vehicles.

Run from the repo root:
    python scripts/make_registry_large.py --n 1600
Writes data/registry_large.csv (does NOT overwrite data/registry.csv). Existing rows from
data/registry.csv and data/registry_extra.csv are kept first, so demo / test-clip plates survive.
Colours use the palette names of matching/attributes.py; types use the labels scoring understands.
"""
import argparse
import csv
import os
import random
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"          # real series skip I and O

STATES = {  # state/UT code: relative weight
    "DL": 9, "MH": 9, "KA": 8, "TN": 8, "UP": 8, "GJ": 6, "RJ": 6, "CG": 6, "MP": 6, "HR": 5,
    "PB": 4, "WB": 5, "KL": 5, "TS": 5, "AP": 5, "BR": 4, "OD": 4, "JH": 3, "AS": 3, "UK": 2,
    "HP": 2, "JK": 2, "GA": 1, "CH": 2, "PY": 1, "AN": 1, "AR": 1, "MN": 1, "ML": 1, "MZ": 1,
    "NL": 1, "SK": 1, "TR": 1, "DN": 1, "DD": 1, "LD": 1, "LA": 1,
}
FORMATS = {"std2": 55, "single_rto": 18, "std1": 8, "std3": 4, "bh": 5}   # BH = Bharat series

TYPES = {"hatchback": 28, "sedan": 15, "SUV": 22, "truck": 8, "bus": 4, "motorcycle": 20, "auto-rickshaw": 3}
MODELS = {
    "hatchback": [("Maruti Suzuki", "Swift"), ("Maruti Suzuki", "WagonR"), ("Maruti Suzuki", "Alto"),
                  ("Maruti Suzuki", "Celerio"), ("Hyundai", "i10"), ("Hyundai", "i20"), ("Tata", "Tiago"),
                  ("Tata", "Altroz"), ("Renault", "Kwid"), ("Toyota", "Glanza")],
    "sedan": [("Honda", "City"), ("Honda", "Amaze"), ("Hyundai", "Verna"), ("Maruti Suzuki", "Ciaz"),
              ("Maruti Suzuki", "Dzire"), ("Skoda", "Slavia"), ("Volkswagen", "Virtus"), ("Tata", "Tigor")],
    "SUV": [("Hyundai", "Creta"), ("Hyundai", "Venue"), ("Mahindra", "XUV700"), ("Mahindra", "Scorpio"),
            ("Mahindra", "Thar"), ("Tata", "Nexon"), ("Tata", "Harrier"), ("Maruti Suzuki", "Brezza"),
            ("Kia", "Seltos"), ("Toyota", "Fortuner")],
    "truck": [("Tata", "407"), ("Tata", "Ace"), ("Ashok Leyland", "Dost"), ("Mahindra", "Bolero Pickup"),
              ("Eicher", "Pro 2049"), ("BharatBenz", "1617")],
    "bus": [("Eicher", "Skyline"), ("Tata", "Starbus"), ("Ashok Leyland", "Lynx"), ("Force", "Traveller")],
    "auto-rickshaw": [("Bajaj", "RE"), ("Piaggio", "Ape"), ("Mahindra", "Treo")],
    "motorcycle": [("Hero", "Splendor"), ("Honda", "Shine"), ("Honda", "Activa"), ("Bajaj", "Pulsar"),
                   ("TVS", "Apache"), ("Royal Enfield", "Classic 350"), ("Yamaha", "FZ"), ("Suzuki", "Access")],
}
COLOURS_PRIVATE = {"white": 30, "silver": 15, "grey": 12, "black": 14, "red": 7, "blue": 8,
                   "brown": 3, "beige": 2, "green": 2, "orange": 2, "yellow": 1}
COLOURS_COMMERCIAL = {"white": 20, "yellow": 18, "blue": 16, "red": 10, "green": 10, "orange": 10,
                      "silver": 8, "grey": 8}


def pick(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def number(rng):
    return "%04d" % rng.randint(1, 9999)


def letters(rng, k):
    return "".join(rng.choice(LETTERS) for _ in range(k))


def make_plate(rng):
    st, fmt = pick(rng, STATES), pick(rng, FORMATS)
    if fmt == "std2":
        return f"{st}{rng.randint(1, 99):02d}{letters(rng, 2)}{number(rng)}"
    if fmt == "std1":
        return f"{st}{rng.randint(1, 99):02d}{letters(rng, 1)}{number(rng)}"
    if fmt == "std3":
        return f"{st}{rng.randint(1, 99):02d}{letters(rng, 3)}{number(rng)}"
    if fmt == "single_rto":                     # e.g. DL1CS2343
        return f"{st}{rng.randint(1, 9)}{letters(rng, 2)}{number(rng)}"
    return f"{rng.randint(21, 30)}BH{number(rng)}{letters(rng, 2)}"     # e.g. 22BH1234AB


def make_row(rng, plate, colour=None, typ=None):
    typ = typ or pick(rng, TYPES)
    key = {"two-wheeler": "motorcycle", "car": "hatchback"}.get(typ, typ)   # existing rows may use other labels
    make, model = rng.choice(MODELS.get(key, MODELS["hatchback"]))
    if typ == "motorcycle" and model in ("Activa", "Access"):
        typ = "two-wheeler"                      # scooters use the registry's other label
    colours = COLOURS_COMMERCIAL if typ in ("truck", "bus", "auto-rickshaw") else COLOURS_PRIVATE
    return {"plate": plate, "colour": colour or pick(rng, colours), "type": typ, "make": make, "model": model}


def near_duplicate(rng, plate):
    """One digit in the number, or one series letter, changed."""
    chars = list(plate)
    digit_idx = [i for i, c in enumerate(chars) if c.isdigit() and i >= len(chars) - 4 - (2 if "BH" in plate else 0)]
    if "BH" in plate:
        digit_idx = [i for i in range(4, 8)]
    letter_idx = [i for i, c in enumerate(chars) if c.isalpha() and i >= 2 and "BH" not in plate]
    if letter_idx and rng.random() < 0.25:
        i = rng.choice(letter_idx)
        chars[i] = rng.choice([c for c in LETTERS if c != chars[i]])
    elif digit_idx:
        i = rng.choice(digit_idx)
        chars[i] = rng.choice([d for d in "0123456789" if d != chars[i]])
    return "".join(chars)


def norm(p):
    return re.sub(r"[^A-Z0-9]", "", (p or "").upper())


def load_existing(paths):
    rows, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r = {k.strip().lower(): (v or "").strip() for k, v in r.items()}
                plate = norm(r.get("plate"))
                if plate and plate not in seen:
                    seen.add(plate)
                    rows.append({"plate": plate, "colour": r.get("colour", ""), "type": r.get("type", ""),
                                 "make": r.get("make", ""), "model": r.get("model", "")})
    return rows, seen


def fmt_of(plate):
    if re.fullmatch(r"\d{2}BH\d{4}[A-Z]{2}", plate):
        return "bh"
    m = re.fullmatch(r"[A-Z]{2}(\d{1,2})([A-Z]{1,3})\d{4}", plate)
    if not m:
        return "other"
    if len(m.group(1)) == 1:
        return "single_rto"
    return f"std{len(m.group(2))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "registry_large.csv"))
    ap.add_argument("--keep", nargs="*", default=[os.path.join(ROOT, "data", "registry.csv"),
                                                  os.path.join(ROOT, "data", "registry_extra.csv")])
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows, seen = load_existing(a.keep)
    kept = len(rows)
    while len(rows) < a.n:
        if rng.random() < 0.08 and rows:                       # near-duplicate of an existing row
            base = rng.choice(rows)
            p = near_duplicate(rng, base["plate"])
            if p in seen or p == base["plate"]:
                continue
            same = rng.random() < 0.5                          # half also share colour and type (hardest case)
            row = make_row(rng, p, base["colour"] or None, base["type"] or None) if same else make_row(rng, p)
        else:
            p = make_plate(rng)
            if p in seen:
                continue
            row = make_row(rng, p)
        seen.add(p)
        rows.append(row)

    rng.shuffle(rows)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["plate", "colour", "type", "make", "model"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out} ({kept} kept from existing files, {len(rows) - kept} added)")
    print("formats:", dict(Counter(fmt_of(r["plate"]) for r in rows)))
    print("types  :", dict(Counter(r["type"] for r in rows)))
    print("states (top 8):", Counter(r["plate"][:2] for r in rows if not r["plate"][:2].isdigit()).most_common(8))


if __name__ == "__main__":
    main()
