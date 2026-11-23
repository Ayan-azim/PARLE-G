"""Generate a synthetic registry with deliberate near-duplicates and same-colour/type collisions.
Run from the repo root:  python scripts/make_registry.py"""
import csv
import random

random.seed(42)
STATES = ["CG"] * 7 + ["MP", "MH", "OD", "DL"]
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
COLOURS = ["white"] * 5 + ["silver"] * 3 + ["grey"] * 2 + ["black"] * 3 + ["red", "blue", "brown", "yellow"]
MODELS = {
    "hatchback": [("Maruti", "Swift"), ("Hyundai", "i20"), ("Tata", "Altroz")],
    "sedan": [("Honda", "City"), ("Hyundai", "Verna"), ("Maruti", "Ciaz")],
    "SUV": [("Hyundai", "Creta"), ("Mahindra", "XUV700"), ("Tata", "Nexon")],
    "truck": [("Tata", "407"), ("Ashok Leyland", "Dost")],
    "bus": [("Eicher", "Skyline")],
    "motorcycle": [("Hero", "Splendor"), ("Honda", "Shine"), ("Bajaj", "Pulsar")],
}
TYPES = ["hatchback"] * 3 + ["sedan"] * 2 + ["SUV"] * 3 + ["truck", "bus"] + ["motorcycle"] * 2


def plate():
    series = "".join(random.choice(LETTERS) for _ in range(random.choice([1, 2, 2, 2, 3])))
    return f"{random.choice(STATES)}{random.randint(1, 30):02d}{series}{random.randint(1, 9999):04d}"


def row(p, colour=None, typ=None):
    typ = typ or random.choice(TYPES)
    make, model = random.choice(MODELS[typ])
    return {"plate": p, "colour": colour or random.choice(COLOURS), "type": typ, "make": make, "model": model}


rows, seen = [], set()
while len(rows) < 260:
    p = plate()
    if p not in seen:
        seen.add(p)
        rows.append(row(p))

# 40 near-duplicates: one digit changed; half share colour AND type (hardest collisions)
for base in random.sample(rows[:260], 40):
    p = base["plate"]
    i = random.randrange(len(p) - 4, len(p))
    q = p[:i] + random.choice([d for d in "0123456789" if d != p[i]]) + p[i + 1:]
    if q in seen:
        continue
    seen.add(q)
    if random.random() < 0.5:
        rows.append({**base, "plate": q})
    else:
        rows.append(row(q))

random.shuffle(rows)
with open("data/registry.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["plate", "colour", "type", "make", "model"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to data/registry.csv")
