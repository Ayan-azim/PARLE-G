"""Registry loading and plate normalisation (Person B)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pandas as pd

REQUIRED_COLUMNS = ["plate", "colour", "type", "make", "model"]


def normalise_plate(s: str | None, keep_unknown: bool = False) -> str:
    """Uppercase, strip spaces/hyphens/punctuation. With keep_unknown, '?' and '*' survive as '?'."""
    if not s:
        return ""
    pattern = r"[^A-Za-z0-9?*]" if keep_unknown else r"[^A-Za-z0-9]"
    return re.sub(pattern, "", s).upper().replace("*", "?")


@dataclass(frozen=True)
class RegistryEntry:
    plate: str
    colour: str
    type: str
    make: str
    model: str


def load_registry(path: str) -> List[RegistryEntry]:
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"registry CSV is missing columns: {missing}")
    entries = []
    for row in df.itertuples(index=False):
        plate = normalise_plate(row.plate)
        if plate:
            entries.append(RegistryEntry(plate, row.colour.strip(), row.type.strip(), row.make.strip(), row.model.strip()))
    return entries
