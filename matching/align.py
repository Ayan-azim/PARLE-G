"""Confusable-aware edit-distance alignment (Needleman-Wunsch), shared by fusion and scoring."""
from __future__ import annotations

from typing import List, Optional, Tuple

CONFUSABLE = {frozenset(p) for p in ("0O", "1I", "8B", "5S", "2Z")}
GAP = 1.0


def sub_cost(a: str, b: str) -> float:
    if a == "?" or b == "?":
        return 0.2          # unknown vs anything: cheap, but still prefers real matches
    if a == b:
        return 0.0
    if frozenset((a, b)) in CONFUSABLE:
        return 0.3
    return 1.0


def align(a: str, b: str) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    """Returns (cost, pairs). Each pair is (index in a | None, index in b | None); None = gap."""
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * GAP
    for j in range(1, m + 1):
        D[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1]), D[i - 1][j] + GAP, D[i][j - 1] + GAP)
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1]))) < 1e-9:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + GAP)) < 1e-9:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return D[n][m], pairs
