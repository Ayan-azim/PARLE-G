"""Theme-aware statistics cards rendered natively by Streamlit."""
from __future__ import annotations

import streamlit as st

from app.components.common import palette


def render(result):
    p = palette()
    hist = st.session_state.get("history", {})
    total = len(hist)
    matches = sum(1 for item in hist.values() if item.get("status") == "match")
    no_match = sum(1 for item in hist.values() if item.get("status") == "no_confident_match")
    t = sum(result.timings_s.values()) if result is not None and result.timings_s else None
    cards = [
        ("TOTAL ANALYSES", str(total), "Across this session", p["accent"]),
        ("MATCHES FOUND", str(matches), f"{matches / total:.0%} of analyses" if total else "No analyses yet", p["ok_bd"]),
        ("NO CONFIDENT MATCH", str(no_match), f"{no_match / total:.0%} of analyses" if total else "No analyses yet", p["warn_bd"]),
        ("PROCESSING TIME", f"{t:.1f} s" if t is not None else "—", "Current run · all stages" if t is not None else "No timing data", p["teal"]),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, note, accent) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="tv-stat" style="--stat-accent:{accent}">'
                f'<div class="tv-stat-label">{label}</div>'
                f'<div class="tv-stat-value">{value}</div>'
                f'<div class="tv-stat-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
