"""Shared helpers: paths, safe HTML, image loading, session state, theming."""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import streamlit as st

from schema import AnalysisResult, load_result, save_result

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
MOCK_DIR = ROOT / "app" / "mock"
MOCK_FILES = {
    "Successful match": MOCK_DIR / "mock_result.json",
    "No confident match": MOCK_DIR / "mock_no_match.json",
}
# Set True to surface make/model and registry colour/type on candidate cards and in the report again.
SHOW_REGISTRY_DETAILS = False

# Palettes carry every CSS variable the app needs. Add new keys here, never remove one.
PALETTES = {
    "light": dict(
        bg="#f6f8fb", card="#ffffff", line="#e2e8f0", text="#0f1e3d", muted="#475569",
        accent="#1d4ed8", accent2="#7c3aed", soft="#eff6ff", field="#ffffff",
        shadow="rgba(15,30,61,.12)", teal="#0d9488",
        ok_bg="#dcfce7", ok_bd="#16a34a", ok_tx="#14532d",
        warn_bg="#fef3c7", warn_bd="#d97706", warn_tx="#78350f",
        bad_bg="#fee2e2", bad_bd="#dc2626", bad_tx="#7f1d1d",
        grey_bg="#f1f5f9", grey_bd="#94a3b8", grey_tx="#475569",
        hero_grad="linear-gradient(135deg, #dbeafe 0%, #ede9fe 60%, #ffffff 100%)",
        card_top_match="#16a34a", card_top_nomatch="#f59e0b", card_top_time="#0d9488", card_top_total="#1d4ed8",
        toggle_sun="#f59e0b", toggle_moon="#1d4ed8",
    ),
    "dark": dict(
        bg="#0b1220", card="#111c33", line="#223356", text="#e6edf7", muted="#9fb0c9",
        accent="#60a5fa", accent2="#a78bfa", soft="#12213f", field="#0f192e",
        shadow="rgba(0,0,0,.5)", teal="#2dd4bf",
        ok_bg="#0f2a1d", ok_bd="#22c55e", ok_tx="#bbf7d0",
        warn_bg="#2e2410", warn_bd="#f59e0b", warn_tx="#fde68a",
        bad_bg="#2f1416", bad_bd="#f87171", bad_tx="#fecaca",
        grey_bg="#18233d", grey_bd="#64748b", grey_tx="#cbd5e1",
        hero_grad="linear-gradient(135deg, #12213f 0%, #1a1645 60%, #111c33 100%)",
        card_top_match="#22c55e", card_top_nomatch="#f59e0b", card_top_time="#2dd4bf", card_top_total="#60a5fa",
        toggle_sun="#f59e0b", toggle_moon="#60a5fa",
    ),
}


def palette() -> dict:
    """Return the active palette (dark or light) based on session state."""
    return PALETTES["dark" if st.session_state.get("dark") else "light"]


def theme_css() -> str:
    """Render the palette as a :root{...} CSS rule, refreshed on every script run."""
    return ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in palette().items()) + "}"


def html_table(rows, raw=()):
    """Theme-aware HTML table (st.dataframe cannot follow our dark mode)."""
    if not rows:
        return
    cols = list(rows[0])
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c] if c in raw else esc(r[c])}</td>" for c in cols) + "</tr>" for r in rows)
    st.markdown(f'<div class="tv-tablewrap"><table class="tv-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def safe_hex(value, default="#cbd5e1") -> str:
    return value if value and _HEX.match(value) else default


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text)


def resolve_image(run_dir, rel):
    """Find an image path relative to the run folder. Returns a Path or None. Never escapes the run folder."""
    if not rel:
        return None
    base = Path(run_dir).resolve()
    options = []
    p = (base / rel).resolve()
    if base in p.parents:
        options.append(p)
    options.append(MOCK_DIR / "sample_images" / base.name / Path(rel).name)  # demo images per run
    options.append(MOCK_DIR / "sample_images" / Path(rel).name)
    return next((o for o in options if o.is_file()), None)


@st.cache_data(show_spinner=False)
def load_image(path_str: str):
    try:
        import cv2
        img = cv2.imread(path_str)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            from PIL import Image
            return np.array(Image.open(path_str).convert("RGB"))
        except Exception:
            return None


def show_image(run_dir, rel, caption=None) -> bool:
    """Show an image, or a neutral placeholder if it is missing. Never raises."""
    p = resolve_image(run_dir, rel)
    arr = load_image(str(p)) if p else None
    if arr is None:
        st.markdown('<div class="tv-ph">Image not available</div>', unsafe_allow_html=True)
        return False
    st.image(arr, caption=caption, use_container_width=True)
    return True


@st.cache_data(show_spinner=False)
def load_mock(path_str: str) -> AnalysisResult:
    """Loads and validates a mock JSON with the shared schema (raises on invalid JSON)."""
    return load_result(path_str)


def set_result(result: AnalysisResult, source: str) -> None:
    run_dir = RUNS / safe_name(result.run_id)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_result(result, str(run_dir / "result.json"))
    except OSError:
        pass
    st.session_state.update(result=result, source=source, run_dir=str(run_dir))
    # Per-session tally shown on the animated stat cards.
    st.session_state.setdefault("history", {})[result.run_id] = {
        "status": result.verdict.status,
        "time": sum(result.timings_s.values()),
    }


def clear_result() -> None:
    for k in ("result", "source", "run_dir"):
        st.session_state.pop(k, None)
