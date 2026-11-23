"""Shared helpers: branding, theme, paths, safe HTML, images, session state."""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import streamlit as st

from schema import AnalysisResult, load_result, save_result

APP_NAME = "VEHICLE TRACKER"
TAGLINE = "AI-Powered Vehicle Identification & Traceability"
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
MOCK_DIR = ROOT / "app" / "mock"
ASSETS = ROOT / "app" / "assets"
MOCK_FILES = {
    "Successful match": MOCK_DIR / "mock_result.json",
    "No confident match": MOCK_DIR / "mock_no_match.json",
}

# Frontend-only switch. False hides make/model, registry colour/type, observed colour/body type and vehicle class.
# The data stays in the result objects and JSON; set True to show it all again.
SHOW_VEHICLE_DETAILS = False
_DETAIL_WORDS = re.compile(r"\b(colou?r|body type|registered|make|model)\b", re.I)


def visible_reasons(reasons):
    """Candidate reasons, without lines that talk about colour / body type / registry vehicle details."""
    return list(reasons) if SHOW_VEHICLE_DETAILS else [r for r in reasons if not _DETAIL_WORDS.search(r)]


def _version():
    m = re.match(r"(\d+)\.(\d+)", getattr(st, "__version__", "0.0"))
    return (int(m[1]), int(m[2])) if m else (0, 0)


# Newer Streamlit replaced use_container_width with width="stretch"; pick whichever this version expects.
STRETCH = {"width": "stretch"} if _version() >= (1, 50) else {"use_container_width": True}

PALETTES = {
    "light": dict(bg="#f4f6fb", card="#ffffff", line="#e3e8f2", text="#0e1730", muted="#5b6784", accent="#0F9F75", accent2="#34D399", soft="#E8F8F1",
                  field="#ffffff", shadow="rgba(30,41,90,.13)", teal="#0F9F75", ok_bg="#dcfce7", ok_bd="#16a34a", ok_tx="#14532d", warn_bg="#fef3c7",
                  warn_bd="#d97706", warn_tx="#78350f", bad_bg="#fee2e2", bad_bd="#dc2626", bad_tx="#7f1d1d", grey_bg="#f1f5f9", grey_bd="#94a3b8", grey_tx="#475569"),
    "dark": dict(bg="#080d1c", card="#0f1730", line="#1f2a4d", text="#e8ecf8", muted="#93a0c4", accent="#34D399", accent2="#A7F3D0", soft="#102820",
                 field="#0c1329", shadow="rgba(0,0,0,.55)", teal="#34D399", ok_bg="#0f2a1d", ok_bd="#22c55e", ok_tx="#bbf7d0", warn_bg="#2e2410",
                 warn_bd="#f59e0b", warn_tx="#fde68a", bad_bg="#2f1416", bad_bd="#f87171", bad_tx="#fecaca", grey_bg="#18233d", grey_bd="#64748b", grey_tx="#cbd5e1"),
}


def palette() -> dict:
    return PALETTES["dark" if st.session_state.get("dark") else "light"]


def theme_css() -> str:
    return ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in palette().items()) + "}"


def logo_svg(size: int = 44, uid: str = "a") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" role="img" aria-label="{APP_NAME} logo"><defs><linearGradient id="lg{uid}" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#047857"/><stop offset="1" stop-color="#34D399"/></linearGradient></defs>'
            f'<rect width="48" height="48" rx="13" fill="url(#lg{uid})"/>'
            '<path d="M24 8c-6.6 0-12 5.2-12 11.7 0 8.2 12 18.3 12 18.3s12-10.1 12-18.3C36 13.2 30.6 8 24 8z" fill="none" stroke="#fff" stroke-width="2.6"/>'
            '<circle cx="24" cy="20" r="5.2" fill="none" stroke="#fff" stroke-width="2.4"/><circle cx="24" cy="20" r="1.9" fill="#fff"/>'
            '<path d="M9 42h9M30 42h9" stroke="#fff" stroke-opacity=".75" stroke-width="2.4" stroke-linecap="round"/></svg>')


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def case_badges(result) -> str:
    frames = result.vision.frames
    used = sum(1 for f in frames if f.used)
    skipped = len(frames) - used
    out = (f'<span class="tv-badge" title="Run folder name">Case ID <b>{esc(result.run_id)}</b></span>'
           f'<span class="tv-badge" title="Frames captured as evidence">Evidence Count <b>{len(frames)}</b></span>'
           f'<span class="tv-badge tv-badge-ok">{used} used</span>')
    if skipped:
        out += f'<span class="tv-badge tv-badge-warn">{skipped} skipped</span>'
    return f'<div class="tv-badges">{out}</div>'


def html_table(rows, raw=()):
    """Theme-aware table. A row may carry "_cls" (a CSS class for the row). Cells are escaped unless the column is in raw."""
    if not rows:
        return
    cols = [c for c in rows[0] if c != "_cls"]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join(f'<tr class="{esc(r.get("_cls", ""))}">' + "".join(f"<td>{r[c] if c in raw else esc(r[c])}</td>" for c in cols) + "</tr>" for r in rows)
    st.markdown(f'<div class="tv-tablewrap"><table class="tv-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def safe_hex(value, default="#cbd5e1") -> str:
    return value if value and _HEX.match(value) else default


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text)


def resolve_image(run_dir, rel):
    """Find an image path (relative to the run folder). Returns a Path or None. Never leaves the run folder."""
    if not rel:
        return None
    base = Path(run_dir).resolve()
    options = []
    p = (base / rel).resolve()
    if base in p.parents:
        options.append(p)
    options.append(MOCK_DIR / "sample_images" / base.name / Path(rel).name)  # optional demo images, per run
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
    st.image(arr, caption=caption, **STRETCH)
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
    st.session_state.setdefault("history", {})[result.run_id] = {"status": result.verdict.status, "time": sum(result.timings_s.values())}


def clear_result() -> None:
    for k in ("result", "source", "run_dir"):
        st.session_state.pop(k, None)
