"""TraceVision - Person C UI. Run with:  streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TraceVision", page_icon="🔍", layout="wide")

from app.components import common, evidence, input_step, results, stats, viewer  # noqa: E402
from schema import Settings  # noqa: E402

# Theme preference: kept in session state AND in the URL (?theme=dark) so it survives refresh.
if "dark" not in st.session_state:
    qp = st.query_params.get("theme")
    st.session_state.dark = (qp == "dark")

# Inject theme variables + global styles on every run, so toggling reflects instantly.
st.markdown(
    f"<style>{common.theme_css()}\n{(ROOT / 'app' / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

D = Settings()  # defaults come from the shared schema


@st.cache_data(show_spinner=False)
def _count(path: str, mtime: float) -> int:
    return len(pd.read_csv(path))


def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="tv-brand">Trace<span>Vision</span></div>'
            '<div class="tv-tag">AI-Powered Vehicle Identification &amp; Traceability</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="tv-theme-pill"><span class="tv-dot"></span>'
            f'Theme: {"Dark" if st.session_state.dark else "Light"}</div>',
            unsafe_allow_html=True,
        )
        st.toggle("Dark mode", key="dark", help="Switch between dark and light themes. The choice is remembered in the page URL.")
        st.query_params["theme"] = "dark" if st.session_state.dark else "light"

        reg_path, count = None, None
        up = st.file_uploader("Registry CSV", type=["csv"])
        if up:
            reg_path = common.RUNS / "_registry" / "registry.csv"
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_bytes(up.getbuffer())
        elif (ROOT / "registry.csv").is_file():
            reg_path = ROOT / "registry.csv"
        if reg_path:
            try:
                count = _count(str(reg_path), reg_path.stat().st_mtime)
            except Exception as e:
                st.error(f"Registry CSV could not be read: {e}")
                reg_path = None
        sig = (up.name, up.size) if up else None
        if sig and count is not None and st.session_state.get("reg_sig") != sig:
            st.session_state.reg_sig = sig
            st.toast(f"Registry CSV uploaded: {count} entries loaded", icon="✅")
        st.metric("Registry entries", count if count is not None else "-")
        if count is None:
            st.caption("No registry loaded. It is not needed in mock mode.")
        st.divider()
        fps = st.slider("Frame sampling rate (fps)", 1.0, 15.0, float(D.sample_fps), 0.5)
        known = st.slider("Minimum known characters", 1, 10, int(D.min_known_chars))
        thr = st.slider("No-match threshold", 0.0, 1.0, float(D.no_match_threshold), 0.01)
        debug = st.toggle("Show debug view", value=False)
        mock = st.toggle("Use mock data", value=True)
        scenario = (
            st.radio("Mock scenario", list(common.MOCK_FILES), disabled=not mock) if mock else list(common.MOCK_FILES)[0]
        )
    return Settings(sample_fps=fps, min_known_chars=known, no_match_threshold=thr), reg_path, debug, mock, scenario


settings, registry, debug, use_mock, scenario = sidebar()

# Keep the shown result in line with the mock switch (mock loading is separate from the live pipeline)
src = st.session_state.get("source", "")
if use_mock and src != f"mock:{scenario}":
    try:
        common.set_result(common.load_mock(str(common.MOCK_FILES[scenario])), f"mock:{scenario}")
        if src:
            st.toast(f"Loaded mock scenario: {scenario}", icon="ℹ️")
    except Exception as e:
        st.error(f"Could not load the mock file: {e}")
elif not use_mock and src.startswith("mock"):
    common.clear_result()

LOGO = (
    '<svg width="40" height="40" viewBox="0 0 40 40" fill="none">'
    '<circle cx="20" cy="20" r="15" stroke="#1d4ed8" stroke-width="3"/>'
    '<circle cx="20" cy="20" r="6" fill="#0d9488"/>'
    '<path d="M20 2v7M20 31v7M2 20h7M31 20h7" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round"/>'
    '</svg>'
)
st.markdown(
    f'<div class="tv-hero">{LOGO}<div><div class="tv-brand">Trace<span>Vision</span></div>'
    f'<div class="tv-tag">AI-Powered Vehicle Identification &amp; Traceability</div></div></div>',
    unsafe_allow_html=True,
)
stats_slot = st.container()
st.markdown('<div class="tv-step">Step 1: Upload and input</div>', unsafe_allow_html=True)
input_step.render(settings, use_mock, scenario, registry)

res = st.session_state.get("result")
with stats_slot:
    stats.render(res)

if res is None:
    st.info("Run an analysis to see the evidence and results.")
    viewer.maybe_render([])  # nothing to show
else:
    source = st.session_state.get("source", "")
    label = "Mock demo data (not produced by the AI pipeline)" if source.startswith("mock") else "Live pipeline"
    if source.startswith("mock"):
        st.warning("Showing supplied mock data, not a real analysis.")
    run_dir = st.session_state.get("run_dir", str(common.RUNS))
    st.markdown('<div class="tv-step">Step 2: Evidence</div>', unsafe_allow_html=True)
    evidence.render(res, run_dir, debug)
    st.markdown('<div class="tv-step">Step 3: Results</div>', unsafe_allow_html=True)
    results.render(res, run_dir, label, debug)
    # The fullscreen viewer is rendered last so it overlays everything when open.
    viewer.maybe_render(viewer.entries(res))
