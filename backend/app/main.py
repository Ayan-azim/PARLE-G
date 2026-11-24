"""TraceVision - Person C UI. Run with:  streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

_icon = ROOT / "app" / "assets" / "favicon.png"
st.set_page_config(page_title="VEHICLE TRACKER", page_icon=str(_icon) if _icon.is_file() else "🔍", layout="wide")

from app.components import common, evidence, input_step, results, stats, viewer  # noqa: E402
from schema import Settings  # noqa: E402

# Theme preference: kept in session state AND in the URL (?theme=dark) so it survives refresh.
if "dark" not in st.session_state:
    qp = st.query_params.get("theme")
    st.session_state.dark = (qp == "dark")

# Inject theme variables + global styles on every run, so toggling reflects instantly.
st.markdown(
    f"<style>{(ROOT / 'app' / 'styles.css').read_text(encoding='utf-8')}\n{common.theme_css()}</style>",
    unsafe_allow_html=True,
)

D = Settings()  # defaults come from the shared schema


@st.cache_data(show_spinner=False)
def _count(path: str, mtime: float) -> int:
    return len(pd.read_csv(path))


def sidebar():
    with st.sidebar:
        st.markdown(f'<div class="tv-side-brand">{common.logo_svg(38, "s")}<div><div class="tv-brand">VEHICLE <span>TRACKER</span></div>'
                    f'<div class="tv-tag">Vehicle identification</div></div></div>', unsafe_allow_html=True)
        st.toggle("Dark mode", key="dark", help="Switch between dark and light themes.")
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
    return D, reg_path, False, False, list(common.MOCK_FILES)[0]


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

st.markdown(f'<div class="tv-hero">{common.logo_svg(64, "h")}<div><div class="tv-brand">VEHICLE <span>TRACKER</span></div>'
            f'<div class="tv-tag">{common.TAGLINE}</div><span class="tv-chip">Plate recovery</span><span class="tv-chip">Evidence fusion</span>'
            f'<span class="tv-chip">Registry matching</span><span class="tv-chip">Court-ready reports</span></div><span class="tv-rec"><i></i>SYSTEM READY</span></div>', unsafe_allow_html=True)
stats_slot = st.container()
st.markdown('<div class="tv-step"><span class="tv-num">1</span>Upload and input</div>', unsafe_allow_html=True)
input_step.render(settings, use_mock, scenario, registry)

res = st.session_state.get("result")
vehicle_results = st.session_state.get("vehicle_results", [])
if len(vehicle_results) > 1:
    options = list(range(len(vehicle_results)))
    selected = st.selectbox(
        "Vehicle track",
        options,
        index=min(st.session_state.get("active_vehicle", 0), len(options) - 1),
        format_func=lambda i: (
            f"Track {vehicle_results[i].vision.track_id} · "
            f"{vehicle_results[i].fusion.pattern or 'plate unreadable'} · "
            f"{vehicle_results[i].verdict.headline}"
        ),
        key=f"vehicle_select_{vehicle_results[0].run_id}",
    )
    st.session_state.active_vehicle = selected
    res = vehicle_results[selected]
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
    st.markdown('<div class="tv-step"><span class="tv-num">2</span>Evidence</div>', unsafe_allow_html=True)
    st.markdown(common.case_badges(res), unsafe_allow_html=True)
    evidence.render(res, run_dir, debug)
    st.markdown('<div class="tv-step"><span class="tv-num">3</span>Results</div>', unsafe_allow_html=True)
    results.render(res, run_dir, label, debug)
    # The fullscreen viewer is rendered last so it overlays everything when open.
    viewer.maybe_render(viewer.entries(res))
