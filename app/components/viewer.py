"""Evidence image viewer: opens in a large dialog with zoom and previous/next."""
from __future__ import annotations

import base64

import streamlit as st

from app.components.common import STRETCH, esc, resolve_image


def entries(result):
    """List of (caption, relative_path) for every evidence image the result mentions."""
    v, out = getattr(result, "vision", result), []
    if v.vehicle_crop_path:
        out.append(("Vehicle crop", v.vehicle_crop_path))
    for f in v.frames:
        if f.plate_crop_path:
            out.append((f"Frame {f.frame_idx}: plate crop at {f.timestamp_s:.1f} s", f.plate_crop_path))
        if f.enhanced_crop_path:
            out.append((f"Frame {f.frame_idx}: enhanced crop", f.enhanced_crop_path))
    return out


def _step(delta: int, n: int):
    st.session_state.viewer_idx = (st.session_state.get("viewer_idx", 0) + delta) % n


@st.dialog("Evidence viewer", width="large")
def _dialog(items, run_dir):
    n = len(items)
    i = st.session_state.get("viewer_idx", 0) % n
    caption, rel = items[i]
    a, b, c = st.columns([1, 3, 1])
    a.button("Previous", on_click=_step, args=(-1, n), key="vw_prev", **STRETCH)
    b.markdown(f"**{esc(caption)}**  \nImage {i + 1} of {n}")
    c.button("Next", on_click=_step, args=(1, n), key="vw_next", **STRETCH)
    zoom = st.slider("Zoom", min_value=100, max_value=400, step=25, format="%d%%", key="vw_zoom")
    p = resolve_image(run_dir, rel)
    if p is None:
        st.markdown('<div class="tv-ph">Image not available</div>', unsafe_allow_html=True)
        return
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    uri = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    st.markdown(f'<div class="tv-viewer"><img src="{uri}" style="width:{zoom}%" alt="{esc(caption)}"/></div>', unsafe_allow_html=True)
    st.caption("Scroll inside the image when zoomed in.")


def open_at(items, rel, run_dir):
    st.session_state.viewer_idx = next((k for k, (_, r) in enumerate(items) if r == rel), 0)
    st.session_state.vw_zoom = 100
    _dialog(items, run_dir)
