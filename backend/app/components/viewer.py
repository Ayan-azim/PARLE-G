"""Fullscreen evidence image viewer.

Opens as a true fullscreen Streamlit dialog with zoom in/out buttons,
Previous/Next navigation, an explicit Close button, and keyboard
shortcuts (left/right for navigation, + / - for zoom, Esc to close).

Architecture
------------
* ``open_at(items, rel, run_dir)`` is called from the "Open in viewer"
  buttons. It sets ``viewer_open=True``, picks the index, and reruns.
* ``maybe_render(items, run_dir)`` is called once per script run (from
  ``main.py``) and only invokes the dialog when ``viewer_open`` is True.
* The dialog body's buttons update session state and rerun.
"""
from __future__ import annotations

import base64

import streamlit as st

from app.components.common import STRETCH, esc, resolve_image


def entries(result):
    """List of (caption, relative_path) for every evidence image the result mentions."""
    v, out = result.vision, []
    if v.vehicle_crop_path:
        out.append(("Vehicle crop", v.vehicle_crop_path))
    for f in v.frames:
        if f.plate_crop_path:
            out.append((f"Frame {f.frame_idx}: plate crop at {f.timestamp_s:.1f} s", f.plate_crop_path))
        if f.enhanced_crop_path:
            out.append((f"Frame {f.frame_idx}: enhanced crop", f.enhanced_crop_path))
    return out


def _clamp_idx(i: int, n: int) -> int:
    return i % n if n else 0


def _step(delta: int, n: int):
    st.session_state.viewer_idx = _clamp_idx(st.session_state.get("viewer_idx", 0) + delta, n)


def _set_zoom(delta: int):
    cur = st.session_state.get("vw_zoom", 100)
    st.session_state.vw_zoom = max(100, min(400, cur + delta))


def _reset_zoom():
    st.session_state.vw_zoom = 100


def _close():
    """Clear the open flag; the next run will skip the dialog entirely."""
    st.session_state.pop("viewer_open", None)


def _kb_fragment() -> str:
    """Tiny HTML snippet that listens for keyboard shortcuts inside the dialog."""
    return (
        '<span class="tv-viewer-fs-marker" data-tv-viewer="1"></span>'
        '<script>'
        'document.addEventListener("keydown", function(e){'
        'var k=e.key;'
        'if(k==="ArrowRight"){document.querySelector("[data-testid=stBaseButton-secondary][aria-label=Next]],[data-testid=stButton-secondary] button")&&0;}'
        '});'
        '</script>'
    )


@st.dialog("Evidence viewer", width="large")
def _dialog(items, run_dir):
    n = len(items)
    if n == 0:
        st.markdown('<div class="tv-vw-empty">No evidence images available for this run.</div>', unsafe_allow_html=True)
        st.button("Close", key="vw_close_empty", on_click=_close, type="primary", **STRETCH)
        return
    i = _clamp_idx(st.session_state.get("viewer_idx", 0), n)
    caption, rel = items[i]

    # Marker element so global CSS turns this dialog into a fullscreen overlay.
    st.markdown('<span class="tv-viewer-fs-marker" data-tv-viewer="1"></span>', unsafe_allow_html=True)

    # Top toolbar: Previous / caption + counter / Next / Close
    top_cols = st.columns([1, 2, 1, 1])
    top_cols[0].button("◀ Previous", on_click=_step, args=(-1, n), key="vw_prev",
                       disabled=n <= 1, **STRETCH,
                       help="Previous image (Left arrow)")
    top_cols[1].markdown(
        f'<div style="text-align:center">'
        f'<div style="font-weight:600;color:var(--text)">{esc(caption)}</div>'
        f'<div class="tv-vw-counter">Image {i + 1} of {n}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    top_cols[2].button("Next ▶", on_click=_step, args=(1, n), key="vw_next",
                       disabled=n <= 1, **STRETCH,
                       help="Next image (Right arrow)")
    top_cols[3].button("✕ Close", on_click=_close, key="vw_close", type="primary",
                       help="Close the viewer (Esc)", **STRETCH)

    # Zoom toolbar: slider + Zoom out / Reset / Zoom in
    zc = st.columns([4, 1, 1, 1])
    zoom = zc[0].slider("Zoom", 100, 400, 100, 25,
                        format="%d%%", key="vw_zoom", help="Drag to zoom (use + / -)")
    zc[1].button("−", on_click=_set_zoom, args=(-25,), key="vw_zout", **STRETCH,
                 help="Zoom out (-)")
    zc[2].button("Reset", on_click=_reset_zoom, key="vw_zreset", **STRETCH,
                 help="Reset zoom to 100%")
    zc[3].button("+", on_click=_set_zoom, args=(25,), key="vw_zin", **STRETCH,
                 help="Zoom in (+)")

    # The image stage: scrollable, checkerboard background so transparent crops are visible.
    p = resolve_image(run_dir, rel)
    if p is None:
        st.markdown('<div class="tv-vw-empty">Image not available for this run.</div>', unsafe_allow_html=True)
        st.caption("Tip: drop a file with the same name into app/mock/sample_images/<run_id>/ to display it.")
        return
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    uri = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    st.markdown(
        f'<div class="tv-vw-stage"><img src="{uri}" style="width:{zoom}%" alt="{esc(caption)}"/></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Scroll inside the image when zoomed in. "
        "Shortcuts: ← / → to navigate, + / − to zoom, Esc to close. "
        "This is mock sample data, not real evidence."
    )

    # Hidden keyboard-shortcut helper: posts messages back into Streamlit via virtual clicks.
    st.components.v1.html(
        """
        <script>
        (function(){
          function click(label){
            var btns = parent.document.querySelectorAll('button[kind="secondary"], button[data-testid="stBaseButton-secondary"], button[kind="primary"], button[data-testid="stBaseButton-primary"]');
            for (var i=0;i<btns.length;i++){
              var t = (btns[i].innerText||'').trim();
              if (t === label){ btns[i].click(); return true; }
            }
            return false;
          }
          document.addEventListener('keydown', function(e){
            var k = e.key;
            if (k === 'ArrowLeft'){ click('◀ Previous'); }
            else if (k === 'ArrowRight'){ click('Next ▶'); }
            else if (k === '+' || k === '='){ click('+'); }
            else if (k === '-' || k === '_'){ click('−'); }
            else if (k === 'Escape'){ click('✕ Close'); }
          });
        })();
        </script>
        """,
        height=0,
    )


def open_at(items, rel, run_dir):
    """Trigger the viewer to open at the image matching ``rel``.

    Sets the viewer_open flag and reruns. The dialog itself is rendered
    by maybe_render() on the next run. Zoom level is left to the slider
    widget's own state to avoid the session_state pre-set warning.
    """
    if not items:
        return
    st.session_state.viewer_idx = next((k for k, (_, r) in enumerate(items) if r == rel), 0)
    st.session_state.viewer_open = True
    # The actual dialog is rendered on the next run by maybe_render().
    st.rerun()


def maybe_render(items):
    """Render the fullscreen dialog if it has been opened."""
    if not st.session_state.get("viewer_open", False):
        return
    run_dir = st.session_state.get("run_dir", None)
    if run_dir is None:
        _close()
        return
    _dialog(items or [], run_dir)
