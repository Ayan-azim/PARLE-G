"""Step 2: evidence panel (frames, plate recovery, per-frame table, attributes)."""
from __future__ import annotations

import streamlit as st

from app.components import viewer
from app.components.common import esc, html_table, safe_hex, show_image
from schema import AnalysisResult, UNKNOWN


def _pill(text, kind):
    return f'<span class="tv-pill tv-pill-{kind}">{esc(text)}</span>'


def fused_html(fusion) -> str:
    boxes = []
    for ch in fusion.chars:
        if ch.status == "known" and ch.char != UNKNOWN:
            cls = "tv-hi" if ch.confidence >= 0.8 else "tv-mid"
            tip = f"Known. Support {ch.confidence:.2f}, share {ch.share:.2f}"
        else:
            cls = "tv-unk"
            lead = f" Leading candidate (not used): {ch.top_candidate}." if ch.top_candidate else ""
            tip = f"Unknown.{lead}"
        boxes.append(f'<div class="tv-char {cls}" title="{esc(tip)}"><b>{esc(ch.char)}</b><small>{ch.confidence:.0%}</small></div>')
    return '<div class="tv-chars">' + "".join(boxes) + "</div>"


def render(result: AnalysisResult, run_dir: str, debug: bool):
    v, f = result.vision, result.fusion
    items = viewer.entries(result)
    a, b = st.columns([1, 2], gap="large")
    with a:
        show_image(run_dir, v.vehicle_crop_path, "Best vehicle crop")
        if v.vehicle_crop_path and st.button("Open in viewer", key="vw_vehicle", use_container_width=True):
            viewer.open_at(items, v.vehicle_crop_path, run_dir)
    with b:
        st.markdown(f"**Clip:** {esc(v.clip_name)}  \n**Duration:** {v.duration_s:.1f} s at {v.fps:g} fps, analysed at {v.sampled_fps:g} fps  \n"
                    f"**Vehicle class:** {esc(v.vehicle_class)} (track {v.track_id})  \n**Target chosen because:** {esc(v.target_selection_reason)}")
        for n in v.notes:
            st.caption(n)
    if not v.frames:
        st.warning("No vehicle or plate was detected in this clip, so there is no evidence to show.")
        return

    st.subheader("A. Frames used")
    cols = st.columns(min(4, len(v.frames)))
    for i, fr in enumerate(v.frames):
        with cols[i % len(cols)], st.container(border=True):
            show_image(run_dir, fr.plate_crop_path)
            pill = _pill("Used", "ok") if fr.used else _pill("Skipped", "bad")
            st.markdown(f"**Frame {fr.frame_idx}** at {fr.timestamp_s:.1f} s  \nSharpness {fr.sharpness:.0f}, weight {fr.quality_weight:.3f}  \n{pill}", unsafe_allow_html=True)
            if fr.skip_reason:
                st.caption(fr.skip_reason)
            if st.button("Open in viewer", key=f"vw_{fr.frame_idx}", disabled=not fr.plate_crop_path, use_container_width=True):
                viewer.open_at(items, fr.plate_crop_path, run_dir)

    st.subheader("B. Plate recovery")
    if f.known_count == 0:
        st.warning("No plate characters could be read reliably.")
    st.markdown(f'<div class="tv-card"><div class="tv-note">Fused plate pattern</div>{fused_html(f)}'
                f'<div class="tv-legend">{_pill("Green: confidence 80% or more", "ok")} {_pill("Amber: 50% to 79%", "warn")} '
                f'{_pill("Grey: unknown, never guessed", "grey")}</div>'
                f'<div class="tv-note">{f.known_count} of {f.length} characters known. Frames used: {", ".join(map(str, f.frames_used)) or "none"}. {esc(f.format_note or "")}</div></div>',
                unsafe_allow_html=True)

    st.subheader("C. Per-frame readings")
    html_table([{"Frame": r.frame_idx, "Time (s)": r.timestamp_s, "Raw text": r.raw_text, "Sharpness": round(r.sharpness, 1),
                 "Status": _pill("Used", "ok") if r.used else _pill("Skipped", "bad"),
                 "Character confidences": ", ".join(f"{c.char} {c.conf:.2f}" for c in r.chars)} for r in v.frames], raw={"Status"})

    st.subheader("D. Vehicle attributes")
    at = result.attributes
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown(f'<span class="tv-swatch" style="background:{safe_hex(at.colour.hex)}"></span>**Colour: {esc(at.colour.value)}** ({at.colour.confidence:.0%})', unsafe_allow_html=True)
        st.caption("Alternatives: " + (", ".join(f"{x.value} {x.confidence:.0%}" for x in at.colour.alternatives) or "none"))
        if at.colour.hex:
            st.caption(f"Swatch {at.colour.hex}")
    with c2, st.container(border=True):
        st.markdown(f"**Body type: {esc(at.body_type.value)}** ({at.body_type.confidence:.0%})")
        st.caption("Alternatives: " + (", ".join(f"{x.value} {x.confidence:.0%}" for x in at.body_type.alternatives) or "none"))
    with c3, st.container(border=True):
        if at.make_model:
            st.markdown(f"**Make/model: {esc(at.make_model.value)}** ({at.make_model.confidence:.0%})")
        else:
            st.caption("Make and model were not estimated for this run.")
    if debug:
        with st.expander("Debug: per-position votes"):
            html_table([{"Position": c.position, "Char": c.char, "Status": c.status, "Support": c.confidence, "Share": c.share,
                         "Top candidate": c.top_candidate or "", "Votes": str(c.votes)} for c in f.chars])
