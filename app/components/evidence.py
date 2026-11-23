"""Step 2: compact evidence panel with expandable details."""
from __future__ import annotations

import streamlit as st

from app.components import viewer
from app.components.common import SHOW_VEHICLE_DETAILS, STRETCH, esc, html_table, safe_hex, show_image
from schema import AnalysisResult, UNKNOWN

PER_PAGE = 8


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


def _frame_card(fr, run_dir, enhanced, items):
    if not fr.used:  # CSS greys out the whole card
        st.markdown('<span class="tv-skip"></span>', unsafe_allow_html=True)
    rel = fr.enhanced_crop_path if (enhanced and fr.enhanced_crop_path) else fr.plate_crop_path
    if rel:
        show_image(run_dir, rel)
    else:
        st.markdown('<div class="tv-ph">No plate crop saved</div>', unsafe_allow_html=True)
    pill = _pill("Used", "ok") if fr.used else _pill("Skipped", "bad")
    st.markdown(f"**Frame {fr.frame_idx}** at {fr.timestamp_s:.1f} s  \nSharpness {fr.sharpness:.0f}, weight {fr.quality_weight:.3f}  \n{pill}", unsafe_allow_html=True)
    if fr.skip_reason:
        st.caption(fr.skip_reason)
    if not fr.chars:
        st.caption("No characters read")
    if st.button("Open in viewer", key=f"vw_{fr.frame_idx}", disabled=not rel, **STRETCH):
        viewer.open_at(items, rel, run_dir)


def _gallery(v, run_dir, items):
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("Show frames", ["All", "Used", "Skipped"], horizontal=True, key="fr_mode")
    enhanced = c2.toggle("Enhanced crops", key="show_enh")
    shown = [f for f in v.frames if mode == "All" or (mode == "Used") == f.used]
    if not shown:
        return st.info("No frames match this filter.")
    pages = -(-len(shown) // PER_PAGE)
    page = st.number_input("Page", min_value=1, max_value=pages, step=1, key=f"fr_page_{mode}_{pages}") if pages > 1 else 1
    chunk = shown[(page - 1) * PER_PAGE: page * PER_PAGE]
    for r in range(0, len(chunk), 4):
        for col, fr in zip(st.columns(4), chunk[r:r + 4]):
            with col, st.container(border=True):
                _frame_card(fr, run_dir, enhanced, items)


def render(result: AnalysisResult, run_dir: str, debug: bool):
    v, f = result.vision, result.fusion
    if not v.frames or v.track_id == -1:
        st.warning("No vehicle was detected in this clip, so there is no evidence to show.")
        for n in v.notes:
            st.info(n)
        return
    items = viewer.entries(result)
    used = sum(1 for x in v.frames if x.used)

    a, b = st.columns([1, 2.2], gap="large")
    with a:
        show_image(run_dir, v.vehicle_crop_path, "Target vehicle")
        if v.vehicle_crop_path and st.button("Open in viewer", key="vw_vehicle", **STRETCH):
            viewer.open_at(items, v.vehicle_crop_path, run_dir)
    with b:
        if f.known_count == 0:
            st.warning("No plate characters could be read reliably.")
        st.markdown(f'<div class="tv-card"><div class="tv-note">Recovered plate pattern: {f.known_count} of {f.length} characters known</div>{fused_html(f)}'
                    f'<div class="tv-legend">{_pill("80% or more", "ok")} {_pill("50% to 79%", "warn")} {_pill("Unknown, never guessed", "grey")}</div>'
                    f'<div class="tv-note">{esc(f.format_note or "")}</div></div>', unsafe_allow_html=True)
        meta = f"**Clip:** {esc(v.clip_name)}  \n**Duration:** {v.duration_s:.1f} s at {v.fps:g} fps, analysed at {v.sampled_fps:g} fps  \n**Track:** {v.track_id} ({esc(v.target_selection_reason)})"
        if SHOW_VEHICLE_DETAILS:
            meta += f"  \n**Vehicle class:** {esc(v.vehicle_class)}"
        st.markdown(meta)
        for n in v.notes:
            st.caption(n)

    with st.expander(f"Frames ({used} used, {len(v.frames) - used} skipped)"):
        _gallery(v, run_dir, items)
    with st.expander("Per-frame readings"):
        html_table([{"Frame": r.frame_idx, "Time (s)": r.timestamp_s, "Raw text": r.raw_text or "(empty)", "Sharpness": round(r.sharpness, 1),
                     "Status": _pill("Used", "ok") if r.used else _pill("Skipped", "bad"),
                     "Character confidences": ", ".join(f"{c.char} {c.conf:.2f}" for c in r.chars) or "No characters read",
                     "Note": r.skip_reason or "", "_cls": "" if r.used else "tv-dim"} for r in v.frames], raw={"Status"})
    with st.expander("Plate recovery details (votes per position)"):
        html_table([{"Position": c.position, "Result": c.char, "Status": c.status, "Support": c.confidence, "Share": c.share,
                     "Leading candidate": c.top_candidate or "", "Votes": ", ".join(f"{k} {x:.3f}" for k, x in c.votes.items())} for c in f.chars])
    if SHOW_VEHICLE_DETAILS:
        with st.expander("Vehicle attributes"):
            at = result.attributes
            st.markdown(f'<span class="tv-swatch" style="background:{safe_hex(at.colour.hex)}"></span>**Colour: {esc(at.colour.value)}** ({at.colour.confidence:.0%})  \n'
                        f"**Body type: {esc(at.body_type.value)}** ({at.body_type.confidence:.0%})", unsafe_allow_html=True)
            if at.make_model:
                st.markdown(f"**Make/model: {esc(at.make_model.value)}** ({at.make_model.confidence:.0%})")
