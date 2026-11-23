"""Step 1: upload / input."""
from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.components import common, integration
from schema import Settings, UserHints

STAGES = ["Detecting vehicles", "Tracking", "Reading plates", "Fusing evidence", "Matching registry"]
COLOURS = ["(not specified)", "white", "silver", "grey", "black", "red", "blue", "green", "yellow", "brown", "orange"]
TYPES = ["(not specified)", "car", "SUV", "hatchback", "sedan", "two-wheeler", "auto-rickshaw", "truck", "bus"]
EXTS = {"mp4", "avi", "mov"}


def _probe(path: Path):
    """Check that OpenCV can open the video and read a frame."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok, _ = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        return ok
    except Exception:
        return False


def _hints(plate: str, colour: str, body: str) -> UserHints:
    return UserHints(partial_plate=plate.strip().upper() or None,
                     colour=None if colour == COLOURS[0] else colour,
                     body_type=None if body == TYPES[0] else body)


def render(settings: Settings, use_mock: bool, scenario: str, registry_path):
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        up = st.file_uploader("CCTV video (MP4, AVI or MOV)", type=sorted(EXTS))
        if up:
            st.caption(f"{up.name} - {up.size / 1e6:.1f} MB")
            st.video(up)
        elif use_mock:
            st.info("Mock mode: no video is needed. The supplied JSON is loaded directly.")
    with c2:
        plate = st.text_input("Partial plate hint (optional)", placeholder="CG04??1234", help="Use ? or * for unknown characters.")
        colour = st.selectbox("Vehicle colour (optional)", COLOURS)
        body = st.selectbox("Vehicle type (optional)", TYPES)
        go = st.button("Analyse Footage", type="primary", use_container_width=True)
        if use_mock:
            st.caption("Hints and settings do not change mock results; they apply to live runs.")

    if not go:
        return
    if plate.strip() and any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?*" for ch in plate.strip().upper()):
        return st.error("Plate hint can only contain letters, digits, ? and *.")
    hints = _hints(plate, colour, body)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = common.RUNS / run_id

    video_path = None
    if up:
        if up.name.rsplit(".", 1)[-1].lower() not in EXTS:
            return st.error("Unsupported format. Upload an MP4, AVI or MOV file.")
        dest = (run_dir if not use_mock else common.RUNS / "_uploads") / "input"
        dest.mkdir(parents=True, exist_ok=True)
        video_path = dest / Path(up.name).name.replace("/", "_")
        video_path.write_bytes(up.getbuffer())
        if not _probe(video_path):
            return st.error("This video could not be opened. It may be corrupted; try re-exporting it.")
    elif not use_mock:
        return st.error("Upload a video first.")

    if use_mock:
        with st.status("Loading mock result...", expanded=True) as s:
            for name in STAGES:
                st.write(f"Done (mock): {name}")
                time.sleep(0.2)
            try:
                result = common.load_mock(str(common.MOCK_FILES[scenario]))
            except Exception as e:
                s.update(label="Mock file problem", state="error")
                return st.error(f"Could not load the mock JSON: {e}")
            common.set_result(result, f"mock:{scenario}")
            s.update(label="Mock result loaded (no AI was run)", state="complete")
            st.toast("Analysis complete (mock data, no AI was run)", icon="✅")
        return

    if registry_path is None:
        return st.error("No registry CSV. Upload one in the sidebar.")
    with st.status("Analysing footage...", expanded=True) as s:
        try:
            st.write("Detecting vehicles, tracking and reading plates...")
            vision = integration.run_vision_step(video_path, settings, run_dir)
            st.write("Fusing evidence and matching registry...")
            result = integration.analyse_step(vision, registry_path, settings, hints, run_dir)
        except integration.PipelineNotReady as e:
            s.update(label="Live pipeline not available", state="error")
            return st.error(str(e))
        except Exception as e:
            s.update(label="Pipeline error", state="error")
            st.error(f"The pipeline failed: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            return
        common.set_result(result, "live")
        s.update(label="Analysis complete", state="complete")
        st.toast("Analysis complete", icon="✅")
