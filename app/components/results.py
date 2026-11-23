"""Step 3: verdict, candidates, weak leads, rule checks, warnings, downloads."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.components.common import SHOW_REGISTRY_DETAILS, esc, html_table, palette
from app.report.build_report import ReportError, build_html, build_pdf
from schema import AnalysisResult, Candidate, Verdict

LEGEND = ('<div class="tv-legend"><span class="tv-pill tv-pill-ok">match</span> <span class="tv-pill tv-pill-warn">confusable</span> '
          '<span class="tv-pill tv-pill-bad">mismatch</span> <span class="tv-pill tv-pill-grey">unknown</span> '
          '&nbsp; Top: observed in footage. Bottom: registry plate.</div>')


def verdict(v: Verdict):
    if v.status == "match":
        conf = f"Confidence {v.top_confidence:.0%}" if v.top_confidence is not None else ""
        st.markdown(f'<div class="tv-card tv-verdict tv-verdict-match"><span class="tv-pill tv-pill-ok">Possible match</span>'
                    f'<h3>{esc(v.headline)}</h3><div>{esc(v.reason)}</div><div class="tv-note">{conf}. A lead for human verification, not proof of identity.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="tv-card tv-verdict tv-verdict-none"><span class="tv-pill tv-pill-warn">No Confident Match</span>'
                    f'<h3>{esc(v.headline)}</h3><div>{esc(v.reason)}</div></div>', unsafe_allow_html=True)


def _candidate(c: Candidate, key: str, open_: bool):
    with st.expander(f"#{c.rank}  {c.plate}  -  {c.confidence:.0%}", expanded=open_):
        a, b = st.columns([1, 1.4], gap="large")
        with a:
            st.markdown(f'<span class="tv-pill tv-pill-ok">Rank {c.rank}</span>', unsafe_allow_html=True)
            st.markdown(f"### {esc(c.plate)}")
            if SHOW_REGISTRY_DETAILS:
                name = " ".join(x for x in (c.make, c.model) if x) or "Make/model not recorded"
                st.markdown(f"**{esc(name)}**  \nRegistry colour: {esc(c.registry_colour)}  \nRegistry type: {esc(c.registry_type)}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{c.confidence:.0%}")
            m2.metric("Evidence", f"{c.breakdown.evidence_factor:.2f}")
            m3.metric("Margin", f"{c.breakdown.margin_factor:.2f}")
            st.caption("Confidence = final score x evidence factor x margin factor")
        with b:
            bd = c.breakdown
            labels, vals = ["Plate", "Colour", "Type", "Final"], [bd.plate_score, bd.colour_score, bd.type_score, bd.final_score]
            pal = palette()
            fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", text=[f"{x:.2f}" for x in vals], textposition="outside",
                                   marker_color=[pal["grey_bd"], pal["grey_bd"], pal["grey_bd"], pal["accent"]]))
            fig.update_layout(height=200, margin=dict(l=0, r=30, t=5, b=0), xaxis=dict(range=[0, 1.15], gridcolor=pal["line"]), yaxis=dict(autorange="reversed"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=pal["text"]))
            st.plotly_chart(fig, use_container_width=True, key=f"bd_{key}_{c.rank}")
        cells = "".join(f'<div class="tv-m tv-m-{m.status}" title="position {m.position}: {m.status}"><b>{esc(m.observed)}</b><i>{esc(m.registry)}</i></div>' for m in c.char_matches)
        st.markdown(f'<div class="tv-chars">{cells}</div>{LEGEND}', unsafe_allow_html=True)
        for r in c.reasons:
            st.markdown(f"- {r}")


def _summary(cs):
    return [{"Rank": c.rank, "Plate": c.plate, "Confidence": f"{c.confidence:.0%}", "Plate score": c.breakdown.plate_score,
             "Colour score": c.breakdown.colour_score, "Type score": c.breakdown.type_score, "Final score": c.breakdown.final_score} for c in cs]


def _toast(msg: str):
    st.toast(msg, icon="✅")


@st.cache_data(show_spinner=False)
def _html(result_json: str, run_dir: str, source: str) -> str:
    return build_html(AnalysisResult.model_validate_json(result_json), run_dir, source)


def render(result: AnalysisResult, run_dir: str, source: str, debug: bool):
    v = result.verdict
    st.subheader("A. Verdict")
    verdict(v)

    st.subheader("B. Top candidates")
    if v.status == "match":
        if not result.candidates:
            st.warning("The verdict reports a match but the result contains no candidates.")
        else:
            html_table(_summary(result.candidates[:5]))
            for c in result.candidates[:5]:
                _candidate(c, "cand", c.rank == 1)
    else:
        st.info("No candidate is presented as a match for this clip.")
        if result.candidates:
            with st.expander("Candidates supplied by the pipeline (not a match, shown for transparency)"):
                for c in result.candidates[:5]:
                    _candidate(c, "cand_nm", False)

    if v.status == "no_confident_match" and result.weak_leads:
        st.subheader("C. Weak leads")
        st.warning("These are weak leads, not confirmed matches. They fit only a few readable characters and must not be treated as identification.")
        for c in result.weak_leads:
            _candidate(c, "weak", False)

    st.subheader("D. Rule checks")
    html_table([{"Rule": r.name, "Result": '<span class="tv-pill tv-pill-ok">Passed</span>' if r.passed else '<span class="tv-pill tv-pill-bad">Failed</span>',
                 "Explanation": r.detail} for r in v.rule_checks], raw={"Result"})

    st.subheader("E. Warnings")
    if result.warnings:
        for w in result.warnings:
            st.warning(w)
    else:
        st.caption("No warnings.")

    st.markdown('<div class="tv-step">Report</div>', unsafe_allow_html=True)
    try:
        html = _html(result.model_dump_json(), run_dir, source)
    except Exception as e:
        st.error(f"The report could not be built: {e}")
        html = None
    c1, c2, c3 = st.columns(3)
    rid = result.run_id
    if html:
        c1.download_button("Download Report (HTML)", html.encode("utf-8"), f"tracevision_{rid}.html", "text/html", on_click=_toast, args=("HTML report downloaded",), use_container_width=True)
        key = f"pdf_{rid}"
        if c2.button("Prepare PDF report", use_container_width=True):
            try:
                with st.spinner("Building PDF..."):
                    st.session_state[key] = build_pdf(html)
                st.toast("PDF report is ready to download", icon="✅")
            except ReportError as e:
                st.error(f"PDF could not be created ({e}). The HTML report still works.")
        if key in st.session_state:
            c2.download_button("Download Report (PDF)", st.session_state[key], f"tracevision_{rid}.pdf", "application/pdf", on_click=_toast, args=("PDF report downloaded",), use_container_width=True)
    c3.download_button("Download Raw JSON", result.model_dump_json(indent=2), f"tracevision_{rid}.json", "application/json", on_click=_toast, args=("Raw JSON downloaded",), use_container_width=True)
    if debug:
        with st.expander("Debug: timings and raw result"):
            st.json(result.timings_s)
            st.json(result.model_dump())
