"""Self-contained HTML (Jinja2) and PDF (xhtml2pdf) report from an AnalysisResult."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.components.common import APP_NAME, SHOW_VEHICLE_DETAILS, resolve_image, safe_hex, visible_reasons
from schema import AnalysisResult

GREEN, AMBER, GREY, RED = "#dcfce7", "#fef3c7", "#e5e7eb", "#fee2e2"


class ReportError(RuntimeError):
    pass


def _uri(path: Path | None):
    if not path:
        return None
    try:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x: "n/a" if x is None else f"{x * 100:.0f}%"
    env.globals["fused_bg"] = lambda ch: GREY if ch.status != "known" else (GREEN if ch.confidence >= 0.8 else AMBER)
    env.globals["match_bg"] = lambda s: {"match": GREEN, "confusable": AMBER, "mismatch": RED}.get(s, GREY)
    env.globals["safe_hex"] = safe_hex
    env.globals["show_details"] = SHOW_VEHICLE_DETAILS
    env.globals["app_name"] = APP_NAME
    env.filters["visible"] = visible_reasons
    return env


def build_html(result: AnalysisResult, run_dir: str, source: str = "") -> str:
    paths = [result.vision.vehicle_crop_path]
    for f in result.vision.frames:
        paths += [f.plate_crop_path, f.enhanced_crop_path]
    images = {p: u for p in paths if p and (u := _uri(resolve_image(run_dir, p)))}
    return _env().get_template("template.html").render(r=result, images=images, source=source or "unspecified")


def build_pdf(html: str) -> bytes:
    try:
        from xhtml2pdf import pisa
    except ImportError as e:
        raise ReportError("xhtml2pdf is not installed") from e
    buf = io.BytesIO()
    try:
        status = pisa.CreatePDF(src=html, dest=buf, encoding="utf-8")
    except Exception as e:
        raise ReportError(str(e)) from e
    if status.err:
        raise ReportError("the PDF engine reported errors")
    return buf.getvalue()
