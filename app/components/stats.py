"""Animated statistics cards.

Four cards: Total Vehicles, Matches Found, No Match, Processing Time.
Each card has a colored top accent, an SVG icon, and a number that
counts up from 0 to its final value over ~900ms (eased cubic).
Respects prefers-reduced-motion.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app.components.common import palette


# Compact SVG icons (no external font dependency).
ICON_CAR = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 13l2-5a3 3 0 0 1 3-2h8a3 3 0 0 1 3 2l2 5"/>'
    '<path d="M3 13h18v5a1 1 0 0 1-1 1h-1a2 2 0 0 1-2-2H7a2 2 0 0 1-2 2H4a1 1 0 0 1-1-1z"/>'
    '<circle cx="7.5" cy="15.5" r="1.2"/><circle cx="16.5" cy="15.5" r="1.2"/>'
    '</svg>'
)
ICON_CHECK = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>'
    '</svg>'
)
ICON_WARN = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3l9 16H3z"/><path d="M12 10v4"/><circle cx="12" cy="17" r=".7" fill="currentColor"/>'
    '</svg>'
)
ICON_CLOCK = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'
    '</svg>'
)


def render(result):
    p = palette()
    hist = st.session_state.get("history", {})
    total = len(hist)
    matches = sum(1 for h in hist.values() if h["status"] == "match")
    t = sum(result.timings_s.values()) if result is not None and result.timings_s else None

    # (label, value, decimals, suffix, sub, accent color, icon)
    cards = [
        ("Total Vehicles", total, 0, "", "this session", p["card_top_total"], ICON_CAR),
        ("Matches Found", matches, 0, "", "verdict: match", p["card_top_match"], ICON_CHECK),
        ("No Match", total - matches, 0, "", "no confident match", p["card_top_nomatch"], ICON_WARN),
        ("Processing Time", t, 1, " s", "current run" if t is not None else "no timing data",
         p["card_top_time"], ICON_CLOCK),
    ]

    cards_html = "".join(
        f'<div class="c" style="--accent-c:{col}">'
        f'<div class="c-top"></div>'
        f'<div class="c-body">'
        f'<div class="c-head"><span class="c-icon">{icon}</span><span class="l">{lab}</span></div>'
        f'<div class="n" data-v="{"" if v is None else v}" data-d="{d}" data-s="{s}">0</div>'
        f'<div class="s">{sub}</div>'
        f'</div></div>'
        for lab, v, d, s, sub, col, icon in cards
    )

    page = f"""<style>
body {{ margin:0; font-family:Inter,'Segoe UI',system-ui,sans-serif; }}
.g {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
@media (max-width:720px) {{ .g {{ grid-template-columns:repeat(2,1fr); }} }}
.c {{
  position:relative;
  background:{p['card']};
  border:1px solid {p['line']};
  border-radius:14px;
  box-shadow:0 1px 3px {p['shadow']};
  overflow:hidden;
  transition:transform .22s ease, box-shadow .25s ease, border-color .25s ease;
}}
.c:hover {{
  transform:translateY(-4px);
  box-shadow:0 14px 30px {p['shadow']};
  border-color:var(--accent-c);
}}
.c-top {{
  height:5px;
  background:linear-gradient(90deg, var(--accent-c), color-mix(in srgb, var(--accent-c) 50%, transparent));
}}
.c-body {{ padding:14px 16px; }}
.c-head {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.c-icon {{ color:var(--accent-c); display:inline-flex; }}
.l {{ font-size:.82rem; color:{p['muted']}; font-weight:600; letter-spacing:.02em; text-transform:uppercase; }}
.n {{ font-size:2.1rem; font-weight:700; color:{p['text']}; margin:2px 0; transition:color .25s; }}
.s {{ font-size:.76rem; color:{p['muted']}; }}
@keyframes pop {{ 0% {{ transform:scale(.6); opacity:0; }} 60% {{ transform:scale(1.08); opacity:1; }} 100% {{ transform:scale(1); }} }}
.c:hover .c-icon {{ animation: pop .45s ease both; }}
</style>
<div class="g">{cards_html}</div>
<script>
document.querySelectorAll('[data-v]').forEach(function(el){{
  var v=parseFloat(el.dataset.v), d=+el.dataset.d, s=el.dataset.s||'';
  if (isNaN(v)) {{ el.textContent='n/a'; el.style.opacity='0.7'; return; }}
  var fmt=function(x){{ return x.toFixed(d)+s; }};
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {{ el.textContent=fmt(v); return; }}
  var t0=performance.now(), dur=900;
  (function tick(t){{
    var k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3);
    el.textContent=fmt(v*e);
    if (k<1) requestAnimationFrame(tick);
    else el.textContent=fmt(v);
  }})(t0);
}});
</script>"""
    components.html(page, height=130)
