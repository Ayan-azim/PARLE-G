"""Animated statistics cards (numbers count up). Rendered as a small HTML component."""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app.components.common import palette

ICONS = {
    "stack": '<path d="M12 3 3 8l9 5 9-5-9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "slash": '<circle cx="12" cy="12" r="9"/><path d="m6 6 12 12"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
}


def render(result):
    p = palette()
    hist = st.session_state.get("history", {})
    total = len(hist)
    matches = sum(1 for h in hist.values() if h["status"] == "match")
    nomatch = total - matches
    t = sum(result.timings_s.values()) if result is not None and result.timings_s else None
    rate = f"{matches / total:.0%} of analyses" if total else "no analyses yet"
    cards = [("Total Analyses", total, 0, "", "this session", p["accent"], "stack"),
             ("Matches Found", matches, 0, "", rate, p["ok_bd"], "check"),
             ("No Match", nomatch, 0, "", f"{nomatch / total:.0%} of analyses" if total else "no analyses yet", p["warn_bd"], "slash"),
             ("Processing Time", t, 1, " s", "current run, all stages" if t is not None else "no timing data", p["teal"], "clock")]
    html = "".join(
        f'<div class="c" style="--k:{col}"><svg class="i" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{col}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[ic]}</svg>'
        f'<div class="l">{lab}</div><div class="n" data-v="{"" if v is None else v}" data-d="{d}" data-s="{s}">0</div><div class="s">{sub}</div></div>'
        for lab, v, d, s, sub, col, ic in cards)
    page = f"""<style>
body{{margin:0;font-family:Inter,'Segoe UI',system-ui,sans-serif}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:3px}}
@media(max-width:640px){{.g{{grid-template-columns:repeat(2,1fr)}}}}
.c{{position:relative;background:{p['card']};border:1px solid {p['line']};border-radius:14px;padding:16px 18px;box-shadow:0 1px 3px {p['shadow']};overflow:hidden;transition:transform .2s,box-shadow .2s,border-color .2s}}
.c:before{{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--k)}}
.c:hover{{transform:translateY(-3px);box-shadow:0 12px 26px {p['shadow']};border-color:var(--k)}}
.i{{position:absolute;right:14px;top:16px;opacity:.85}}
.l{{font-size:.8rem;color:{p['muted']};font-weight:500}} .n{{font-size:2.1rem;font-weight:800;color:{p['text']};margin:3px 0;font-family:'Plus Jakarta Sans',Inter,sans-serif}} .s{{font-size:.75rem;color:{p['muted']}}}
</style><div class="g">{html}</div><script>
document.querySelectorAll('[data-v]').forEach(function(el){{
 var v=parseFloat(el.dataset.v),d=+el.dataset.d,s=el.dataset.s||'';
 if(isNaN(v)){{el.textContent='n/a';return}}
 var f=function(x){{return x.toFixed(d)+s}};
 if(matchMedia('(prefers-reduced-motion: reduce)').matches){{el.textContent=f(v);return}}
 var t0=performance.now();
 (function tick(t){{var k=Math.min(1,(t-t0)/900),e=1-Math.pow(1-k,3);el.textContent=f(v*e);if(k<1)requestAnimationFrame(tick)}})(t0);
}});
</script>"""
    components.html(page, height=140)
