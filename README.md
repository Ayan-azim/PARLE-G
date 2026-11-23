# TraceVision - Person C (Streamlit UI and Report)

AI-Powered Vehicle Identification & Traceability. This app shows an `AnalysisResult` (defined in `schema.py`) and builds the report.

## Run it (Windows, Mac, Linux; Python 3.11+)
```
python -m venv .venv
# Windows: .venv\Scripts\activate      |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```
The browser opens by itself (usually http://localhost:8501).

## Try the demo
"Use mock data" is ON by default. Pick a scenario in the sidebar:
- **Successful match** loads `app/mock/mock_result.json`
- **No confident match** loads `app/mock/mock_no_match.json`

Click **Analyse Footage** to replay the stages. No AI runs in mock mode. The original JSON files are never changed.
Plate crop images are optional: drop files with the same names (e.g. `f012_plate.png`) into `app/mock/sample_images/<run_id>/` to see them. Missing images show a grey placeholder.

## What's new in this version

### Polished theme system
- **Dark / light toggle** at the top of the sidebar, with a colored "Theme: Light/Dark" pill above it. Persists across refreshes via the URL (`?theme=dark` or `?theme=light`).
- Two complete palettes (light + dark) carrying every CSS variable the app needs. Toggle is instant — no page reload, no flash of the wrong theme.

### Subtle hover effects everywhere
- **Buttons** (primary and secondary): gentle lift (`translateY(-1px)`), colored shadow, gradient shift on primary.
- **Cards** (verdict, candidate, plate boxes, character cells): lift + colored shadow + border accent.
- **Images**: slight scale (1.02) with soft shadow.
- **Table rows, expanders, swatches, file uploader**: all themed with smooth transitions.
- Honours `prefers-reduced-motion` (all transitions disabled).

### Animated stat cards
Four cards across the top, each with a colored top accent, an SVG icon, and a value that counts up from 0 to its final value over ~900ms (eased cubic):
| Card | Color | Icon |
|---|---|---|
| Total Vehicles | blue | car |
| Matches Found | green | check-circle |
| No Match | amber | warning triangle |
| Processing Time | teal | clock |

### Fullscreen evidence viewer
Press **Open in viewer** under any image. A **true fullscreen dialog** opens (no large-but-still-modal dialog), with:
- Previous / Next navigation buttons (with arrow icons)
- Image counter ("Image X of N")
- **Zoom slider** (100% to 400%) plus **Zoom out / Reset / Zoom in** buttons
- **Close** button (top right) — also dismissable with the **Esc** key
- Keyboard shortcuts: ← / → to navigate, + / − to zoom, Esc to close
- Checkerboard background so transparent crops stay legible
- Scrollable stage when the image is zoomed in beyond the viewport

### Toast notifications
- **Analysis complete** (mock data): fires after the staged mock load finishes
- **PDF report is ready**: fires when the PDF build completes
- **Registry CSV uploaded**: fires when a CSV is parsed, shows entry count
- **Loaded mock scenario**: fires when you switch scenarios mid-session
- HTML / PDF / JSON download buttons each wire `st.toast` to their `on_click`

### Registry details hidden (UI + report)
`SHOW_REGISTRY_DETAILS = False` in `app/components/common.py` hides make/model and registry colour/type on candidate cards and in the HTML/PDF report. Schema fields and mock JSON files are unchanged — only the rendering is gated. Set it to `True` to surface them again.

## Connect Person A and Person B
Open `app/components/integration.py` and set the two names at the top:
`VISION_IMPORT` (must have `run_vision`) and `ANALYSE_IMPORT` (must have `analyse`). Then turn off "Use mock data" and upload a registry CSV.
Outputs for each live run go to `runs/<run_id>/`. Nobody edits `schema.py` except by adding Optional fields.

## Folders
- `app/main.py` — page and sidebar; `app/components/` the three steps and helpers
- `app/components/viewer.py` — fullscreen evidence viewer
- `app/components/stats.py` — animated stat cards
- `app/report/` — report template and builder (HTML and PDF)
- `app/styles.css` — global theme + hover + fullscreen-dialog styles
- `app/pages/` — intentionally empty, so Streamlit keeps a single page
- `runs/` — generated outputs (ignored by Git)

## Rules the UI follows
Unknown plate characters stay `?`. The verdict (`match` or `no_confident_match`) decides what is shown; scores never override it. Weak leads are labelled as leads, not matches. Every report says a human must verify the result.

## Common problems
- `streamlit` not found: activate the virtual environment first.
- PDF fails: the HTML report still downloads; check that `xhtml2pdf` installed.
- Viewer keyboard shortcuts do not work: click inside the dialog once so it has focus.
