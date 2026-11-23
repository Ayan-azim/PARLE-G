# VEHICLE TRACKER - Person C (Streamlit UI and Report)

AI-Powered Vehicle Identification & Traceability. This app shows an `AnalysisResult` (defined in `schema.py`) and builds the report.

## Run it (Windows, Python 3.11)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```
Mac/Linux: use `source .venv/bin/activate` instead of the activate line above.
The browser opens by itself (usually http://localhost:8501).

## Try the demo
"Use mock data" is ON by default. Pick a scenario in the sidebar:
- **Successful match** loads `app/mock/mock_result.json`
- **No confident match** loads `app/mock/mock_no_match.json`

Click **Analyse Footage** to replay the stages. No AI runs in mock mode. The original JSON files are never changed.
Plate crop images are optional: drop files with the same names (e.g. `f012_plate.png`) into `app/mock/sample_images/` to see them. Missing images show a grey placeholder.

## Dashboard polish
- VEHICLE TRACKER branding and a consistent indigo/cyan visual identity.
- Native Streamlit statistic cards so they inherit the selected theme and avoid embedded HTML components.
- Improved light/dark input contrast and dropdown readability.
- Compact introduction and consistent investigation-report wording.

## New in this version
- **Dark / light mode:** toggle in the sidebar. The selected theme is stored in the page address (`?theme=dark` or `?theme=light`).
- **Evidence viewer:** press "Open in viewer" under any image. Zoom slider, Previous and Next buttons.
- **Dashboard stat cards:** total analyses, matches, no-confident-match outcomes and processing time of the current run.
- **Toasts:** analysis complete, report downloads, registry upload.
- **Hidden for now:** make/model and registry colour/type on candidate cards and in the report. To show them again set `SHOW_VEHICLE_DETAILS = True` in `app/components/common.py`.
- **Demo images:** `app/mock/sample_images/` holds images drawn by `python tools/make_sample_images.py`. They are labelled MOCK SAMPLE and are not real evidence.

## Hidden vehicle details (frontend only)
Make, model, registry colour/type, observed colour/body type and vehicle class are hidden in the UI and the report. Reasons that mention them are filtered out of the text. The data is untouched in the result and the raw JSON. To show it again set `SHOW_VEHICLE_DETAILS = True` in `app/components/common.py`.

## Branding
Logo and favicon live in `app/assets/` (regenerate with `python tools/make_brand_assets.py`). Colours are in `PALETTES` in `app/components/common.py`.

## Connect Person A and Person B
Open `app/components/integration.py` and set the two names at the top:
`VISION_IMPORT` (must have `run_vision`) and `ANALYSE_IMPORT` (must have `analyse`). Then turn off "Use mock data" and upload a registry CSV.
Outputs for each live run go to `runs/<run_id>/`. Nobody edits `schema.py` except by adding Optional fields.

## Folders
- `app/main.py` page and sidebar; `app/components/` the three steps and helpers
- `app/report/` report template and builder (HTML and PDF)
- `app/pages/` intentionally empty, so Streamlit keeps a single page
- `runs/` generated outputs (ignored by Git)

## Rules the UI follows
Unknown plate characters stay `?`. The verdict (`match` or `no_confident_match`) decides what is shown; scores never override it. Weak leads are labelled as leads, not matches. Every report says a human must verify the result.

## Common problems
- `streamlit` not found: activate the virtual environment first.
- PDF fails: the HTML report still downloads; check that `xhtml2pdf` installed.

### Background animation
The interface includes a lightweight radar sweep and subtle CCTV grid/scan effect. It is decorative only, stays behind controls, and respects reduced-motion preferences.
