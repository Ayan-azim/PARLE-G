# TraceVision

CCTV vehicle and number plate recognition app for the hackathon project.

## Python

Use **Python 3.11.9** for both the backend and model pipeline. The required version is recorded in `.python-version`.

## Project layout

- `backend/` — Streamlit app, shared schema, vision adapter, and registry matching.
- `models/` — current YOLO weights, inference pipeline, and model-specific outputs.
- `backend/legacy/` — preserved earlier implementation and local assets.

For setup and app usage, see [backend/README.md](backend/README.md).
