<!-- Copilot instructions for contributors and AI coding agents -->
# Project snapshot

This repo is a small PDF text-extraction tool with a Streamlit frontend:
- `pdfExtract.py`: core extraction, normalization, and translation helpers.
- `app_streamlit.py`: Streamlit-based UI that imports the `pdfExtract` functions.
- `extract_and_save.py`: disabled CLI helper (Streamlit is the primary UI).

# Big-picture architecture (what to know first)

- Extraction pipeline (priority): PyMuPDF (`pymupdf`). Code lives in `extract_pages_from_pdf_bytes`.
- Post-processing: `reflow_from_word_tuples` and `reflow_and_segment` normalize layout and do sentence segmentation (uses `spacy`/`kss` when available).
- The UI and CLI share the same core functions in `pdfExtract.py` — prefer changes there for behavior updates.

# Key files to reference when making changes

- `pdfExtract.py` — core logic, fallbacks, download store (`DOWNLOAD_STORE`), translation helpers (`get_translator`, `safe_translate`).
- `app_streamlit.py` — alternate frontend demonstrating how to call `extract_pages_from_pdf_bytes`, `reflow_and_segment`, and `fix_urls_and_dois`.
- `extract_and_save.py` — example CLI usage and the format of `.extracted.txt` and `.extracted.docx` outputs.
- `templates/index.html` — small jinja template used by the Flask app; update this for UI changes.
- `requirements.txt` — current dependency list and optional/conditional libraries such as `pymupdf`, `googletrans`, `python-docx`.

# Important patterns & conventions (code-level)

- Text assembly: produced text uses page headers like `=== 페이지 N ===` and image markers `\n[이미지 또는 표가 포함되어 있습니다]\n`. Keep these markers when editing output code.
- Fail-open and graceful degradation: the code prefers optional libraries but falls back silently (e.g., `HAVE_PYMUPDF`, `HAVE_PDFPLUMBER`, `HAVE_PYPDF2`). Follow this pattern: try optional feature, degrade to simpler behavior if missing.
- No persistent store: `DOWNLOAD_STORE` is an in-memory dict (ephemeral). For production, replace with Redis/S3 and preserve the existing API semantics.
- Translation: `googletrans` (recommended `googletrans==4.0.0-rc1`) is optional and guarded by `HAVE_TRANSFORMERS`.

# How to run & debug (Windows PowerShell examples)

- Create environment and install deps:
```
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```
- Run Flask dev server (default):
```
python pdfExtract.py
# open http://127.0.0.1:5000
```
- Run with ASGI (uvicorn) on port 8000:
```
$env:USE_UVICORN = '1'
python pdfExtract.py
# or: uvicorn pdfExtract:asgi_app --host 0.0.0.0 --port 8000
```
- Streamlit UI (alternative):
```
streamlit run app_streamlit.py
```

# API and quick examples

- API endpoint: `POST /api/extract` (multipart/form-data, field `file`). Returns JSON: `{"success": true, "text": "..."}`.
- Curl example (local Flask dev server):
```
curl -F "file=@/path/to/file.pdf" http://127.0.0.1:5000/api/extract
```

# Dependencies and optional features

-- For best extraction quality install `pymupdf` (`pymupdf` is used first).
-- `python-docx` enables optional `.docx` output.
-- `googletrans==4.0.0-rc1` enables server-side translation flows (optional).

# Testing / validation hints

- To validate extraction changes, use `extract_and_save.py` on a representative PDF and inspect the generated `.extracted.txt`.
- Unit tests are not provided — prefer small manual checks when changing parsing heuristics. Use the CLI and web UI to verify outputs.

# Migration / production notes for contributors

- Replace `DOWNLOAD_STORE` with a persistent store for concurrent/long-running servers.
- Keep environment flags: `USE_UVICORN` (to switch to ASGI) and `PORT` are respected by `pdfExtract.py`.

# If you're an AI coding agent: where to make safe edits

- Fix text-processing bugs: modify `reflow_from_word_tuples`, `normalize_extracted_text`, or `reflow_and_segment` in `pdfExtract.py`.
- UI tweaks: update `templates/index.html` or `app_streamlit.py` (Streamlit UI is simpler for fast iterations).
- Add new extraction backends: add detection logic and preserve the existing fallback order and `has_images` semantics.

# Feedback

If any section is unclear or you'd like examples expanded (e.g., more curl examples, detailed tests), say which area and I'll extend this doc.
