# Tone Transformer — Politeness and Professional Tone Rewriter

## Abstract

Tone Transformer is a small full-stack application that rewrites user-provided text into a polite, formal, or friendly tone using pre-trained sequence-to-sequence LLMs (e.g. Flan-T5 / T5-small). It helps people convert blunt or informal messages into professionally phrased communication suitable for work, email, or public-facing content.

## Problem Statement

People often write messages that sound abrupt, rude, or informal. Miscommunication can cause friction in professional settings. An automated tone transformer improves clarity and reduces social friction by rephrasing text into appropriate tones.

## Methodology

- Preprocessing: minimal — trim input, basic validation, optional prompt prefix depending on tone.
- Model: a seq2seq LLM (default: `google/flan-t5-small` or `t5-small`) is loaded using Hugging Face Transformers. Tokenizer and model are cached in memory.
- Backend: FastAPI exposes a single POST endpoint `/rewrite` that accepts JSON `{ "text": "...", "tone": "polite" }` and returns `{ "rewritten": "..." }`.
- Frontend: static HTML + JS sends user input to the backend, shows a loader, and displays the result.

## Evaluation Plan

- Automatic metrics: BLEU / ROUGE when paired with human-crafted references (for offline evaluation).
- Human evaluation: A/B tests asking raters to judge politeness, formality, and meaning preservation.
- Success criteria: outputs preserve the original meaning, increase politeness/formality scores, and rate highly in human evaluation.

## Expected Output Examples

- Input: "Send me the file now."
  Output: "Could you please send me the file at your earliest convenience?"

- Input: "Why didn't you finish the report?"
  Output: "I noticed the report is incomplete; could you please let me know the status and when it might be finished?"

## Project structure

```
nlp/
├─ backend/
│  ├─ main.py
│  ├─ model.py
│  ├─ utils.py
│  └─ history.json
├─ frontend/
│  ├─ index.html
│  ├─ style.css
│  └─ app.js
├─ requirements.txt
├─ .gitignore
└─ sample_run.sh
```

## Environment setup

1. Create a virtualenv and activate it (Linux zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the backend (development):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open `frontend/index.html` in your browser (or serve the frontend from the same backend — it's simple enough to open locally).

If you prefer the backend to serve the frontend, you can add static file mounting in `backend/main.py` by adding:

```py
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
```

Then access the app at `http://127.0.0.1:8000/`.

## API

POST /rewrite

Request JSON:

```json
{
  "text": "I need this done.",
  "tone": "polite"
}
```

Response JSON:

```json
{
  "rewritten": "Could you please complete this when you have a moment?",
  "tone": "polite",
  "original": "I need this done."
}
```

## Sample curl

See `sample_run.sh` for an example.

## Deployment tips

- For small scale, deploy the FastAPI app to Render or a small VPS. If using GPU instances, install CUDA-enabled PyTorch.
- Consider converting the model to an optimized runtime (ONNX, quantization) for production.

## Notes

- This is a simple, easy-to-run scaffold. For production-grade usage add authentication, rate limiting, batching, monitoring, and robust safety checks.
