"""FastAPI app exposing /rewrite endpoint."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from .model import init_global_model, get_global_model
from .utils import save_history, validate_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tone Transformer")

# Allow local dev frontend to call the API. In production, restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RewriteRequest(BaseModel):
    text: str
    tone: Optional[str] = "polite"


class RewriteResponse(BaseModel):
    original: str
    rewritten: str
    tone: str


@app.on_event("startup")
def startup_event():
    # Initialize the model once at startup; this keeps tokenizer and weights in memory.
    try:
        init_global_model()
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    try:
        text = validate_input(req.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = get_global_model()

    # Create a prompt for the seq2seq model. We keep this simple.
    prompt = f"Rewrite the following text to be {req.tone} and professional while preserving meaning:\n\n{text}"
    try:
        rewritten = model.rewrite(prompt)
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed")

    resp = {"original": text, "rewritten": rewritten, "tone": req.tone}
    # Save minimal history (non-blocking for this simple example)
    try:
        save_history(resp)
    except Exception:
        logger.warning("Failed to save history")

    return resp
