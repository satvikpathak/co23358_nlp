"""Model loading and inference utilities.

This module loads a seq2seq model and exposes a `rewrite_text` function.
"""
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger(__name__)

# Default model name; small and fast for local testing. You can switch to a larger Flan-T5 if you have RAM/GPU.
DEFAULT_MODEL = "google/flan-t5-small"


class ToneModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def rewrite(self, prompt: str, max_length: int = 256, num_beams: int = 4) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0]


# Global model instance (will be set by the FastAPI startup event)
_GLOBAL_MODEL: Optional[ToneModel] = None


def init_global_model(model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = ToneModel(model_name=model_name, device=device)
    return _GLOBAL_MODEL


def get_global_model() -> ToneModel:
    if _GLOBAL_MODEL is None:
        raise RuntimeError("Model is not initialized. Call init_global_model() first.")
    return _GLOBAL_MODEL
