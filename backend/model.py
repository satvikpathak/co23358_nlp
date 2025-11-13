"""Model loading and inference utilities.

This module loads a seq2seq model and exposes a `rewrite_text` function.
Supports both pretrained models and custom fine-tuned models.
"""
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
# Use fine-tuned model if available, otherwise use pretrained model
FINE_TUNED_MODEL_PATH = "./models/informal-to-formal-t5"
DEFAULT_MODEL = "google/flan-t5-small"  # Fallback pretrained model

# Check if fine-tuned model exists
if Path(FINE_TUNED_MODEL_PATH).exists():
    logger.info(f"Fine-tuned model found at {FINE_TUNED_MODEL_PATH}")
    DEFAULT_MODEL = FINE_TUNED_MODEL_PATH
else:
    logger.info(f"Using pretrained model: {DEFAULT_MODEL}")


class ToneModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on device {self.device}")
        
        # Check if this is a fine-tuned model (local path)
        self.is_fine_tuned = Path(model_name).exists()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully! Type: {'Fine-tuned' if self.is_fine_tuned else 'Pretrained'}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def rewrite(self, prompt: str, max_length: int = 256, num_beams: int = 4) -> str:
        # If using fine-tuned model, use the training prompt format
        if self.is_fine_tuned:
            if not prompt.startswith("Convert informal to formal:"):
                prompt = f"Convert informal to formal: {prompt}"
        
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
