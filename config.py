"""
config.py

"""

import os

from dotenv import load_dotenv

load_dotenv()


DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"

_MODEL_ENV_VARS = {
    "image": "OPENROUTER_IMAGE_MODEL",
    "document": "OPENROUTER_DOCUMENT_MODEL",
    "audio": "OPENROUTER_AUDIO_MODEL",
    "answer": "OPENROUTER_ANSWER_MODEL",
}


def use_mocks() -> bool:
    mock_mode = os.getenv("USE_MOCKS", "true").strip().lower()
    return mock_mode == "true"


def openrouter_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def openrouter_model(task: str) -> str:
    """OpenRouter model id for task: 'image' | 'document' | 'audio' | 'answer'."""
    return os.getenv(_MODEL_ENV_VARS[task], "").strip() or DEFAULT_OPENROUTER_MODEL
