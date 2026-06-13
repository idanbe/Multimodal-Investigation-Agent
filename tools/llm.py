"""
tools/llm.py

Real model calls through OpenRouter (OpenAI-compatible API).
Used by the dispatch functions in tools/__init__.py when USE_MOCKS is falsy.

"""

import base64
import json

from openai import OpenAI

import config
import prompts

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_EXTENSION_TO_MEDIA_TYPE = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


def _client() -> OpenAI:
    api_key = config.openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in .env (see .env.example) "
            "or set USE_MOCKS=true to use mock tools."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _chat(model: str, messages: list[dict]) -> str:
    response = _client().chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def _chat_json(model: str, messages: list[dict]) -> dict:
    """Chat call that expects a JSON object reply:
    {"content": str, "confidence": float, "ref": dict}.

    A parse failure raises, which run_tool_for_state turns into a tool_failed
    retry — same path as a network error.
    """
    response = _client().chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    # Some providers ignore response_format and wrap the JSON in fences anyway.
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return json.loads(raw)


def _parse_confidence(result: dict, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(result["confidence"])))
    except (KeyError, TypeError, ValueError):
        return default


def _merge_ref(result: dict, **base) -> dict:
    """Combine LLM-supplied ref details with code-owned keys.

    "type" and "path" always come from code — the validator's grounding
    check depends on them, so the model must not be able to override them.
    """
    llm_ref = result.get("ref")
    if not isinstance(llm_ref, dict):
        llm_ref = {}
    return {**llm_ref, **base}


def _encode_file(file_path: str) -> str:
    with open(file_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def build_image_messages(question: str, image_b64: str, media_type: str) -> list[dict]:
    return [
        {"role": "system", "content": prompts.IMAGE_ANALYSIS_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"User question: {question}"},
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
        ]},
    ]


def build_document_messages(question: str, document_text: str) -> list[dict]:
    return [
        {"role": "system", "content": prompts.DOCUMENT_ANALYSIS_PROMPT},
        {"role": "user",
         "content": f"User question: {question}\n\nDocument:\n{document_text}"},
    ]


def build_audio_messages(question: str, audio_b64: str, audio_format: str) -> list[dict]:
    # OpenRouter input_audio supports wav/mp3 on audio-capable models.
    return [
        {"role": "system", "content": prompts.AUDIO_ANALYSIS_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"User question: {question}"},
            {"type": "input_audio",
             "input_audio": {"data": audio_b64, "format": audio_format}},
        ]},
    ]


def build_answer_messages(question: str, evidence: list[dict]) -> list[dict]:
    evidence_lines = "\n".join(
        f"- [{e['modality']}] {e['content']} "
        f"(source: {e.get('ref', {}).get('path', 'unknown')})"
        for e in evidence
    )
    return [
        {"role": "system", "content": prompts.FINAL_ANSWER_PROMPT},
        {"role": "user",
         "content": f"User question: {question}\n\nEvidence:\n{evidence_lines}"},
    ]


def analyze_image(file_path: str, question: str = "") -> dict:
    extension = file_path.split(".")[-1].lower()
    media_type = _EXTENSION_TO_MEDIA_TYPE.get(extension, "image/png")
    message = build_image_messages(
        question, _encode_file(file_path), media_type)
    result = _chat_json(config.openrouter_model("image"), message)
    return {"modality": "image",
            "content": result.get("content", ""),
            "confidence": result.get("confidence", 0.0),
            "ref": _merge_ref(result, type="image_region", path=file_path)}


def analyze_document(file_path: str, question: str = "") -> dict:
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read().strip()

    message = build_document_messages(question, text)
    result = _chat_json(config.openrouter_model("document"), message)
    return {"modality": "document",
            "content": result.get("content", ""),
            "confidence": result.get("confidence", 0.0),
            "ref": {"page": 1, **_merge_ref(result, type="doc_span", path=file_path)}}


def transcribe_audio(file_path: str, question: str = "") -> dict:
    audio_format = file_path.split(".")[-1].lower()

    message = build_audio_messages(
        question, _encode_file(file_path), audio_format)
    result = _chat_json(config.openrouter_model("audio"), message)
    return {"modality": "audio",
            "content": result.get("content", ""),
            "confidence": result.get("confidence", 0.0),
            "ref": _merge_ref(result, type="audio_segment", path=file_path)}


def generate_answer(state: dict) -> str:
    message = build_answer_messages(state["user_question"], state["evidence"])
    return _chat(config.openrouter_model("answer"), message)
