"""
tools/mock.py

Mock tool implementations — no network, no cost. Active by default
(USE_MOCKS unset or truthy).
"""


def analyze_image(file_path: str) -> dict:
    return {"modality": "image",
            "content": "The dashboard shows a sharp drop in the sales metric after March.",
            "confidence": 0.7,
            "ref": {"type": "image_region", "path": file_path, "bbox": [120, 340, 500, 600]}}


def analyze_document(file_path: str) -> dict:
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if text:
        content = f"The document states: {text}"
    else:
        content = "The document could not be read, but it was detected as a document."
    return {"modality": "document", "content": content,
            "confidence": 0.8,
            "ref": {"type": "doc_span", "path": file_path, "page": 1}}


def transcribe_audio(file_path: str) -> dict:
    return {"modality": "audio",
            "content": "(mock transcription) The audio transcript states that the sales metric dropped after March.",
            "confidence": 0.6,
            "ref": {"type": "audio_segment", "path": file_path, "start": 0, "end": 8}}


def generate_answer(state: dict) -> str:
    lines = ["Based on the available multimodal evidence:"]
    for e in state["evidence"]:
        ref = e.get("ref", {})
        loc = ref.get("path", "")
        lines.append(
            f"- [{e['modality']}] {e['content']}  (ref: {ref.get('type', '?')} {loc})")
    return "\n".join(lines)
