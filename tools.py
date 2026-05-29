""" 
tools.py

Tools that the agent can use.
"""

print("tools.py imported")

MODALITY_TO_TOOL_MAP: dict[str, str] = {
    "image": "analyze_image",
    "document": "analyze_document",
    "audio": "transcribe_audio"
}

MODALITY_TO_EXTENSION_MAP: dict[str, list[str]] = {
    "image": ["png", "jpg", "jpeg"],
    "document": ["txt", "pdf"],
    "audio": ["mp3", "wav", "m4a"]
}


def analyze_image(file_path: str) -> dict:
    """
    Analyze an image file. Mock tool for now.
    """
    return {"modality": "image",
            "content": "The dashboard shows a sharp drop in the sales metric after March."}


def analyze_document(file_path: str) -> dict:
    """
    Analyze a document file. Mock tool for now.
    """
    text = open(file_path, encoding="utf-8").read().strip()
    if text:
        content = f"The document states: {text}"
    else:
        content = "The document could not be read, but it was detected as a document."
    return {"modality": "document",
            "content": content}


def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe an audio file. Mock tool for now.
    """
    return {"modality": "audio",
            "content": "(mock transcription) The audio transcript states that the sales metric dropped after March."}


def detect_modalities(files: list[str]) -> dict:
    """
    Detect the modalities available in the provided files.
    """
    modalities = []
    for file in files:
        extension = file.split(".")[-1]
        for modality, extensions in MODALITY_TO_EXTENSION_MAP.items():
            if extension in extensions:
                modalities.append(modality)
    return {"type": "modalities_detected",
            "modalities": modalities}


def select_tools_for_modalities(modalities: list[str]) -> dict:
    """
    Select the tools for the available modalities.
    """
    selected_tools = []
    for modality in modalities:
        selected_tools.append(MODALITY_TO_TOOL_MAP[modality])
    return {"type": "tools_selected",
            "tools": selected_tools}


def generate_answer(state: dict) -> str:
    """Compose the answer body from evidence (no Confidence line — respond() adds it)."""
    lines = [
        f"Based on the available multimodal evidence: {state['user_question']}"]
    for e in state["evidence"]:
        lines.append(f"- [{e['modality']}] {e['content']}")
    lines.append("")
    lines.append(f"Confidence: {state['confidence']}")
    lines.append(f"Used modalities: {state['used_modalities']}")
    lines.append(f"Grounded: {state['grounded']}")
    return "\n".join(lines)


TOOLS = {
    "analyze_image": analyze_image,
    "analyze_document": analyze_document,
    "transcribe_audio": transcribe_audio,
    "detect_modalities": detect_modalities,
    "select_tools_for_modalities": select_tools_for_modalities,
    "generate_answer": generate_answer,
}
