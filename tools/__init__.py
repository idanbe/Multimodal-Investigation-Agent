"""
tools package.

__init__.py — shared infra (modality detection, tool selection, trace,
output formatting) + mock-vs-real dispatch on config.use_mocks().
mock.py     — mock tool implementations (default).
llm.py      — real model calls through OpenRouter.
"""

from datetime import datetime, timezone

import config
from state import AgentState

from . import llm, mock


def _trace_meta() -> dict:
    return {"timestamp": datetime.now(timezone.utc).isoformat()}


def append_trace(state: AgentState, entry: dict) -> list[dict]:
    return [*state["trace"], {**_trace_meta(), **entry}]


MODALITY_TO_TOOL_MAP: dict[str, str] = {
    "image": "analyze_image",
    "document": "analyze_document",
    "audio": "transcribe_audio"
}

TOOL_TO_MODALITY_MAP: dict[str, str] = {
    tool: modality for modality, tool in MODALITY_TO_TOOL_MAP.items()
}

MODALITY_TO_EXTENSION_MAP: dict[str, list[str]] = {
    "image": ["png", "jpg", "jpeg"],
    "document": ["txt", "pdf"],
    "audio": ["mp3", "wav", "m4a"]
}


def select_tools(state: AgentState) -> list[str]:
    return [MODALITY_TO_TOOL_MAP[m] for m in state["detected_modalities"]]


def pick_next_tool(state: AgentState) -> str:
    """First selected tool whose modality has no evidence yet, else ''."""
    already_run = {e["modality"] for e in state["evidence"]}
    for tool in state["selected_tools"]:
        if TOOL_TO_MODALITY_MAP[tool] not in already_run:
            return tool
    return ""


def analyze_image(file_path: str, question: str = "") -> dict:
    """Mock by default; real OpenRouter call when USE_MOCKS is falsy."""
    if config.use_mocks():
        return mock.analyze_image(file_path)
    return llm.analyze_image(file_path, question)


def analyze_document(file_path: str, question: str = "") -> dict:
    """Mock by default; real OpenRouter call when USE_MOCKS is falsy."""
    if config.use_mocks():
        return mock.analyze_document(file_path)
    return llm.analyze_document(file_path, question)


def transcribe_audio(file_path: str, question: str = "") -> dict:
    """Mock by default; real OpenRouter call when USE_MOCKS is falsy."""
    if config.use_mocks():
        return mock.transcribe_audio(file_path)
    return llm.transcribe_audio(file_path, question)


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


def run_tool_for_state(tool_name: str, files: list[str], question: str = "") -> dict:
    """Run tool_name on the first file matching that tool's modality."""
    modality = TOOL_TO_MODALITY_MAP.get(tool_name)
    if modality is None:
        return {"error": f"unknown tool: {tool_name}"}
    for f in files:
        if f.split(".")[-1] in MODALITY_TO_EXTENSION_MAP[modality]:
            try:
                return TOOLS[tool_name](f, question)
            except Exception as e:
                return {"error": str(e)}
    return {"error": f"no file for modality {modality}"}


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
    """Real LLM answer when USE_MOCKS is falsy; falls back to the mock
    evidence summary on failure (respond() is not wrapped by the retry path)."""
    if not config.use_mocks():
        try:
            return llm.generate_answer(state)
        except Exception as e:
            print(f"LLM answer generation failed ({e}); falling back to evidence summary.")
    return mock.generate_answer(state)


def format_final_output(final_state: dict) -> str:
    """Format the final agent state into a readable multi-line string."""
    lines = []

    lines.append("Final Answer:")
    lines.append("")
    lines.append(
        "Based on the available multimodal evidence, here is the answer:")

    lines.append("User question:")
    lines.append(final_state.get("user_question", "(no question)"))
    lines.append("")

    lines.append("Evidence used:")
    for e in final_state.get("evidence", []):
        lines.append(f"- [{e['modality']}] {e['content']}")
    lines.append("")

    lines.append("Conclusion:")
    if final_state.get("grounded"):
        modalities = final_state.get("used_modalities") or []
        lines.append(
            f"The agent found evidence from multiple modalities "
            f"({', '.join(modalities)}) and generated a grounded answer."
        )
    else:
        lines.append(
            "The agent could not fully ground the answer in the available evidence.")
    lines.append("")

    lines.append("Answer:")
    lines.append(final_state.get("answer", "(no answer)"))
    lines.append("")

    lines.append("Agent Trace:")
    for i, entry in enumerate(final_state.get("trace", []), start=1):
        lines.append(f"Step {i}:")
        lines.append(str(entry))
    lines.append("")

    return "\n".join(lines)


def save_output_to_file(final_state: dict, path: str = "outputs/run_output.json") -> None:
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(final_state), f, ensure_ascii=False,
                  indent=2, default=str)
    print(f"\nWrote {path}")


TOOLS = {
    "analyze_image": analyze_image,
    "analyze_document": analyze_document,
    "transcribe_audio": transcribe_audio,
    "detect_modalities": detect_modalities,
    "select_tools_for_modalities": select_tools_for_modalities,
    "generate_answer": generate_answer,
    "format_final_output": format_final_output,
    "append_trace": append_trace,
    "save_output_to_file": save_output_to_file,
}
