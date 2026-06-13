"""actions.py — action constants and implementations for the multimodal agent.

Each Action constant is a name used as a key in ACTIONS, stored in
state["next_action"] / state["actions_taken"], and logged in the trace.
"""

from state import AgentState
from events import Event
from validator import validate_evidence
import tools


class Action:
    """The actions the agent can take."""
    INGRESS = "INGRESS"
    DETECT_MODALITIES = "DETECT_MODALITIES"
    SELECT_TOOLS = "SELECT_TOOLS"
    EXTRACT_EVIDENCE = "EXTRACT_EVIDENCE"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    VALIDATE = "VALIDATE"
    RESPOND = "RESPOND"
    CLARIFY = "CLARIFY"
    DONE = "DONE"


def ingress(state: AgentState) -> dict:
    event = Event.FILES_LOADED if state.get("files") else Event.NO_FILES
    return {
        "action": Action.INGRESS,
        "event": event,
        "trace_entry": {"message": "Agent received user input", "number_of_files": len(state.get("files", []))},
    }


def detect_modalities(state: AgentState) -> dict:
    detected = tools.detect_modalities(state["files"])["modalities"]
    event = Event.MODALITIES_DETECTED if detected else Event.NO_SUPPORTED_FILES
    return {
        "action": Action.DETECT_MODALITIES,
        "event": event,
        "detected_modalities": detected,
        "trace_entry": {"modalities": detected},
    }


def select_tools(state: AgentState) -> dict:
    detected = state["detected_modalities"]
    if len(detected) < state["minimum_modalities"]:
        return {
            "action": Action.SELECT_TOOLS,
            "event": Event.INSUFFICIENT_MODALITIES,
            "selected_tools": [],
            "trace_entry": {"reason": "fewer modalities than minimum"},
        }
    selected = tools.select_tools(state)
    event = Event.TOOLS_SELECTED if selected else Event.NO_TOOLS_SELECTED
    return {
        "action": Action.SELECT_TOOLS,
        "event": event,
        "selected_tools": selected,
        "trace_entry": {"tools": selected},
    }


def extract_evidence(state: AgentState) -> dict:
    tool = tools.pick_next_tool(state)
    result = tools.run_tool_for_state(tool, state["files"], state["user_question"])
    if "error" in result:
        return {
            "action": Action.EXTRACT_EVIDENCE,
            "event": Event.TOOL_FAILED,
            "trace_entry": {"tool": tool, "error": result["error"]},
        }
    return {
        "action": Action.EXTRACT_EVIDENCE,
        "event": Event.TOOL_RESULTS_RECEIVED,
        "evidence": [*state["evidence"], result],
        "trace_entry": {"tool": tool, "modality": result["modality"], "content": str(result["content"]), "ref": result["ref"]},
    }


def error_recovery(state: AgentState) -> dict:
    retries = state["retries"] + 1
    event = Event.RETRY_AVAILABLE if retries <= state["max_retries"] else Event.RETRIES_EXHAUSTED
    return {
        "action": Action.ERROR_RECOVERY,
        "event": event,
        "retries": retries,
        "trace_entry": {"retries": retries, "max_retries": state["max_retries"]},
    }


def validate(state: AgentState) -> dict:
    result = validate_evidence(state, state["minimum_modalities"])
    event = Event.EVIDENCE_READY if result["grounded"] else Event.EVIDENCE_INSUFFICIENT
    return {
        "action": Action.VALIDATE,
        "event": event,
        **result,  # used_modalities, grounded, confidence
        "trace_entry": {"grounded": result["grounded"], "confidence": result["confidence"]},
    }


def respond(state: AgentState) -> dict:
    answer = tools.generate_answer(state)
    event = Event.ANSWER_READY if state.get(
        "grounded") else Event.ANSWER_UNGROUNDED
    return {
        "action": Action.RESPOND,
        "event": event,
        "answer": answer,
        "trace_entry": {"grounded": state.get("grounded", False)},
    }


CLARIFY_MESSAGES = {
    Event.NO_FILES: "No files were provided. Please provide a readable document and an image.",
    Event.NO_SUPPORTED_FILES: "No supported files were provided. Please provide a readable document and an image.",
    Event.NO_TOOLS_SELECTED: "No tools were selected for the given modalities.",
    Event.TOOL_FAILED: "The tool failed to produce a result. Try again.",
    Event.RETRIES_EXHAUSTED: "The agent has exhausted all retries. Please provide a clearer question or files.",
}


def clarify(state: AgentState) -> dict:
    event = state.get("event")
    if event == Event.INSUFFICIENT_MODALITIES:
        msg = (f"Insufficient modalities were detected. "
               f"Please provide at least {state['minimum_modalities']} modalities.")
    else:
        msg = CLARIFY_MESSAGES.get(event,
                                   "I could not produce a grounded answer from at least two modalities. "
                                   "Please provide clearer files (e.g. a readable document and an image).")
    return {
        "action": Action.CLARIFY,
        "event": Event.CLARIFICATION_EMITTED,
        "answer": msg,
        "trace_entry": {"triggering_event": event, "message": msg},
    }


ACTIONS = {
    Action.INGRESS: ingress,
    Action.DETECT_MODALITIES: detect_modalities,
    Action.SELECT_TOOLS: select_tools,
    Action.EXTRACT_EVIDENCE: extract_evidence,
    Action.ERROR_RECOVERY: error_recovery,
    Action.VALIDATE: validate,
    Action.RESPOND: respond,
    Action.CLARIFY: clarify,
}
