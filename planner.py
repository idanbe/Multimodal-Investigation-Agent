from tools import pick_next_tool
from state import AgentState
from events import Event
from actions import Action


CLARIFY_EVENTS = {
    Event.NO_FILES,
    Event.NO_SUPPORTED_FILES,
    Event.INSUFFICIENT_MODALITIES,
    Event.NO_TOOLS_SELECTED,
    Event.EVIDENCE_INSUFFICIENT,
    Event.ANSWER_UNGROUNDED,
    Event.RETRIES_EXHAUSTED,
}


def plan_next_action(state: AgentState) -> dict:
    """Plan the next action based on the last event and state."""
    event = state.get("event")

    if event in (Event.ANSWER_READY, Event.CLARIFICATION_EMITTED):
        return {"next_action": Action.DONE}

    if state["retries"] > state["max_retries"]:
        return {"next_action": Action.CLARIFY}

    if event == Event.USER_INPUT_RECEIVED:
        return {"next_action": Action.INGRESS}

    if event == Event.FILES_LOADED:
        return {"next_action": Action.DETECT_MODALITIES}

    if event == Event.MODALITIES_DETECTED:
        return {"next_action": Action.SELECT_TOOLS}

    if event in (Event.TOOLS_SELECTED, Event.TOOL_RESULTS_RECEIVED, Event.RETRY_AVAILABLE):
        if pick_next_tool(state):
            return {"next_action": Action.EXTRACT_EVIDENCE}
        return {"next_action": Action.VALIDATE}

    if event == Event.TOOL_FAILED:
        return {"next_action": Action.ERROR_RECOVERY}

    if event == Event.EVIDENCE_READY:
        return {"next_action": Action.RESPOND}

    if event in CLARIFY_EVENTS:
        return {"next_action": Action.CLARIFY}

    return {"next_action": Action.CLARIFY}  # unknown event — safest exit


def route_after_plan(state: AgentState) -> str:
    return "DONE" if state["next_action"] == Action.DONE else "ACT"
