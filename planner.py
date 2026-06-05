from tools import MODALITY_TO_TOOL_MAP, TOOL_TO_MODALITY_MAP
from state import AgentState


def select_tools(state: AgentState) -> list[str]:
    """
    Select the tools for the available modalities.
    """
    return [MODALITY_TO_TOOL_MAP[modality] for modality in state["detected_modalities"]]


def pick_next_tool(state: AgentState) -> str:
    """
    Next selected tool whose modality has no evidence yet, or '' when no tool is available
    """
    already_run = {e["modality"] for e in state["evidence"]}
    for tool in state["selected_tools"]:
        if TOOL_TO_MODALITY_MAP[tool] not in already_run:
            return tool
    return ""


# ~~~~~~~routing-functions-for-conditional-edges~~~~~~~

def route_after_detect(state: AgentState) -> str:
    return "SELECT_TOOLS" if state["detected_modalities"] else "CLARIFY"


def route_after_select(state: AgentState) -> str:
    return "PLAN_NEXT_ACTION" if len(state["selected_tools"]) >= 2 else "CLARIFY"


def route_after_plan(state: AgentState) -> str:
    return "ACT" if state["next_tool"] else "EXTRACT_EVIDENCE"


def route_after_observe(state: AgentState) -> str:
    return "ERROR_RECOVERY" if state["last_observation"].get("error") else "PLAN_NEXT_ACTION"


def route_after_error(state: AgentState) -> str:
    return "PLAN_NEXT_ACTION" if state["retries"] < state["max_retries"] else "CLARIFY"


def route_after_validate(state: AgentState) -> str:
    return "RESPOND" if state["grounded"] else "CLARIFY"


def route_after_respond(state: AgentState) -> str:
    return "DONE" if state["answer"] else "CLARIFY"
