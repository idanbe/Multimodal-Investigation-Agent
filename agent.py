"""
agent.py

Main MultimodalAgent implementation.

This file contains the agent loop:
run -> plan -> act -> observe -> update state
"""

from state import AgentState
from tools import append_trace
from langgraph.graph import StateGraph, START, END
from validator import validate_evidence
import tools
import planner


def ingress(state: AgentState) -> dict:
    """
    Ingress state.
    """
    return {
        "current_state": "INGRESS",
        "trace": append_trace(state, {"type": "ingress", "message": "Agent is starting"}),
    }


def detect_modalities(state: AgentState) -> dict:
    """
    Detect modalities.
    """
    detected = tools.detect_modalities(state["files"])
    modalities = detected["modalities"]
    return {
        "detected_modalities": modalities,
        "current_state": "DETECT_MODALITIES",
        "trace": append_trace(state, {"type": "detect_modalities", "message": f"Detected modalities: {modalities}"}),
    }


def select_tools(state: AgentState) -> dict:
    """
    Select tools.
    """
    selected_tools = planner.select_tools(state)
    return {
        "selected_tools": selected_tools,
        "current_state": "SELECT_TOOLS",
        "trace": append_trace(state, {"type": "select_tools", "message": f"Selecting tools: {selected_tools}"}),
    }


def plan_next_action(state: AgentState) -> dict:
    """
    Plan next action.
    """
    next_tool = planner.pick_next_tool(state)
    return {
        "next_tool": next_tool,
        "current_state": "PLAN_NEXT_ACTION",
        "trace": append_trace(state, {"type": "plan_next_action", "message": f"Planning next action: {next_tool}"}),
    }


def act(state: AgentState) -> dict:
    """Execute next_tool and store the result in last_observation."""
    tool_name = state["next_tool"]
    result = tools.run_tool_for_state(tool_name, state["files"])
    return {
        "last_observation": result,
        "current_state": "ACT",
        "trace": append_trace(state, {"type": "act", "tool": tool_name}),
    }


def observe(state: AgentState) -> dict:
    obs = state["last_observation"]
    if obs.get("error"):
        return {"current_state": "OBSERVE",
                "trace": append_trace(state, {"type": "observe", "status": "error", "detail": obs["error"]})}
    return {"evidence": state["evidence"] + [obs], "current_state": "OBSERVE",
            "trace": append_trace(state, {"type": "observe", "status": "ok", "modality": obs["modality"]})}


def error_recovery(state: AgentState) -> dict:
    """
    Recover from an error.
    """
    retries = state["retries"] + 1
    return {"retries": retries, "current_state": "ERROR_RECOVERY",
            "trace": append_trace(state, {"type": "error_recovery", "retries": retries})}


def extract_evidence(state: AgentState) -> dict:
    """
    Extract evidence.
    """
    evidence = [*state["evidence"]]

    return {
        "evidence": evidence,
        "current_state": "EVIDENCE_EXTRACTED",
        "trace": append_trace(state, {"type": "extract_evidence", "message": f"Extracted evidence: {evidence}"}),
    }


def respond(state: AgentState) -> dict:
    """
    Generate answer.
    """
    answer = tools.generate_answer(state)
    return {
        "answer": answer,
        "current_state": "RESPOND",
        "trace": append_trace(state, {"type": "final_answer", "answer": answer}),
    }


def clarify(state: AgentState) -> dict:
    msg = ("I could not produce a grounded answer from at least two modalities. "
           "Please provide clearer files (e.g. a readable document and an image).")
    return {"answer": msg, "current_state": "CLARIFY",
            "trace": append_trace(state, {"type": "clarify", "message": msg})}


def done(state: AgentState) -> dict:
    """
    Done.
    """
    return {
        "current_state": "DONE",
        "trace": append_trace(state, {"type": "done", "message": "Done"}),
    }


def build_agent() -> StateGraph[AgentState, str]:
    """
    Build the agent graph.
    """
    graph = StateGraph(AgentState)
    # Add nodes to the graph
    graph.add_node("INGRESS", ingress)
    graph.add_node("DETECT_MODALITIES", detect_modalities)
    graph.add_node("SELECT_TOOLS", select_tools)
    graph.add_node("PLAN_NEXT_ACTION", plan_next_action)
    graph.add_node("ACT", act)
    graph.add_node("OBSERVE", observe)
    graph.add_node("ERROR_RECOVERY", error_recovery)
    graph.add_node("EXTRACT_EVIDENCE", extract_evidence)
    graph.add_node("VALIDATE", validate_evidence)
    graph.add_node("RESPOND", respond)
    graph.add_node("CLARIFY", clarify)
    graph.add_node("DONE", done)

    # Add edges to the graph
    graph.add_edge(START, "INGRESS")
    graph.add_edge("INGRESS", "DETECT_MODALITIES")
    graph.add_edge("ACT", "OBSERVE")
    graph.add_edge("EXTRACT_EVIDENCE", "VALIDATE")
    graph.add_edge("CLARIFY", "DONE")
    graph.add_edge("DONE", END)  # type: ignore

    # Add conditional edges to the graph
    graph.add_conditional_edges(
        "DETECT_MODALITIES", planner.route_after_detect,
        {"SELECT_TOOLS": "SELECT_TOOLS", "CLARIFY": "CLARIFY"})
    graph.add_conditional_edges(
        "SELECT_TOOLS", planner.route_after_select,
        {"PLAN_NEXT_ACTION": "PLAN_NEXT_ACTION", "CLARIFY": "CLARIFY"})
    graph.add_conditional_edges(
        "PLAN_NEXT_ACTION", planner.route_after_plan,
        {"ACT": "ACT", "EXTRACT_EVIDENCE": "EXTRACT_EVIDENCE"})
    graph.add_conditional_edges(
        "OBSERVE", planner.route_after_observe,
        {"ERROR_RECOVERY": "ERROR_RECOVERY", "PLAN_NEXT_ACTION": "PLAN_NEXT_ACTION"})
    graph.add_conditional_edges(
        "ERROR_RECOVERY", planner.route_after_error,
        {"PLAN_NEXT_ACTION": "PLAN_NEXT_ACTION", "CLARIFY": "CLARIFY"})
    graph.add_conditional_edges(
        "VALIDATE", planner.route_after_validate,
        {"RESPOND": "RESPOND", "CLARIFY": "CLARIFY"})
    graph.add_conditional_edges(
        "RESPOND", planner.route_after_respond,
        {"DONE": "DONE", "CLARIFY": "CLARIFY"})

    # Return the graph
    return graph.compile()
