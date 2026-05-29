"""
agent.py

Main MultimodalAgent implementation.

This file contains the agent loop:
run -> plan -> act -> observe -> update state
"""

from state import AgentState, trace_log
from langgraph.graph import StateGraph, START, END
import tools
print("agent.py imported")


def ingress(state: AgentState) -> dict:
    """
    Ingress state.
    """
    return {
        "current_state": "INGRESS",
        "trace": trace_log(state, {"type": "ingress", "message": "Agent is starting"}),
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
        "trace": trace_log(state, {"type": "detect_modalities", "message": f"Detected modalities: {modalities}"}),
    }


def select_tools(state: AgentState) -> dict:
    """
    Select tools.
    """
    selected_tools = tools.select_tools_for_modalities(
        state["detected_modalities"])
    return {
        "selected_tools": selected_tools["tools"],
        "current_state": "SELECT_TOOLS",
        "trace": trace_log(state, {"type": "select_tools", "message": f"Selecting tools: {selected_tools['tools']}"}),
    }


def extract_evidence(state: AgentState) -> dict:
    """
    Extract evidence.
    """
    evidence = [*state["evidence"]]

    return {
        "evidence": evidence,
        "current_state": "EVIDENCE_EXTRACTED",
        "trace": trace_log(state, {"type": "extract_evidence", "message": f"Extracted evidence: {evidence}"}),
    }


def validate(state: AgentState) -> dict:
    used_modalities = sorted({e["modality"] for e in state["evidence"]})
    grounded = len(used_modalities) >= 2
    confidence = 0.8 if grounded else 0.4  # Mock confidence
    return {
        "current_state": "VALIDATE",
        "used_modalities": used_modalities,
        "grounded": grounded,
        "confidence": confidence,
        "trace": trace_log(state, {
            "type": "validation_result", "message": f"Validating: used_modalities: {used_modalities}, grounded: {grounded}, confidence: {confidence}"}),
    }


def respond(state: AgentState) -> dict:
    """
    Generate answer.
    """
    answer = tools.generate_answer(state)
    return {
        "answer": answer,
        "current_state": "RESPOND",
        "trace": trace_log(state, {"type": "final_answer", "answer": answer}),
    }


def done(state: AgentState) -> dict:
    """
    Done.
    """
    return {
        "current_state": "DONE",
        "trace": trace_log(state, {"type": "done", "message": "Done"}),
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
    graph.add_node("EVIDENCE_EXTRACTED", extract_evidence)
    graph.add_node("VALIDATE", validate)
    graph.add_node("RESPOND", respond)
    graph.add_node("DONE", done)

    # Add edges to the graph
    graph.add_edge(START, "INGRESS")  # Start the graph with the ingress node
    graph.add_edge("INGRESS", "DETECT_MODALITIES")
    graph.add_edge("DETECT_MODALITIES", "SELECT_TOOLS")
    graph.add_edge("SELECT_TOOLS", "EVIDENCE_EXTRACTED")
    graph.add_edge("EVIDENCE_EXTRACTED", "VALIDATE")
    graph.add_edge("VALIDATE", "RESPOND")
    graph.add_edge("RESPOND", "DONE")
    graph.add_edge("DONE", END)

    # Return the graph
    return graph.compile()
