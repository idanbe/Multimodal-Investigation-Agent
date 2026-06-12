"""
agent.py

Multimodal AI agent: RUN -> (PLAN -> ACT -> OBSERVE)* -> DONE.

"""

from langgraph.graph import StateGraph, START, END
from state import AgentState
from tools import append_trace
from events import Event
from actions import ACTIONS
import planner


def run(state: AgentState) -> dict:
    return {
        "current_state": "RUN",
        "event": Event.USER_INPUT_RECEIVED,
        "trace": append_trace(state, {"type": "run", "message": "starting agent run"}),
    }


def plan(state: AgentState) -> dict:
    result = planner.plan_next_action(state)
    next_action = result["next_action"]
    return {
        "next_action": next_action,
        "current_state": "PLAN",
        "trace": append_trace(state, {"type": "plan", "next_action": next_action}),
    }


def act(state: AgentState) -> dict:
    action = state["next_action"]
    action_function = ACTIONS[action]
    action_result = action_function(state)
    action_trace = action_result.get("trace_entry", {})
    return {
        "current_state": "ACT",
        "action_result": action_result,
        "actions_taken": [*state["actions_taken"], action],
        "trace": append_trace(state, {"type": "act", "action": action, **action_trace}),
    }


def observe(state: AgentState) -> dict:
    action_result = state["action_result"]
    fields_to_ignore = {"action", "trace_entry"}
    fields_to_update = {key: value for key,
                        value in action_result.items() if key not in fields_to_ignore}
    return {
        **fields_to_update,
        "current_state": "OBSERVE",
        "trace": append_trace(state, {
            "type": "observe",
            "action": action_result["action"],
            "event": action_result.get("event"),
        }),
    }


def done(state: AgentState) -> dict:
    return {
        "current_state": "DONE",
        "trace": append_trace(state, {"type": "done", "message": "Done"}),
    }


def build_agent() -> StateGraph[AgentState, str]:
    """Build the ReAct agent graph: RUN -> (PLAN -> ACT -> OBSERVE)* -> DONE."""
    graph = StateGraph(AgentState)

    graph.add_node("RUN", run)
    graph.add_node("PLAN", plan)
    graph.add_node("ACT", act)
    graph.add_node("OBSERVE", observe)
    graph.add_node("DONE", done)

    graph.add_edge(START, "RUN")
    graph.add_edge("RUN", "PLAN")
    graph.add_conditional_edges("PLAN", planner.route_after_plan,
                                {"ACT": "ACT", "DONE": "DONE"})
    graph.add_edge("ACT", "OBSERVE")
    graph.add_edge("OBSERVE", "PLAN")
    graph.add_edge("DONE", END)

    return graph.compile()
