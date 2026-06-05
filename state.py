"""
state.py
Creates the external state object for the agent.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    The state of the agent.
    """
    goal: str
    user_question: str
    files: list[str]  # ["examples/dashboard.png", "examples/context.txt"]
    detected_modalities: list[str]  # modalities detected by the agent
    selected_tools: list[str]  # tools selected by the agent
    next_tool: str   # tool the planner chose to run next
    last_observation: dict  # the result of the last action
    evidence: list[dict]
    used_modalities: list[str]  # modalities used to extract evidence
    answer: str  # the final answer
    confidence: float  # the confidence in the answer
    grounded: bool  # whether the answer is grounded to the evidence
    retries: int  # the number of times the agent has retried to answer the question
    trace: list[dict]  # the trace of the agent's actions
    current_state: str  # the current state of the agent
    max_retries: int  # the maximum number of retries the agent can make


def create_initial_state(question: str, files: list[str]) -> AgentState:
    """
    Create the initial state for the agent.
    """
    return {
        "goal": "Answer the user's question using multimodal evidence from the files",
        "user_question": question,
        "files": files,
        "detected_modalities": [],
        "selected_tools": [],
        "next_tool": None,
        "last_observation": None,
        "evidence": [],
        "used_modalities": [],
        "answer": None,
        "confidence": 0.0,
        "grounded": False,
        "retries": 0,
        "trace": [],
        "current_state": "INGRESS",
        "max_retries": 2
    }
