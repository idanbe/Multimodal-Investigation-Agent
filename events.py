"""events.py — named event constants for the agent state machine.

Each constant is an event string used in planner.TRANSITIONS and logged in the
trace by the agent nodes.
"""


class Event:
    """
    The events that can occur in the agent.
    """
    NO_FILES = "no_files"
    FILES_LOADED = "files_loaded"
    NO_SUPPORTED_FILES = "no_supported_files"
    MODALITIES_DETECTED = "modalities_detected"
    INSUFFICIENT_MODALITIES = "insufficient_modalities"
    NO_TOOLS_SELECTED = "no_tools_selected"
    TOOLS_SELECTED = "tools_selected"
    TOOL_RESULTS_RECEIVED = "tool_results_received"
    TOOL_FAILED = "tool_failed"
    RETRY_AVAILABLE = "retry_available"
    RETRIES_EXHAUSTED = "retries_exhausted"
    EVIDENCE_READY = "evidence_ready"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    ANSWER_READY = "answer_ready"
    ANSWER_UNGROUNDED = "answer_ungrounded"
    CLARIFICATION_EMITTED = "clarification_emitted"
    USER_INPUT_RECEIVED = "user_input_received"
