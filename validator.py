from typing import Any, Dict, List

from state import AgentState
from tools import append_trace


def validate_evidence(state: AgentState, minimum_modalities: int = 2) -> Dict[str, Any]:
    """
    Validate the evidence according to the minimum modalities.
    """

    used_modalities = sorted({e["modality"] for e in state["evidence"]})
    grounded = len(used_modalities) >= minimum_modalities
    confidence = 0.8 if grounded else 0.4  # Mock confidence
    return {
        "current_state": "VALIDATE",
        "used_modalities": used_modalities,
        "grounded": grounded,
        "confidence": confidence,
        "trace": append_trace(state, {
            "type": "validation_result",
            "message": f"Validating: used_modalities: {used_modalities}, grounded: {grounded}, confidence: {confidence}",
        }),
    }
