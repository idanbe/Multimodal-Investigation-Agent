from typing import Any, Dict

from state import AgentState


def validate_evidence(state: AgentState, minimum_modalities: int = 2) -> Dict[str, Any]:
    """
    Validate the evidence: grounded === (used_modalities >= minimum_modalities AND
    every evidence item has a reference).
    """
    evidence = state["evidence"]
    used_modalities = sorted({e["modality"] for e in evidence})
    has_refs = bool(evidence) and all("ref" in e for e in evidence)
    grounded = len(used_modalities) >= minimum_modalities and has_refs
    if grounded and evidence:
        avg = sum(float(e.get("confidence", 0.0))
                  for e in evidence) / len(evidence)
        confidence = round(avg, 2)
    else:
        confidence = 0.4
    return {
        "used_modalities": used_modalities,
        "grounded": grounded,
        "confidence": confidence,
    }
