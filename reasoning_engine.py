"""Transparent multimodal evidence fusion for MOSAIC-X.

This module is intentionally independent of Flask and the vision model.  It accepts
probability distributions produced by other components and returns a structured,
auditable decision.  It never treats a facial expression as proof of an internal feeling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    weight: float
    status: str
    explanation: str


def _normalize(values: Mapping[str, float], labels: list[str]) -> dict[str, float]:
    cleaned = {label: max(0.0, float(values.get(label, 0.0))) for label in labels}
    total = sum(cleaned.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: value / total for label, value in cleaned.items()}


def reason_with_evidence(
    context: Mapping[str, float],
    labels: list[str],
    emotion_info: Mapping[str, Mapping[str, str]],
    face: Mapping | None = None,
    self_report: str | None = None,
    abstain_below: float = 0.45,
) -> dict:
    """Fuse context, optional expression scores, and optional self-report.

    Design invariant: when a valid self-report is supplied it is the primary conclusion.
    The face channel is ignored when its model abstains.  Output probabilities express
    plausible interpretations; they are not claims about a person's internal state.
    """
    if not labels:
        raise ValueError("labels must not be empty")
    if self_report is not None and self_report not in labels:
        raise ValueError("self_report must be one of the configured labels")

    context_p = _normalize(context, labels)
    face_usable = bool(face and not face.get("unsure") and isinstance(face.get("scores"), Mapping))
    face_p = _normalize(face.get("scores", {}), labels) if face_usable else None

    if self_report:
        weights = {"context": 0.25, "face": 0.10 if face_usable else 0.0, "self_report": 0.65}
        if not face_usable:
            weights["context"] += 0.10
    else:
        weights = {"context": 0.75, "face": 0.25 if face_usable else 0.0, "self_report": 0.0}
        if not face_usable:
            weights["context"] = 1.0

    fused = {}
    for label in labels:
        fused[label] = weights["context"] * context_p[label]
        if face_p:
            fused[label] += weights["face"] * face_p[label]
        if self_report == label:
            fused[label] += weights["self_report"]
    fused = _normalize(fused, labels)
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    top_label, top_probability = ranked[0]

    # Self-report is authoritative even if another source conflicts.  The distribution
    # still exposes disagreement rather than silently deleting it.
    if self_report:
        top_label = self_report
        top_probability = fused[self_report]
    should_abstain = not self_report and top_probability < abstain_below

    evidence = [
        EvidenceItem("situation", weights["context"], "used",
                     "Context suggests possibilities, not a single certain feeling."),
        EvidenceItem("expression", weights["face"], "used" if face_usable else "ignored",
                     "Visible expression cue only." if face_usable else
                     "Missing or low-confidence expression evidence."),
        EvidenceItem("self_report", weights["self_report"], "used" if self_report else "missing",
                     "The person's own report has priority." if self_report else
                     "Ask the person rather than assuming."),
    ]
    alternatives = [
        {"emotion": label, "probability": round(probability, 4)}
        for label, probability in ranked[:3]
    ]
    info = emotion_info.get(top_label, {})
    if self_report:
        guidance = f"They said they feel {info.get('name', top_label).lower()}. Listen to them. {info.get('support', '')}".strip()
    elif should_abstain:
        guidance = "The evidence is uncertain. Ask how the person feels and listen to their answer."
    else:
        guidance = "These are possibilities, not a verdict. Ask, listen, and avoid assuming."

    return {
        "schema_version": "1.0",
        "decision": {
            "emotion": None if should_abstain else top_label,
            "probability": round(top_probability, 4),
            "should_abstain": should_abstain,
            "basis": "self_report" if self_report else "multimodal_inference",
        },
        "possibilities": alternatives,
        "evidence": [asdict(item) for item in evidence],
        "missing_information": [] if self_report else ["self_report"],
        "guidance": guidance,
        "disclaimer": "Expression and context are clues; they do not reveal a person's internal state.",
    }
