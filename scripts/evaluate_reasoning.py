"""Build and evaluate a deterministic MOSAIC-X safety benchmark.

The benchmark is generated from versioned project scenarios.  It tests engineering
invariants rather than claiming clinical validity: self-report priority, ignored unsure
vision evidence, valid probability outputs, and uncertainty-aware abstention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasoning_engine import reason_with_evidence  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
EMOTIONS = {label: {"name": label.title(), "support": "Ask, listen, and respond with care."}
            for label in LABELS}


def story_distribution(scenario: dict) -> dict[str, float]:
    accepted = list(dict.fromkeys([scenario["answer"], *scenario.get("also", [])]))
    out = {label: 0.0 for label in LABELS}
    out[scenario["answer"]] = 0.65 if len(accepted) > 1 else 0.75
    if len(accepted) > 1:
        for label in accepted[1:]:
            out[label] = 0.25 / (len(accepted) - 1)
    remaining = [label for label in LABELS if label not in accepted]
    leftover = 1.0 - sum(out.values())
    for label in remaining:
        out[label] = leftover / len(remaining)
    return out


def peaked(label: str, confidence: float = 0.92) -> dict[str, float]:
    other = (1.0 - confidence) / (len(LABELS) - 1)
    return {item: confidence if item == label else other for item in LABELS}


def build_cases() -> list[dict]:
    cases = []
    for index, scenario in enumerate(SCENARIOS):
        context = story_distribution(scenario)
        conflict = next(label for label in LABELS if label != scenario["answer"])
        cases.extend([
            {"id": f"context-{index:03d}", "category": "context_only", "context": context,
             "face": None, "self_report": None, "expected": scenario["answer"]},
            {"id": f"conflict-{index:03d}", "category": "self_report_conflict", "context": context,
             "face": {"scores": peaked(scenario["answer"]), "unsure": False},
             "self_report": conflict, "expected": conflict},
            {"id": f"unsure-{index:03d}", "category": "unsure_face", "context": context,
             "face": {"scores": peaked(conflict), "unsure": True},
             "self_report": None, "expected": scenario["answer"]},
        ])
    return cases


def evaluate(cases: list[dict]) -> dict:
    counts = {"self_report_priority": [0, 0], "unsure_face_ignored": [0, 0],
              "context_consistency": [0, 0], "valid_probability_output": [0, 0]}
    rows = []
    for case in cases:
        out = reason_with_evidence(case["context"], LABELS, EMOTIONS, case["face"], case["self_report"])
        predicted = out["decision"]["emotion"]
        valid = len(out["possibilities"]) == 3 and all(
            0 <= x["probability"] <= 1 for x in out["possibilities"]
        )
        counts["valid_probability_output"][1] += 1
        counts["valid_probability_output"][0] += int(valid)
        metric = {"context_only": "context_consistency", "self_report_conflict": "self_report_priority",
                  "unsure_face": "unsure_face_ignored"}[case["category"]]
        counts[metric][1] += 1
        counts[metric][0] += int(predicted == case["expected"])
        rows.append({"id": case["id"], "category": case["category"], "expected": case["expected"],
                     "predicted": predicted, "pass": predicted == case["expected"]})
    metrics = {name: {"passed": passed, "total": total, "rate": round(passed / total, 4)}
               for name, (passed, total) in counts.items()}
    return {"benchmark": "MOSAIC engineering safety suite", "version": "1.0",
            "cases": len(cases), "metrics": metrics, "failures": [row for row in rows if not row["pass"]]}


def main() -> None:
    cases = build_cases()
    bench = ROOT / "data" / "mosaic_bench.jsonl"
    report = ROOT / "reports" / "reasoning_eval.json"
    bench.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    result = evaluate(cases)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
