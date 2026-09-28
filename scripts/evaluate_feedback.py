"""Evaluate check-in feedback reviewers against the hand-labeled benchmark.

    python scripts/evaluate_feedback.py                       # offline rule baseline (default)
    MOSAIC_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... \\
        python scripts/evaluate_feedback.py --backend llm      # LLM reviewer

Reports per-criterion precision/recall/F1, derived-verdict accuracy, per-category accuracy, and every
failure, next to an always-majority baseline. LLM raw outputs are cached in
reports/feedback_llm_cache.jsonl so reruns are free and results are reproducible. Calls that fail or
return invalid JSON are counted and scored as the fallback answer; they are never dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from message_feedback import CRITERIA, LLMError, LLMReviewer, RuleReviewer, reviewer_from_env  # noqa: E402

VERDICT = "respectful"
KEYS = (*CRITERIA, VERDICT)


def load(split: str) -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / "data" / "checkin_eval.jsonl").read_text().splitlines() if line]
    return [r for r in rows if split == "all" or r["split"] == split]


def binary_scores(gold: list[bool], pred: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(p and not g for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "accuracy": round(accuracy, 3), "positives": sum(gold), "n": len(gold)}


def score(rows: list[dict], preds: list[dict]) -> dict:
    per = {k: binary_scores([r["labels"][k] for r in rows], [p[k] for p in preds]) for k in KEYS}
    macro_f1 = round(sum(per[k]["f1"] for k in CRITERIA) / len(CRITERIA), 3)
    exact = round(sum(all(r["labels"][k] == p[k] for k in CRITERIA) for r, p in zip(rows, preds)) / len(rows), 3)
    by_cat = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        by_cat[r["category"]][0] += r["labels"][VERDICT] == p[VERDICT]
        by_cat[r["category"]][1] += 1
    return {
        "criteria": per,
        "macro_f1_criteria": macro_f1,
        "exact_match": exact,
        "verdict_accuracy": per[VERDICT]["accuracy"],
        "by_category_verdict_accuracy": {c: {"correct": a, "n": n, "accuracy": round(a / n, 3)}
                                         for c, (a, n) in sorted(by_cat.items())},
    }


def majority_preds(rows: list[dict]) -> list[dict]:
    majority = {k: sum(r["labels"][k] for r in rows) * 2 > len(rows) for k in KEYS}
    return [dict(majority) for _ in rows]


def run_reviewer(reviewer, rows: list[dict], cache_path: Path) -> tuple[list[dict], list[dict], int]:
    cache = {}
    if isinstance(reviewer, LLMReviewer) and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            item = json.loads(line)
            cache[(item["model"], item["message"])] = item["raw"]
    preds, outputs, fallbacks = [], [], 0
    for r in rows:
        if isinstance(reviewer, LLMReviewer):
            key = (reviewer.model, r["message"])
            if key not in cache:
                try:
                    cache[key] = reviewer.complete(r["message"])
                except LLMError as exc:
                    cache[key] = None
                    print(f"  ! {r['id']}: {exc}", file=sys.stderr)
                if cache[key] is not None:
                    with cache_path.open("a") as fh:
                        fh.write(json.dumps({"model": reviewer.model, "message": r["message"], "raw": cache[key]}) + "\n")
            out = reviewer.review(r["message"], raw=cache[key] if cache[key] is not None else "")
        else:
            out = reviewer.review(r["message"])
        d = out.to_dict()
        fallbacks += d["fallback_reason"] is not None
        preds.append({**d["labels"], VERDICT: d["respectful"]})
        outputs.append(d)
    return preds, outputs, fallbacks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "llm"], default="rules")
    ap.add_argument("--split", choices=["test", "dev", "all"], default="test")
    args = ap.parse_args()

    rows = load(args.split)
    if args.backend == "llm":
        reviewer = reviewer_from_env()
        if not isinstance(reviewer, LLMReviewer):
            sys.exit("Set MOSAIC_LLM_PROVIDER (anthropic|openai) and an API key to evaluate the LLM reviewer.")
    else:
        reviewer = RuleReviewer()

    preds, outputs, fallbacks = run_reviewer(reviewer, rows, ROOT / "reports" / "feedback_llm_cache.jsonl")
    result = score(rows, preds)
    failures = [
        {"id": r["id"], "category": r["category"], "message": r["message"],
         "wrong": {k: {"gold": r["labels"][k], "pred": p[k]} for k in KEYS if r["labels"][k] != p[k]},
         "suggestion": o["suggestion"]}
        for r, p, o in zip(rows, preds, outputs) if any(r["labels"][k] != p[k] for k in KEYS)
    ]
    report = {
        "backend": reviewer.name,
        "split": args.split,
        "n": len(rows),
        "fallbacks": fallbacks,
        **result,
        "majority_baseline": {k: v for k, v in score(rows, majority_preds(rows)).items()
                              if k in ("macro_f1_criteria", "exact_match", "verdict_accuracy")},
        "failures": failures,
        "note": ("Single-annotator labels written by the project author; the rule baseline was developed on the dev "
                 "split. Scores measure agreement with this rubric, not real-world helpfulness."),
    }
    tag = "rules" if args.backend == "rules" else "llm"
    out = ROOT / "reports" / f"feedback_eval_{tag}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"{reviewer.name} on {args.split} (n={len(rows)}, fallbacks={fallbacks})")
    print(f"{'criterion':<16}{'P':>7}{'R':>7}{'F1':>7}{'acc':>7}{'pos':>6}")
    for k in KEYS:
        s = result["criteria"][k]
        print(f"{k:<16}{s['precision']:>7.3f}{s['recall']:>7.3f}{s['f1']:>7.3f}{s['accuracy']:>7.3f}{s['positives']:>6}")
    mb = report["majority_baseline"]
    print(f"macro-F1 (4 criteria) {result['macro_f1_criteria']:.3f}  exact match {result['exact_match']:.3f}  "
          f"verdict acc {result['verdict_accuracy']:.3f}")
    print(f"majority baseline:    {mb['macro_f1_criteria']:.3f}  exact match {mb['exact_match']:.3f}  "
          f"verdict acc {mb['verdict_accuracy']:.3f}")
    print("verdict accuracy by category:")
    for c, v in result["by_category_verdict_accuracy"].items():
        print(f"  {c:<24}{v['correct']:>3}/{v['n']:<3} {v['accuracy']:.2f}")
    print(f"{len(failures)} failures -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
