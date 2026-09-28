"""Tests for check-in feedback: rule reviewer, LLM output validation/fallback, API and benchmark."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app import create_app  # noqa: E402
from message_feedback import (CRITERIA, LLMError, LLMReviewer, RuleReviewer, parse_llm_output,  # noqa: E402
                              reviewer_from_env)
from test_app import fake_predictor  # noqa: E402


class FakeLLM(LLMReviewer):
    """LLMReviewer whose network call is replaced by a canned reply or error."""

    def __init__(self, reply=None, error=None):
        super().__init__("anthropic", "test-key")
        self.reply, self.error, self.calls = reply, error, 0

    def complete(self, message):
        self.calls += 1
        if self.error:
            raise LLMError(self.error)
        return self.reply


GOOD_JSON = json.dumps({"assumes_feeling": True, "judgmental": False, "pressuring": False, "checks_in": True,
                        "explanations": {"assumes_feeling": "'You look upset' names their feeling."},
                        "suggestion": "I noticed you went quiet. Want to talk?"})


class RuleReviewerTests(unittest.TestCase):
    def setUp(self):
        self.r = RuleReviewer()

    def labels(self, text):
        return self.r.review(text).to_dict()

    def test_respectful_check_in(self):
        out = self.labels("Would you like to talk, or have some quiet time?")
        self.assertTrue(out["respectful"])
        self.assertEqual(out["issues"], [])

    def test_assumption_is_flagged_with_quote(self):
        out = self.labels("You look sad. Do you want to talk?")
        self.assertTrue(out["labels"]["assumes_feeling"])
        self.assertFalse(out["respectful"])
        self.assertIn("you look sad", out["issues"][0]["explanation"])

    def test_hedged_confirmation_is_not_assumption(self):
        self.assertFalse(self.labels("You seem a bit quiet. Am I reading that right?")["labels"]["assumes_feeling"])

    def test_presupposing_why_is_assumption_not_check_in(self):
        out = self.labels("Why are you so angry?")["labels"]
        self.assertTrue(out["assumes_feeling"])
        self.assertFalse(out["checks_in"])

    def test_pressure_and_exemption(self):
        self.assertTrue(self.labels("Tell me what happened.")["labels"]["pressuring"])
        self.assertFalse(self.labels("Maybe it's loud? Tell me if I'm off.")["labels"]["pressuring"])

    def test_rejects_bad_input(self):
        for bad in (None, "", "   ", 5, "x" * 501):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.r.review(bad)


class LLMReviewerTests(unittest.TestCase):
    def test_valid_output_is_used(self):
        out = FakeLLM(reply="Sure!\n" + GOOD_JSON).review("You look upset. Want to talk?").to_dict()
        self.assertTrue(out["backend"].startswith("llm:anthropic"))
        self.assertIsNone(out["fallback_reason"])
        self.assertTrue(out["labels"]["assumes_feeling"])
        self.assertEqual(out["suggestion"], "I noticed you went quiet. Want to talk?")

    def test_malformed_output_falls_back_to_rules_visibly(self):
        for reply in ("not json", '{"assumes_feeling": "yes"}', "{broken"):
            with self.subTest(reply=reply):
                out = FakeLLM(reply=reply).review("Tell me what happened.").to_dict()
                self.assertIn("fallback", out["backend"])
                self.assertIsNotNone(out["fallback_reason"])
                self.assertTrue(out["labels"]["pressuring"])  # the rule answer, not a silent default

    def test_network_error_falls_back(self):
        out = FakeLLM(error="timeout").review("You okay?").to_dict()
        self.assertEqual(out["fallback_reason"], "timeout")

    def test_explanations_limited_to_known_criteria(self):
        raw = json.dumps({**json.loads(GOOD_JSON), "explanations": {"hack": "x", "judgmental": "y" * 999}})
        review = parse_llm_output(raw, "llm:test")
        self.assertEqual(set(review.explanations), {"judgmental"})
        self.assertEqual(len(review.explanations["judgmental"]), 300)

    def test_llm_is_opt_in(self):
        self.assertIsInstance(reviewer_from_env({}), RuleReviewer)
        self.assertIsInstance(reviewer_from_env({"MOSAIC_LLM_PROVIDER": "anthropic"}), RuleReviewer)  # no key
        r = reviewer_from_env({"MOSAIC_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k", "MOSAIC_LLM_MODEL": "m"})
        self.assertIsInstance(r, LLMReviewer)
        self.assertEqual((r.provider, r.model), ("openai", "m"))


class FeedbackApiTests(unittest.TestCase):
    def client(self, reviewer=None):
        return create_app(fake_predictor(), feedback_reviewer=reviewer or RuleReviewer()).test_client()

    def test_feedback_api_schema(self):
        r = self.client().post("/api/feedback", json={"message": "Stop being weird. Tell me now."})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(set(data["labels"]), set(CRITERIA))
        self.assertFalse(data["respectful"])
        self.assertFalse(data["sent_to_ai_provider"])
        self.assertTrue({"judgmental", "pressuring"} <= {i["id"] for i in data["issues"]})

    def test_feedback_api_rejects_bad_payload(self):
        c = self.client()
        self.assertEqual(c.post("/api/feedback", json={}).status_code, 400)
        self.assertEqual(c.post("/api/feedback", json={"message": "x" * 600}).status_code, 400)

    def test_status_reports_whether_text_sent_to_ai_provider(self):
        self.assertFalse(self.client().get("/api/status").get_json()["feedback_sent_to_ai_provider"])
        llm_client = self.client(FakeLLM(reply=GOOD_JSON))
        self.assertTrue(llm_client.get("/api/status").get_json()["feedback_sent_to_ai_provider"])

    def test_llm_fallback_still_reports_text_was_sent_to_provider(self):
        data = self.client(FakeLLM(error="down")).post("/api/feedback", json={"message": "You okay?"}).get_json()
        self.assertTrue(data["sent_to_ai_provider"])  # never under-report where the text went
        self.assertEqual(data["fallback_reason"], "down")


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_is_valid_and_split_is_disjoint(self):
        rows = [json.loads(line) for line in (ROOT / "data" / "checkin_eval.jsonl").read_text().splitlines()]
        dev = {r["message"] for r in rows if r["split"] == "dev"}
        test = {r["message"] for r in rows if r["split"] == "test"}
        self.assertFalse(dev & test)
        self.assertGreaterEqual(len(test), 90)
        for r in rows:
            lab = r["labels"]
            expected = lab["checks_in"] and not (lab["assumes_feeling"] or lab["judgmental"] or lab["pressuring"])
            self.assertEqual(lab["respectful"], expected, r["id"])

    def test_evaluator_scores_and_never_approves_disrespectful_messages(self):
        import evaluate_feedback as ev
        rows = ev.load("test")
        preds, _, fallbacks = ev.run_reviewer(RuleReviewer(), rows, ROOT / "reports" / "unused.jsonl")
        result = ev.score(rows, preds)
        self.assertEqual(fallbacks, 0)
        self.assertGreater(result["macro_f1_criteria"], ev.score(rows, ev.majority_preds(rows))["macro_f1_criteria"])
        # The costly error: telling a learner a disrespectful message is fine.
        self.assertEqual(result["criteria"]["respectful"]["precision"], 1.0)

    def test_evaluation_page_shows_feedback_benchmark(self):
        html = self.client_html()
        self.assertIn("Check-in feedback benchmark", html)
        self.assertIn("All four criteria right", html)

    def client_html(self):
        return create_app(fake_predictor(), feedback_reviewer=RuleReviewer()).test_client().get("/evaluation").get_data(as_text=True)


if __name__ == "__main__":
    unittest.main()
