"""Tests that run without TensorFlow: a stand-in model replaces the Keras network,
while face detection (OpenCV), request parsing and all routes are exercised for real.

Run:  python -m unittest discover -s tests -v     (or: pytest)
"""

from __future__ import annotations

import base64
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import EMOTIONS, create_app, _story_distribution  # noqa: E402
from emotion_model import DEFAULT_LABELS, EmotionPredictor, KerasCalibratedModel  # noqa: E402
from reasoning_engine import reason_with_evidence  # noqa: E402



def _find_face_fixture() -> Path | None:
    """Use a face from the local (git-ignored) data split, padded to look like a webcam frame.
    No personal photos are committed to the repo; face tests are skipped if there is no data."""
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    out = ROOT / "tests" / ".face_fixture.jpg"
    for p in sorted((ROOT / "data" / "split" / "test").glob("*/*.jpg")):
        img = cv2.imread(str(p))
        if img is None or img.shape[0] < 250:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(casc.detectMultiScale(g, 1.1, 5, minSize=(48, 48))) == 1:
            cv2.imwrite(str(out), cv2.copyMakeBorder(img, 80, 80, 160, 160, cv2.BORDER_CONSTANT, value=(210, 205, 200)))
            return out
    return None


FIXTURE = _find_face_fixture() or ROOT / "tests" / ".missing"


class FakeModel:
    """Always answers 'happiness' with 80% confidence; records the batch it received."""

    def __init__(self):
        self.last_batch = None

    def __call__(self, batch, training=False):
        self.last_batch = batch
        p = np.full((len(batch), len(DEFAULT_LABELS)), 0.2 / (len(DEFAULT_LABELS) - 1), dtype="float32")
        p[:, DEFAULT_LABELS.index("happiness")] = 0.8
        return p


def fake_predictor():
    pred = EmotionPredictor("does-not-matter.keras")
    pred._model = FakeModel()
    return pred


def data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class PredictorTests(unittest.TestCase):
    def test_keras_probability_adapter_applies_temperature(self):
        class ProbModel:
            def __call__(self, batch, training=False):
                return np.tile(np.array([[0.9, 0.1]], dtype="float32"), (len(batch), 1))
        raw = ProbModel()(np.zeros((1, 1)))
        calibrated = KerasCalibratedModel(ProbModel(), temperature=2.0)(np.zeros((1, 1)))
        self.assertAlmostEqual(float(calibrated.sum()), 1.0, places=5)
        self.assertLess(float(calibrated.max()), float(raw.max()))

    def test_keras_probability_adapter_supports_flip_tta(self):
        class PositionModel:
            def __call__(self, batch, training=False):
                first = batch[:, 0, 0, 0]
                return np.stack([first, 1.0 - first], axis=1).astype("float32")
        batch = np.array([[[[1.0], [0.0]]]], dtype="float32")
        probs = KerasCalibratedModel(PositionModel(), flip_tta_weight=0.5)(batch)
        self.assertTrue(np.allclose(probs, [[0.5, 0.5]], atol=1e-5))

    def test_missing_model_reports_helpful_error(self):
        p = EmotionPredictor("models/nope.keras")
        self.assertFalse(p.ready)
        self.assertIn("Model file not found", p.error)
        with self.assertRaises(RuntimeError):
            p.predict(np.zeros((100, 100, 3), np.uint8))

    def test_no_face_returns_empty_list(self):
        self.assertEqual(fake_predictor().predict(np.full((240, 320, 3), 127, np.uint8)), [])

    @unittest.skipUnless(FIXTURE.exists(), "face fixture missing")
    def test_face_is_found_and_preprocessed_like_training(self):
        pred = fake_predictor()
        img = cv2.imread(str(FIXTURE))
        faces = pred.predict(img)
        self.assertGreaterEqual(len(faces), 1)
        self.assertEqual(faces[0].emotion, "happiness")
        batch = pred._model.last_batch
        self.assertEqual(batch.shape[1:], (224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertGreater(batch.max(), 1.0)  # 0..255 range, not 0..1
        self.assertTrue(np.array_equal(batch[0, ..., 0], batch[0, ..., 1]))  # grayscale as RGB


class ReasoningTests(unittest.TestCase):
    def test_self_report_wins_conflicting_evidence(self):
        context = {label: float(label == "happiness") for label in DEFAULT_LABELS}
        face = {"scores": context, "unsure": False}
        out = reason_with_evidence(context, DEFAULT_LABELS, EMOTIONS, face, "sadness")
        self.assertEqual(out["decision"]["emotion"], "sadness")
        self.assertEqual(out["decision"]["basis"], "self_report")

    def test_unsure_face_is_ignored(self):
        context = {label: float(label == "sadness") for label in DEFAULT_LABELS}
        face = {"scores": {label: float(label == "happiness") for label in DEFAULT_LABELS}, "unsure": True}
        out = reason_with_evidence(context, DEFAULT_LABELS, EMOTIONS, face)
        self.assertEqual(out["decision"]["emotion"], "sadness")
        expression = next(x for x in out["evidence"] if x["source"] == "expression")
        self.assertEqual(expression["status"], "ignored")


MODEL = ROOT / "models" / "emotion_model.npz"


@unittest.skipUnless(MODEL.exists(), "trained model not present")
class RealModelTests(unittest.TestCase):
    def test_numpy_model_loads_and_outputs_probabilities(self):
        pred = EmotionPredictor(MODEL)
        self.assertTrue(pred.ready, pred.error)
        self.assertEqual(pred.labels, DEFAULT_LABELS)
        batch = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
        probs = pred._model(batch)
        self.assertEqual(probs.shape, (2, len(DEFAULT_LABELS)))
        np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-4)

    def test_model_is_calibrated(self):
        pred = EmotionPredictor(MODEL)
        self.assertTrue(pred.ready)
        self.assertGreater(pred._model.meta.get("temperature", 1.0), 1.0)  # raw model was over-confident
        self.assertGreater(pred.unsure_below, 0.0)

    def test_mobilenet_matches_imagenet_reference(self):
        """The NumPy MobileNetV2 must reproduce Keras: Grace Hopper -> 'military uniform' (652)."""
        weights = ROOT / "tests" / ".mnv2_top.npz"
        if not weights.exists():
            self.skipTest("reference ImageNet head not converted (see scripts/convert_mobilenetv2.py)")
        import matplotlib
        from mobilenet_np import MobileNetV2
        w = dict(np.load(weights))
        img = cv2.imread(str(Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"))[:, :, ::-1]
        x = cv2.resize(img, (224, 224)).astype("float32")[None]
        logits = MobileNetV2(w).features(x) @ w["imagenet/k"] + w["imagenet/b"]
        self.assertEqual(int(logits.argmax()), 652)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(fake_predictor())
        self.client = self.app.test_client()

    def test_pages_render(self):
        for path in ["/", "/practice", "/reason", "/learn", "/mirror", "/quiz", "/music", "/evaluation", "/progress", "/about"]:
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertIn(b"MOSAIC", r.data)

    def test_status(self):
        r = self.client.get("/api/status").get_json()
        self.assertTrue(r["model_ready"])
        self.assertEqual(r["labels"], DEFAULT_LABELS)

    def test_communication_practice_api(self):
        r = self.client.get("/api/practice/next").get_json()
        self.assertIn("situation", r)
        self.assertGreaterEqual(len(r["possibilities"]), 2)
        self.assertEqual(len(r["responses"]), 3)
        self.assertIn(r["best"], range(len(r["responses"])))

    @unittest.skipUnless(MODEL.exists(), "trained model not present")
    def test_shipped_model_uses_high_precision_threshold(self):
        r = create_app(EmotionPredictor(MODEL)).test_client().get("/api/status").get_json()
        self.assertEqual(r["unsure_below"], 0.75)

    def test_predict_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"image": "not base64!!"}).status_code, 400)

    def test_predict_blank_image_has_no_faces(self):
        r = self.client.post("/api/predict", json={"image": data_url(np.full((240, 320, 3), 200, np.uint8))})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["faces"], [])

    @unittest.skipUnless(FIXTURE.exists(), "face fixture missing")
    def test_predict_face_json_and_multipart(self):
        img = cv2.imread(str(FIXTURE))
        r = self.client.post("/api/predict", json={"image": data_url(img)}).get_json()
        self.assertEqual(r["faces"][0]["emotion"], "happiness")
        self.assertEqual(set(r["faces"][0]["box"]), {"x", "y", "w", "h"})
        with open(FIXTURE, "rb") as f:
            r2 = self.client.post("/api/predict", data={"image": (f, "face.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r2.status_code, 200)

    def test_predict_503_when_model_missing(self):
        client = create_app(EmotionPredictor("models/nope.keras")).test_client()
        r = client.post("/api/predict", json={"image": data_url(np.zeros((50, 50, 3), np.uint8))})
        self.assertEqual(r.status_code, 503)

    def test_too_large_upload(self):
        big = "data:image/jpeg;base64," + "A" * (4 * 1024 * 1024)
        self.assertEqual(self.client.post("/api/predict", json={"image": big}).status_code, 413)

    def test_no_secrets_or_email_in_source(self):
        src = (ROOT / "app.py").read_text() + (ROOT / "emotion_model.py").read_text()
        for bad in ["smtplib", "@gmail.com", "password ="]:
            self.assertNotIn(bad, src)

    def test_story_quiz_api(self):
        from scenarios import SCENARIOS
        seen = set()
        for _ in range(30):
            r = self.client.get("/api/story/next?exclude=0,1,2").get_json()
            self.assertNotIn(r["id"], {0, 1, 2})
            self.assertIn(r["answer"], [c["id"] for c in r["choices"]])
            self.assertIn(r["answer"], r["accepted"])
            self.assertTrue(set(r["accepted"]).issubset({c["id"] for c in r["choices"]}))
            self.assertAlmostEqual(sum(r["distribution"].values()), 1.0, places=3)
            self.assertEqual(set(r["distribution"]), set(DEFAULT_LABELS))
            self.assertEqual(len({c["id"] for c in r["choices"]}), 4)
            self.assertTrue(r["why"])
            seen.add(r["id"])
        self.assertGreater(len(seen), 5)
        self.assertEqual({s["answer"] for s in SCENARIOS}, set(DEFAULT_LABELS))
        self.assertTrue(all(len({s["answer"], *s.get("also", [])}) <= 3 for s in SCENARIOS))

    def test_reason_api_is_structured_and_respects_self_report(self):
        context = _story_distribution({"answer": "happiness"})
        r = self.client.post("/api/reason", json={"context": context, "self_report": "sadness"})
        self.assertEqual(r.status_code, 200)
        out = r.get_json()
        self.assertEqual(out["schema_version"], "1.0")
        self.assertEqual(out["decision"]["emotion"], "sadness")
        self.assertIn("disclaimer", out)

    def test_reason_api_rejects_bad_payload(self):
        self.assertEqual(self.client.post("/api/reason", json={}).status_code, 400)
        context = _story_distribution({"answer": "happiness"})
        self.assertEqual(self.client.post("/api/reason", json={"context": context, "self_report": "invalid"}).status_code, 400)

    def test_quiz_page_works_without_photos(self):
        r = self.client.get("/quiz")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Stories", r.data)

    def test_explore_page_offers_photo_upload(self):
        r = self.client.get("/learn")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'id="photo-file"', r.data)
        self.assertIn(b"upload a face photo", r.data)

    def test_unsure_flag(self):
        pred = fake_predictor()
        pred.unsure_below = 0.9  # FakeModel is 80% sure -> should be flagged
        if FIXTURE.exists():
            faces = pred.predict(cv2.imread(str(FIXTURE)))
            self.assertTrue(faces[0].unsure)
            self.assertTrue(faces[0].to_dict()["unsure"])

    def test_emotions_cover_model_labels(self):
        self.assertEqual(set(EMOTIONS), set(DEFAULT_LABELS))

    def test_evaluation_artifacts_are_served(self):
        self.assertEqual(self.client.get("/reports/confusion_matrix.png").status_code, 200)

    def test_evaluation_page_shows_shipped_v3_metrics(self):
        r = self.client.get("/evaluation")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("62.1%", html)
        self.assertIn("0.048", html)
        self.assertIn("69.7%", html)


if __name__ == "__main__":
    unittest.main()
