"""MOSAIC: an uncertainty-aware social-cue learning and evaluation prototype.

Privacy by design:
  * The webcam runs in the browser. Frames are sent only to this server (normally your own
    computer) for a prediction and are never saved or emailed.
  * Audio files never leave the browser.
  * Progress is stored in the browser's localStorage and can be exported or erased from
    the Progress page.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import random
import secrets
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for, send_from_directory

from emotion_model import EmotionPredictor
from reasoning_engine import reason_with_evidence
from scenarios import SCENARIOS
from communication_scenarios import COMMUNICATION_SCENARIOS
from message_feedback import LLMReviewer, reviewer_from_env

BASE_DIR = Path(__file__).resolve().parent
QUIZ_DIR = BASE_DIR / "static" / "quiz_faces"

EMOTIONS = {
    "anger": {"color": "#e2725b", "emoji": "😠", "name": "Angry",
              "support": "Pause, take slow breaths, and explain what felt unfair.",
              "tips": ["Eyebrows pulled down and together", "Eyes staring hard", "Lips pressed tight or teeth showing"]},
    "disgust": {"color": "#6fae8c", "emoji": "🤢", "name": "Disgusted",
                "support": "Move away from the unpleasant thing and say what is bothering you.",
                "tips": ["Nose wrinkled up", "Upper lip raised", "Looks like smelling something yucky"]},
    "fear": {"color": "#9483e6", "emoji": "😨", "name": "Scared",
             "support": "Find a trusted person, name what feels scary, and take one safe step at a time.",
             "tips": ["Eyebrows raised and pulled together", "Eyes wide open", "Mouth stretched back"]},
    "happiness": {"color": "#efb92f", "emoji": "😃", "name": "Happy",
                  "support": "Enjoy the moment or share the good feeling with someone.",
                  "tips": ["Corners of the mouth go up", "Cheeks lift", "Little crinkles next to the eyes"]},
    "neutral": {"color": "#8fa1b0", "emoji": "😐", "name": "Calm",
                "support": "Notice the calm feeling and continue at a comfortable pace.",
                "tips": ["Face is relaxed", "Mouth is straight", "Eyebrows are resting"]},
    "sadness": {"color": "#5b8fd6", "emoji": "😢", "name": "Sad",
                "support": "Talk to someone kind, ask for comfort, or take quiet time.",
                "tips": ["Inner eyebrows go up", "Corners of the mouth go down", "Eyes look down"]},
    "surprise": {"color": "#ec7fae", "emoji": "😲", "name": "Surprised",
                 "support": "Pause and find out more before deciding what to do.",
                 "tips": ["Eyebrows raised high", "Eyes very wide", "Mouth open in an O"]},
}

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("mosaic")


def create_app(predictor: EmotionPredictor | None = None, feedback_reviewer=None) -> Flask:
    app = Flask(__name__)
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        secret = secrets.token_hex(32)
        log.info("SECRET_KEY not set; using a random key for this run.")
    app.config.update(SECRET_KEY=secret, MAX_CONTENT_LENGTH=3 * 1024 * 1024)

    app.predictor = predictor or EmotionPredictor(
        os.environ.get("MODEL_PATH", BASE_DIR / "models" / "emotion_model.keras"),
        os.environ.get("LABELS_PATH") or None,
    )

    app.feedback_reviewer = feedback_reviewer or reviewer_from_env()
    log.info("Check-in feedback backend: %s", app.feedback_reviewer.name)

    @app.context_processor
    def inject():
        return {"EMOTIONS": EMOTIONS}

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/learn")
    def learn():
        return render_template("learn.html")

    @app.get("/mirror")
    def mirror():
        return render_template("mirror.html")

    @app.get("/quiz")
    def quiz():
        return render_template("quiz.html", has_faces=bool(_quiz_images()))

    @app.get("/music")
    def music():
        return render_template("music.html")

    @app.get("/progress")
    def progress():
        return render_template("progress.html")

    @app.get("/reason")
    def reason():
        return render_template("reason.html")

    @app.get("/practice")
    def practice():
        return render_template("practice.html")

    @app.get("/evaluation")
    def evaluation():
        def read_report(name):
            path = BASE_DIR / "reports" / name
            return json.loads(path.read_text()) if path.exists() else {}
        return render_template("evaluation.html", metrics=read_report("metrics_v3.json"),
                               calibration=read_report("calibration.json"),
                               audit=read_report("data_audit.json"),
                               reasoning=read_report("reasoning_eval.json"),
                               feedback_rules=read_report("feedback_eval_rules_test.json"),
                               feedback_llm=read_report("feedback_eval_llm_test.json"))

    @app.get("/reports/<path:name>")
    def report_file(name):
        return send_from_directory(BASE_DIR / "reports", name)

    @app.get("/about")
    def about():
        return render_template("about.html")

    # ---------------------------------------------------------------- API
    @app.get("/api/status")
    def status():
        p = app.predictor
        return jsonify({"model_ready": p.ready, "error": p.error, "labels": p.labels, "unsure_below": p.unsure_below,
                        "feedback_backend": app.feedback_reviewer.name,
                        "feedback_sent_to_ai_provider": isinstance(app.feedback_reviewer, LLMReviewer)})

    @app.post("/api/predict")
    def predict():
        """Body: JSON {"image": "data:image/jpeg;base64,..."} or multipart field 'image'."""
        img = _read_image(request)
        if img is None:
            return jsonify({"error": "Send a JPEG/PNG image as JSON {'image': dataURL} or multipart 'image'."}), 400
        if not app.predictor.ready:
            return jsonify({"error": app.predictor.error, "faces": []}), 503
        faces = app.predictor.predict(img)
        h, w = img.shape[:2]
        return jsonify({"width": w, "height": h, "faces": [f.to_dict() for f in faces]})

    @app.get("/api/quiz/next")
    def quiz_next():
        images = _quiz_images()
        if not images:
            return jsonify({"error": "No quiz faces found in static/quiz_faces/<emotion>/"}), 404
        label, rel = random.choice(images)
        others = random.sample([e for e in EMOTIONS if e != label], 3)
        choices = random.sample([label, *others], 4)
        return jsonify({"image": url_for("static", filename=f"quiz_faces/{rel}"), "answer": label,
                        "choices": [{"id": c, **{k: EMOTIONS[c][k] for k in ("emoji", "name")}} for c in choices]})

    @app.get("/api/story/next")
    def story_next():
        """A short situation + 4 feelings to choose from. ?exclude=3,7 avoids recent repeats."""
        exclude = {int(x) for x in request.args.get("exclude", "").split(",") if x.isdigit()}
        pool = [i for i in range(len(SCENARIOS)) if i not in exclude] or list(range(len(SCENARIOS)))
        i = random.choice(pool)
        sc = SCENARIOS[i]
        accepted = list(dict.fromkeys([sc["answer"], *sc.get("also", [])]))
        distribution = _story_distribution(sc)
        # Include every reasonable answer (up to 3), then fill the four choices.
        others = random.sample([e for e in EMOTIONS if e not in accepted], 4 - len(accepted))
        choices = random.sample([*accepted, *others], 4)
        return jsonify({"id": i, "text": sc["text"], "answer": sc["answer"], "accepted": accepted,
                        "distribution": distribution,
                        "why": sc["why"],
                        "choices": [{"id": c, **{k: EMOTIONS[c][k] for k in ("emoji", "name")}} for c in choices]})

    @app.get("/api/practice/next")
    def practice_next():
        """Return a scenario that teaches observation, ambiguity and respectful checking."""
        exclude = {int(x) for x in request.args.get("exclude", "").split(",") if x.isdigit()}
        pool = [i for i in range(len(COMMUNICATION_SCENARIOS)) if i not in exclude] or list(range(len(COMMUNICATION_SCENARIOS)))
        i = random.choice(pool)
        return jsonify({"id": i, **COMMUNICATION_SCENARIOS[i]})

    @app.post("/api/feedback")
    def feedback_api():
        """Review the wording of a learner's own check-in message.

        This app does not save the message. In LLM mode it is forwarded to the configured provider,
        whose retention depends on that provider's API terms.
        """
        data = request.get_json(silent=True) or {}
        try:
            review = app.feedback_reviewer.review(data.get("message"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        result = review.to_dict()
        # In LLM mode the text is sent (or may have been) even if the reply was unusable and rules answered.
        result["sent_to_ai_provider"] = isinstance(app.feedback_reviewer, LLMReviewer)
        return jsonify(result)

    @app.post("/api/reason")
    def reason_api():
        """Return an auditable multimodal decision; never infer a diagnosis or hidden state."""
        data = request.get_json(silent=True) or {}
        context = data.get("context")
        if not isinstance(context, dict):
            return jsonify({"error": "context must be a probability object"}), 400
        try:
            result = reason_with_evidence(
                context=context,
                labels=list(EMOTIONS),
                emotion_info=EMOTIONS,
                face=data.get("face") if isinstance(data.get("face"), dict) else None,
                self_report=data.get("self_report") or None,
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "Image too large (max 3 MB)."}), 413

    return app


def _story_distribution(sc: dict) -> dict[str, float]:
    """A transparent soft label: context suggests possibilities, never a certain inner state."""
    accepted = list(dict.fromkeys([sc["answer"], *sc.get("also", [])]))
    out = {e: 0.0 for e in EMOTIONS}
    out[sc["answer"]] = 0.65 if len(accepted) > 1 else 0.75
    if len(accepted) > 1:
        share = 0.25 / (len(accepted) - 1)
        for e in accepted[1:]:
            out[e] = share
    left = 1.0 - sum(out.values())
    other = [e for e in EMOTIONS if e not in accepted]
    for e in other:
        out[e] = left / len(other)
    return {k: round(v, 4) for k, v in out.items()}


def _quiz_images() -> list[tuple[str, str]]:
    out = []
    if QUIZ_DIR.is_dir():
        for label in EMOTIONS:
            d = QUIZ_DIR / label
            if d.is_dir():
                out += [(label, f"{label}/{p.name}") for p in d.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    return out


def _read_image(req) -> np.ndarray | None:
    raw = None
    if req.files.get("image"):
        raw = req.files["image"].read()
    else:
        data = req.get_json(silent=True) or {}
        s = data.get("image")
        if isinstance(s, str):
            s = s.split(",", 1)[1] if s.startswith("data:") else s
            try:
                raw = base64.b64decode(s, validate=True)
            except (binascii.Error, ValueError):
                return None
    if not raw:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img


app = create_app()

if __name__ == "__main__":
    # Localhost only by default: the webcam images should not be exposed on your network.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
