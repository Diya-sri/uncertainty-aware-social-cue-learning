"""Face detection + emotion prediction, kept separate from the web app so it can be tested alone.

Two model formats are supported:
  * models/emotion_model.keras (default) fine-tuned EfficientNet model made by scripts/train.py.
  * models/emotion_model.npz   legacy NumPy MobileNetV2 ensemble fallback.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
DEFAULT_HIGH_PRECISION_THRESHOLD = 0.75
if not hasattr(cv2, "CascadeClassifier"):
    raise ImportError(
        f"OpenCV {cv2.__version__} has no Haar face detector (removed in OpenCV 5). "
        'Fix: pip install "opencv-python-headless>=4.9,<5"'
    )
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MARGIN = 0.10  # extra border around the detector's box, on each side


def detect_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Faces as (x, y, w, h), largest first."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return sorted((tuple(int(v) for v in f) for f in faces), key=lambda b: b[2] * b[3], reverse=True)


def crop_box(bgr: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    m = int(MARGIN * max(w, h))
    H, W = bgr.shape[:2]
    return bgr[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]


def crop_face_like_app(bgr: np.ndarray) -> np.ndarray:
    """Used on *training* images so they are framed like the app's webcam crops.
    Images that are already tight face crops (face fills > 60% or isn't found) are kept."""
    faces = detect_faces(bgr)
    if faces:
        x, y, w, h = faces[0]
        if w * h < 0.6 * bgr.shape[0] * bgr.shape[1]:
            return crop_box(bgr, faces[0])
    return bgr


@dataclass
class FacePrediction:
    box: tuple[int, int, int, int]  # x, y, w, h in the input image's pixels
    emotion: str
    confidence: float
    scores: dict[str, float]
    unsure: bool = False

    def to_dict(self):
        x, y, w, h = self.box
        return {"box": {"x": x, "y": y, "w": w, "h": h}, "emotion": self.emotion,
                "confidence": round(self.confidence, 4), "unsure": self.unsure,
                "scores": {k: round(v, 4) for k, v in self.scores.items()}}


class NumpyModel:
    """MobileNetV2 features + linear head; returns calibrated softmax probabilities.

    meta["temperature"] (fit by scripts/calibrate.py) softens the over-confident raw
    probabilities so that "70% sure" really means right about 70% of the time.
    """

    def __init__(self, path: Path):
        from mobilenet_np import MobileNetV2

        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        self.net = MobileNetV2({k: d[k] for k in d.files if k.startswith("bb/")})
        self.W, self.b = d["head/W"].astype(np.float32), d["head/b"].astype(np.float32)
        self.ensemble = None
        if self.meta.get("ensemble"):
            import joblib
            ensemble_path = path.with_name(self.meta["ensemble"]["path"])
            self.ensemble = joblib.load(ensemble_path)

    def feature_views(self, batch: np.ndarray) -> np.ndarray:
        fs = self.meta.get("feature_set", "gap")
        views = [batch, batch[:, :, ::-1]] if self.meta.get("flip_tta") else [batch]
        return np.stack([self.net.features(v, fs) for v in views])

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """Raw logits per test-time view: (views, N, classes)."""
        return np.stack([f @ self.W + self.b for f in self.feature_views(batch)])

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        t = float(self.meta.get("temperature", 1.0))
        features = self.feature_views(batch)
        base = _softmax(np.stack([f @ self.W + self.b for f in features]) / t).mean(axis=0)
        if not self.ensemble:
            return base
        x = self.ensemble["scaler"].transform(features.mean(axis=0))
        x = self.ensemble["pca"].transform(x)
        extra = self.ensemble["model"].predict_proba(x)
        weight = float(self.meta["ensemble"].get("base_weight", 0.7))
        blended = weight * base + (1.0 - weight) * extra
        ensemble_t = float(self.meta.get("ensemble_temperature", 1.0))
        return _softmax(np.log(np.clip(blended, 1e-8, 1.0)) / ensemble_t)


class KerasCalibratedModel:
    """Apply validation-selected flip TTA and temperature scaling."""

    def __init__(self, model, temperature: float = 1.0, flip_tta_weight: float = 0.0):
        self.model = model
        self.temperature = max(float(temperature), 1e-6)
        self.flip_tta_weight = min(max(float(flip_tta_weight), 0.0), 1.0)

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        raw = np.asarray(self.model(batch, training=training), dtype=np.float32)
        if self.flip_tta_weight:
            flipped = np.asarray(self.model(batch[:, :, ::-1, :], training=training), dtype=np.float32)
            raw = (1.0 - self.flip_tta_weight) * raw + self.flip_tta_weight * flipped
        logp = np.log(np.clip(raw, 1e-8, 1.0))
        return _softmax(logp / self.temperature)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class EmotionPredictor:
    """Loads the model lazily and predicts an emotion for every face in an image."""

    def __init__(self, model_path: str | Path, labels_path: str | Path | None = None):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path) if labels_path else self.model_path.with_name("labels.json")
        self.labels = DEFAULT_LABELS
        self.image_size = 224
        self.unsure_below = 0.0  # calibrated confidence below which the app says "not sure"
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    # -- model loading -------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._load()

    @property
    def error(self) -> str | None:
        self._load()
        return self._error

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if not self.model_path.exists():
                self._error = (f"Model file not found at '{self.model_path}'. Train one with "
                               f"scripts/train_numpy.py (see README) or set MODEL_PATH.")
                log.warning(self._error)
                return False
            try:
                if self.model_path.suffix == ".npz":
                    model = NumpyModel(self.model_path)
                    self.labels = model.meta["labels"]
                    self.image_size = int(model.meta.get("image_size", 224))
                    saved_threshold = float(model.meta.get("unsure_below", 0.0))
                    configured = float(os.environ.get(
                        "UNSURE_BELOW", max(saved_threshold, DEFAULT_HIGH_PRECISION_THRESHOLD)
                    ))
                    self.unsure_below = min(max(configured, 0.0), 1.0)
                else:
                    meta = {}
                    if self.labels_path.exists():
                        meta = json.loads(self.labels_path.read_text())
                        self.labels = meta.get("labels", DEFAULT_LABELS)
                        self.image_size = int(meta.get("image_size", 224))
                        # A confidence threshold is model-specific. Never reuse the legacy
                        # NumPy model's 0.75 operating point for an uncalibrated Keras model.
                        saved_threshold = float(meta.get("unsure_below", 0.0))
                        configured = float(os.environ.get("UNSURE_BELOW", saved_threshold))
                        self.unsure_below = min(max(configured, 0.0), 1.0)
                    from tensorflow import keras  # optional dependency

                    raw_model = keras.models.load_model(self.model_path, compile=False)
                    model = KerasCalibratedModel(raw_model, meta.get("temperature", 1.0),
                                                 meta.get("flip_tta_weight", 0.0))
                self._model = model
                log.info("Loaded model %s with labels %s", self.model_path, self.labels)
                return True
            except Exception as exc:  # pragma: no cover - depends on environment
                self._error = f"Could not load model: {exc}"
                log.exception("Model load failed")
                return False

    # -- inference -----------------------------------------------------------------------
    def detect_faces(self, bgr: np.ndarray):
        return detect_faces(bgr)

    def _prep(self, bgr: np.ndarray, box) -> np.ndarray:
        gray = cv2.cvtColor(crop_box(bgr, box), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.repeat(gray[..., None], 3, axis=-1).astype("float32")  # 0..255, same as training

    def predict(self, bgr: np.ndarray, max_faces: int = 4) -> list[FacePrediction]:
        if not self._load():
            raise RuntimeError(self._error)
        boxes = detect_faces(bgr)[:max_faces]
        if not boxes:
            return []
        batch = np.stack([self._prep(bgr, b) for b in boxes])
        with self._lock:  # NumPy/Keras inference is not guaranteed thread-safe
            probs = np.asarray(self._model(batch, training=False))
        results = []
        for box, p in zip(boxes, probs):
            i = int(np.argmax(p))
            results.append(FacePrediction(box=box, emotion=self.labels[i], confidence=float(p[i]),
                                          scores={l: float(s) for l, s in zip(self.labels, p)},
                                          unsure=float(p[i]) < self.unsure_below))
        return results
