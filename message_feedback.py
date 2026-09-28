"""Feedback on a learner's own check-in message (the "Ask" step of Notice → Consider → Ask).

Two interchangeable reviewers share one output schema:

* ``RuleReviewer``: an offline, deterministic lexicon baseline. It is the default because it keeps
  the message on the device, and it is the comparison point for the LLM.
* ``LLMReviewer``: an optional language-model reviewer (Anthropic or OpenAI-compatible API),
  enabled only when ``MOSAIC_LLM_PROVIDER`` and an API key are set. Its JSON output is validated.
  If the call fails or the output is malformed, the rule reviewer answers instead and the
  response says so. It never fails silently.

Criteria and edge cases are defined in ``data/checkin_labeling_guide.md``. The reviewer judges
the *wording* of a message. It never judges the learner or the other person's feelings.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

CRITERIA = ("assumes_feeling", "judgmental", "pressuring", "checks_in")
MAX_MESSAGE_CHARS = 500
SCHEMA_VERSION = "1.0"

TIPS = {
    "assumes_feeling": "It states how they feel as a fact. Describe what you noticed, or ask, instead of naming their feeling for them.",
    "judgmental": "It labels, dismisses or makes fun of their reaction. Keep it neutral: their reaction can be valid even if you'd react differently.",
    "pressuring": "It demands an answer or action. Give them a real way to say no or to answer later.",
    "no_check_in": "It doesn't ask anything or offer support. Add a question or an offer they can accept or decline.",
}


@dataclass(frozen=True)
class Review:
    labels: dict
    explanations: dict
    suggestion: str
    backend: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        labels = {k: bool(self.labels[k]) for k in CRITERIA}
        respectful = labels["checks_in"] and not (
            labels["assumes_feeling"] or labels["judgmental"] or labels["pressuring"])
        issues = [k for k in ("assumes_feeling", "judgmental", "pressuring") if labels[k]]
        if not labels["checks_in"]:
            issues.append("no_check_in")
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "fallback_reason": self.fallback_reason,
            "labels": labels,
            "respectful": respectful,
            "issues": [{"id": i, "tip": TIPS[i], "explanation": self.explanations.get(i, "")} for i in issues],
            "suggestion": self.suggestion,
            "disclaimer": "Feedback is about wording only. It is automated and can be wrong.",
        }


def clean_message(message: object) -> str:
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    text = " ".join(message.split())
    if not text:
        raise ValueError("message must not be empty")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
    return text


# --------------------------------------------------------------------------- rule baseline

_FEELINGS = (
    r"sad|upset|angry|mad|annoyed|frustrated|nervous|anxious|scared|afraid|stressed(?: out)?|bored|tired|"
    r"happy|jealous|overwhelmed|lonely|hurt|embarrassed|worried|confused|left out|down"
)
_YOU_STATE = rf"\b(?:you(?:'re| are)|you (?:look|must be|feel)|you(?:'re| are) feeling)\s+(?:\w+\s+)?(?:{_FEELINGS})\b"
_ASSUME_PATTERNS = [
    _YOU_STATE,
    rf"\bi (?:can tell|know) you(?:'re| are)\b",
    rf"\bwhy are you (?:so |being )?(?:\w+ )?(?:{_FEELINGS}|ignoring|avoiding|overreacting)\b",
    rf"\bwhat(?:'s| is) making you\b",
    rf"\bwhat are you so\b",
    rf"\bdon't be (?:so )?(?:{_FEELINGS})\b",
    rf"\bno reason to be\b",
    r"\byou(?:'re| are) (?:ignoring|avoiding|overreacting|being difficult)\b",
    r"\bthey(?:'re| are) being\b",
    r"\byou don't (?:care|like)\b",
    r"\bwhy don't you like\b",
    rf"\byou seem (?:\w+ )?(?:{_FEELINGS})\b",
]
_HEDGE_CONFIRM = re.compile(
    r"\b(?:right|wrong|misreading|reading that|close|fair|off)\b[^.!?]*\?|\bmight be wrong\b|\bcould be (?:totally )?wrong\b|"
    r"\bnot sure if\b|\bi might be\b|\beverything (?:okay|ok|alright)\?|\bwant to talk\?", re.I)
_NEGATED = re.compile(r"\b(?:not saying|not assuming|don't want to guess|don't know how you)\b", re.I)

_JUDGE_PATTERNS = [
    r"\bweird\b", r"\bsilly\b", r"\blazy\b", r"\bdramatic\b", r"\boverreacting\b", r"\bwrong with you\b",
    r"\bgrow up\b", r"\bget over it\b", r"\bwasting time\b", r"\bbeing difficult\b", r"\bnot a big deal\b",
    r"\bit's nothing\b", r"\bcheer up\b", r"\bsmile more\b", r"^\s*relax\b|,\s*relax\b", r"\beveryone else\b",
    r"\bdon't worry\b", r"\bdon't be (?:so )?\w+", r"\bno reason to be\b", r"^\s*(?:wow|oh sure|oh great)\b",
]
_PRESSURE_PATTERNS = [
    r"^\s*(?:please |kindly )?(?:tell me|explain|answer|admit|say something|look at me|stop|put)\b",
    r"(?<=[.!?])\s*(?:please |kindly )?(?:tell me|explain|answer|admit|say something|look at me|stop|put)\b",
    r"\byou (?:need|have) to\b", r"\bjust (?:tell|answer|look)\b", r"\bright now\b", r"\balready\?",
    r"\bno excuses\b", r"\buntil you\b", r"\badmit it\b", r"\banswer me\b", r"\bpay attention\b",
    r"\bexplain yourself\b", r"^\s*relax\b|,\s*relax\b",
]
_PRESSURE_EXEMPT = re.compile(r"\btell me if\b|\blet me know if\b|\bif you (?:want|'d like|would like)\b", re.I)
_OFFER = re.compile(
    r"\?|\bi'm (?:here|around)\b|\bif you (?:want|'d like|would like)\b|\blet me know\b|\bhappy to help\b|\bchecking in\b", re.I)
_PRESUPPOSING_WHY = re.compile(r"\bwhy (?:are|do|don't) you\b|\bwhat(?:'s| is) making you\b|\bwhat are you so\b", re.I)


def _any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(0).strip()
    return None


class RuleReviewer:
    name = "rules-v1"

    def review(self, message: str) -> Review:
        text = clean_message(message)
        low = text.lower().replace("’", "'")
        explanations = {}

        hit = _any(_ASSUME_PATTERNS, low)
        hedged = hit is not None and re.search(r"\bseem|\bmight\b|\bmaybe\b|\bguessing\b", low) and _HEDGE_CONFIRM.search(low)
        assumes = bool(hit) and not hedged and not _NEGATED.search(low)
        if assumes:
            explanations["assumes_feeling"] = f'"{hit}" states their inner state as a fact.'

        j = _any(_JUDGE_PATTERNS, low)
        if j:
            explanations["judgmental"] = f'"{j}" can come across as dismissive or judging.'

        p = None if _PRESSURE_EXEMPT.search(low) and not re.search(r"\banswer me\b", low) else _any(_PRESSURE_PATTERNS, low)
        if p:
            explanations["pressuring"] = f'"{p}" leaves little room to decline.'

        presupposing = bool(_PRESUPPOSING_WHY.search(low))
        checks_in = bool(_OFFER.search(low)) and not (p or j or presupposing)

        labels = {"assumes_feeling": assumes, "judgmental": bool(j), "pressuring": bool(p), "checks_in": checks_in}
        return Review(labels, explanations, _template_suggestion(labels), self.name)


def _template_suggestion(labels: dict) -> str:
    if labels["checks_in"] and not (labels["assumes_feeling"] or labels["judgmental"] or labels["pressuring"]):
        return "It asks instead of deciding for them, and they can say no or answer later."
    return ("Try: describe what you noticed, then offer a choice. For example: \"I noticed ___. "
            "Would you like ___, or ___? Either is okay.\"")


# --------------------------------------------------------------------------- LLM reviewer

SYSTEM_PROMPT = """You review a short message that a learner plans to say to another person after noticing a social cue.
Judge only the wording of the message. Never guess the learner's or the other person's real feelings.

Label four independent criteria (true/false):
- assumes_feeling: states or presupposes the other person's inner state, motive or intent as fact.
  "You look sad." / "Why are you so angry?" / "You're ignoring me." are true.
  A hedged observation followed by a confirmation question ("You seem quiet. Am I reading that right?") is false.
  Describing observable behavior ("You covered your ears") is false. Saying you are NOT assuming is false.
- judgmental: labels, blames, mocks, dismisses or minimises the person or their reaction. Sarcasm is judgmental.
  Dismissive reassurance ("Don't be sad, it's fine") is judgmental.
- pressuring: demands disclosure or action, removes the option to decline, or issues an ultimatum.
  Polite commands ("Please just answer the question") are pressuring. "Tell me if I'm wrong" is not.
- checks_in: genuinely asks, or offers support/options the person can accept or decline. A presupposing "why are you..."
  question, or a question that is really a demand, is not a check-in. A message that pressures is not a check-in.

Reply with ONLY a JSON object, no prose, in exactly this shape:
{"assumes_feeling": bool, "judgmental": bool, "pressuring": bool, "checks_in": bool,
 "explanations": {"<criterion that is a problem>": "<one short sentence quoting the words>"},
 "suggestion": "<one short rewrite that fixes every problem, or a one-line note on why it already works>"}"""


class LLMError(RuntimeError):
    pass


class LLMReviewer:
    def __init__(self, provider: str, api_key: str, model: str | None = None, base_url: str | None = None,
                 timeout: float = 20.0, fallback: RuleReviewer | None = None):
        if provider not in {"anthropic", "openai"}:
            raise ValueError("provider must be 'anthropic' or 'openai'")
        self.provider = provider
        self.api_key = api_key
        self.model = model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini")
        self.base_url = (base_url or ("https://api.anthropic.com" if provider == "anthropic"
                                      else "https://api.openai.com")).rstrip("/")
        self.timeout = timeout
        self.fallback = fallback or RuleReviewer()
        self.name = f"llm:{self.provider}:{self.model}"

    # Split out so tests and the evaluation cache can bypass the network.
    def complete(self, message: str) -> str:
        user = f"Message to review:\n<message>{message}</message>"
        if self.provider == "anthropic":
            url = f"{self.base_url}/v1/messages"
            body = {"model": self.model, "max_tokens": 400, "temperature": 0, "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user}]}
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        else:
            url = f"{self.base_url}/v1/chat/completions"
            body = {"model": self.model, "max_tokens": 400, "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
            headers = {"authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"request failed: {exc}") from exc
        try:
            if self.provider == "anthropic":
                return "".join(block.get("text", "") for block in data["content"])
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected API response shape") from exc

    def review(self, message: str, raw: str | None = None) -> Review:
        text = clean_message(message)
        try:
            raw = raw if raw is not None else self.complete(text)
            return parse_llm_output(raw, self.name)
        except LLMError as exc:
            fallback = self.fallback.review(text)
            return Review(fallback.labels, fallback.explanations, fallback.suggestion,
                          f"{self.fallback.name} (fallback)", fallback_reason=str(exc))


def parse_llm_output(raw: str, backend: str) -> Review:
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        raise LLMError("model did not return JSON")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError("model returned invalid JSON") from exc
    if not all(isinstance(obj.get(k), bool) for k in CRITERIA):
        raise LLMError("model output is missing boolean criteria")
    explanations = obj.get("explanations") if isinstance(obj.get("explanations"), dict) else {}
    explanations = {k: str(v)[:300] for k, v in explanations.items() if k in CRITERIA}
    suggestion = str(obj.get("suggestion") or "")[:400] or _template_suggestion(obj)
    return Review({k: obj[k] for k in CRITERIA}, explanations, suggestion, backend)


def reviewer_from_env(env: dict | None = None):
    """Rules by default. The LLM is opt-in because it sends the message text to a third party."""
    env = os.environ if env is None else env
    provider = (env.get("MOSAIC_LLM_PROVIDER") or "").strip().lower()
    if not provider:
        return RuleReviewer()
    key = env.get("MOSAIC_LLM_API_KEY") or env.get(
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
    if not key:
        return RuleReviewer()
    return LLMReviewer(provider, key, model=env.get("MOSAIC_LLM_MODEL") or None,
                       base_url=env.get("MOSAIC_LLM_BASE_URL") or None)
