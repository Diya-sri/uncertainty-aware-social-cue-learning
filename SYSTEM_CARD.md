# MOSAIC-X system card

## Purpose

MOSAIC-X is a portfolio-grade social-cue learning prototype. It demonstrates how to
combine uncertain machine predictions with contextual evidence while preserving a
person's authority over their own reported feelings. It is not a medical, diagnostic,
surveillance, or educational-placement system.

## Architecture

1. The situation service returns a probability distribution over seven plausible expressions.
2. The vision ensemble optionally returns calibrated expression probabilities or abstains.
3. The user may explicitly report a feeling.
4. The reasoning service fuses available evidence and returns a versioned JSON object.
5. The interface displays possibilities, evidence weights, missing information, and guidance.

## Behavioral guarantees

- Explicit self-report is the primary conclusion when present.
- A face prediction marked `unsure` receives zero reasoning weight.
- Missing self-report is disclosed rather than silently imputed.
- Outputs say that expressions and situations are clues, not internal-state measurements.
- Every response exposes its evidence sources and weights.

These are implemented product rules, not learned capabilities. The automated safety suite
tests them across 126 generated cases derived from 42 versioned situations.

## Model performance

- EfficientNet V3 benchmark accuracy: 62.1% on 298 target images.
- Macro-F1: 0.576.
- Previous MobileNetV2 ensemble accuracy: 49.7%.
- Validation ECE after temperature scaling: 0.0475.
- Selective benchmark result: 69.7% accuracy at 73.2% coverage.

V3's 0.425 threshold was selected on validation only. Because earlier versions have already
been measured on the same 298 images, the benchmark is no longer described as untouched.

## Known limitations

- The face model is weak on anger, fear, and neutral expressions.
- Demographic subgroup performance has not been established.
- The reasoning weights are transparent product policy, not learned causal relationships.
- The scenario suite is synthetic and does not establish usefulness for autistic learners.
- No clinical, classroom, or human-subject effectiveness claim is supported.

## Reproduction

```bash
python scripts/evaluate_reasoning.py
python -m unittest discover -s tests -v
```

The first command regenerates the benchmark and JSON report. The second runs application,
API, reasoning, privacy, and model-contract tests when runtime dependencies are installed.
