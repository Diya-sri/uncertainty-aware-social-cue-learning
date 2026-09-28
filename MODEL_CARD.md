# MOSAIC expression-cue model card

## Intended use

An educational prototype for practising interpretation of visible expression cues. It may
offer possibilities or abstain. It must not diagnose, grade, or claim a person's internal
emotion.

## Model

ImageNet EfficientNetB0 pretrained on 33,977 deduplicated FER-2013 images, adapted to the
clean target training split, and further fine-tuned across three target seeds. Candidate,
weight-soup, and flip-TTA choices were made using validation macro-F1. Seven outputs.

## Evaluation

- Target benchmark: 298 images
- Accuracy: 62.1%
- Macro-F1: 0.576
- Previous MobileNetV2 ensemble accuracy: 49.7%
- Absolute improvement over the original shipped model: 12.4 percentage points
- General-source accuracy: 56.0%
- Kids-source accuracy: 75.8%
- Validation ECE after temperature scaling: 0.0475
- Fixed selective result: 69.7% accuracy at 73.2% coverage (threshold 0.425)

The threshold and temperature were chosen on validation predictions only. The 298-image
benchmark has now been inspected across model versions, so it is a development benchmark;
a new external set is required for the next final unbiased claim.

## Limitations

Fear, anger and neutral are weak. Data is not balanced or evaluated by demographic subgroup.
Facial expressions are not ground truth for feelings. Haar detection is sensitive to pose and
lighting. Do not use for medical, employment, education-placement, policing, or surveillance decisions.
