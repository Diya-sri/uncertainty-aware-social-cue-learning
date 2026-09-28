# V3 upgrade notes

## Current calibrated result

V3 starts from the source-pretrained V2 checkpoint and compares three target-domain seeds,
model-weight interpolation, and flip test-time augmentation using validation macro-F1 only.
The selected seed-79 model reached 62.1% accuracy and 0.576 macro-F1 on the 298-image target
benchmark. Temperature scaling achieved 0.0475 validation ECE. The validation-selected 0.425
threshold reached 69.7% benchmark accuracy at 73.2% coverage. Photo upload is now supported
alongside the webcam. Because prior versions were inspected on the same benchmark, future final
claims require a new external test set.

# v5 historical upgrade notes

## GPU fine-tuning result

EfficientNetB0 was pretrained on 33,977 exact-deduplicated FER-2013 images and fine-tuned
on the cleaned target split. On the unchanged 298-image target test set it reached 57.1%
accuracy and 0.526 macro-F1, versus 49.7% and 0.442 for the previous shipped ensemble.
Confidence calibration remains pending; the old selective threshold is not reused.

## Legacy high-precision experiment

The previous MobileNetV2 model used a 0.75 confidence threshold. At that operating point, it answered
37 of 298 held-out examples and got 34 correct (91.9% selective accuracy, 12.4% coverage).
Its overall accuracy was 49.7%. This threshold is not transferred to the new EfficientNet model.

## Product and learning design

- Ambiguous story situations can accept more than one reasonable feeling.
- Correct alternatives explicitly teach that people can react differently.
- Every story answer now suggests a supportive next action, moving the activity from
  simple labeling toward empathy and emotional vocabulary.
- Mirror practice adapts toward expressions with more unsuccessful attempts.
- The mirror game no longer rewards a prediction the calibrated model labels “not sure”.
- The progress page recommends the next expression to practise.
- Privacy wording now accurately distinguishes local-server processing from storage or
  third-party transmission.

## Model-performance path

- `scripts/train.py` accepts `--pretrain-data` for large-source emotion pretraining before
  fine-tuning on the small, leak-free target dataset.
- Source validation remains separate and the target test set stays untouched.
- `scripts/calibrate.py` now supports both the shipped NumPy model and fine-tuned `.keras`
  models.
- Keras inference applies the saved temperature and “not sure” threshold from `labels.json`.

The shipped model now includes a validation-selected PCA/RBF-SVM ensemble head. On the untouched
298-image test set it improves accuracy from 48.0% to 49.7% and macro-F1 from 0.432 to 0.442.
The improvement is deliberately reported as modest; the next material gain still requires
fine-tuning the visual backbone on more licensed data.
