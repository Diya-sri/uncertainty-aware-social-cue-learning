# MOSAIC dataset card

The target corpus combines a general facial-expression collection and a smaller children's
collection, both downloaded from Kaggle. Exact source pages and licenses are being confirmed and will be
linked in the README; no images are redistributed here.

The audit found 2,653 files, removed 432 exact duplicate copies, and dropped 112 duplicate
groups with contradictory labels. The remaining 1,986 images were split 1,390/298/298 with no
content hash crossing train, validation and test.

FER-2013 source: Kaggle folder-format copy ([msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)),
license shown there as "Database: Open Database, Contents: Database Contents" (ODbL 1.0 / DbCL 1.0). Original
release: ICML 2013 Workshop on Challenges in Representation Learning (Goodfellow et al., ICONIP 2013).

FER-2013 was separately audited for transfer experiments. Of 35,887 files, perceptual grouping
left 33,977 unique non-conflicting images. The supplied FER train/test folders contained 531
exact and 562 near-duplicate cross-split groups, so they were not treated as a clean benchmark.

Known gaps: no verified identity metadata, demographic metadata, consent documentation in this
repository, or subgroup evaluation. Images are intentionally not redistributed with the app.
