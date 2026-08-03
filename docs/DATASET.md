# Dataset provenance and split manifest

The raw images and YOLO label files are intentionally not stored in this
repository. This document and the accompanying manifest make the dataset used
by the controlled experiments reproducible without committing the data itself.

## Source

- Dataset: [Self-Driving Cars](https://universe.roboflow.com/mark-kevin-mafzt/self-driving-cars-lfjou-nrfly/dataset/1)
- Provider: Roboflow workspace `mark-kevin-mafzt`, project
  `self-driving-cars-lfjou-nrfly`
- Export version: **1**
- Export format: YOLO detection
- License: CC BY 4.0

The local directory name `datasets/self-driving-cars-v6/` is a legacy project
path. The source export metadata confirms that the dataset actually used for
the experiments above is Roboflow Version 1. Do not substitute a different
Roboflow version when reproducing the reported results.

## Controlled split

The Version 1 export contains 4,969 images and 15 classes. The local split was
rebuilt deterministically with seed `42`, in `train:test:val` order and a
`7:2:1` ratio:

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 3,478 | 3,478 |
| val | 497 | 497 |
| test | 994 | 994 |

Every image has a same-named YOLO label file. Validation found no missing or
invalid labels.

The class order is fixed as follows:

1. Green Light
2. Red Light
3. Speed Limit 10
4. Speed Limit 100
5. Speed Limit 110
6. Speed Limit 120
7. Speed Limit 20
8. Speed Limit 30
9. Speed Limit 40
10. Speed Limit 50
11. Speed Limit 60
12. Speed Limit 70
13. Speed Limit 80
14. Speed Limit 90
15. Stop

## Manifest

[`dataset-split-manifest.csv`](dataset-split-manifest.csv) contains one row
per image in the controlled split. Its columns are:

- `split`: the final controlled split (`train`, `val`, or `test`)
- `image`: image filename
- `label`: matching YOLO label filename
- `source_split`: split assigned by the original Roboflow export

After downloading Version 1, export it in YOLO format and recreate the
controlled split with [`scripts/build_dataset_split.ps1`](../scripts/build_dataset_split.ps1)
using seed `42`. Keep the resulting class order unchanged.
