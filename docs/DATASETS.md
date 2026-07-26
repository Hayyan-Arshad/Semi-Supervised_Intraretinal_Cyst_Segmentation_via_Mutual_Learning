# Datasets

This repository follows the common medical-imaging practice of keeping datasets outside version control. Data should be prepared locally in an SSL4MIS-style HDF5 layout.

## Expected HDF5 Layout

```text
data/OCT_IRF/
  train_slices.list
  val.list
  test.list
  data/
    slices/
      patient001_slice_000.h5
      patient001_slice_001.h5
    patient008.h5
    patient009.h5
```

Training slice files contain:

- `image`: 2D OCT B-scan.
- `label`: 2D binary cyst/fluid mask.

Validation and test volume files contain:

- `image`: 3D stack shaped as `(num_slices, height, width)`.
- `label`: 3D binary mask stack with the same shape.

## Patient-Wise Splitting

Experiments should use patient-wise splits to avoid leakage between training and evaluation. Slices from the same patient must not appear in more than one split.

Example split for a ten-patient dataset:

```text
train: Subject_01 ... Subject_07
val:   Subject_08
test:  Subject_09 ... Subject_10
```

The training list may contain slice-level entries, but those entries must be generated only from training patients. Validation and test lists should refer to held-out patient volumes.

## Dataset Sources

### Duke DME / Chiu BOE 2014

The Duke DME dataset provides OCT volumes with fluid-related annotations in MATLAB format. It is suitable for preparing a segmentation dataset after conversion to the HDF5 layout above.

### Kermany OCT 2017

The Kermany OCT dataset is a classification dataset. It can support unlabeled pretraining or semi-supervised experiments when used without masks, but it is not a direct cyst-segmentation benchmark.

### RETOUCH

RETOUCH is relevant for retinal fluid segmentation and is appropriate for evaluating fluid/cyst segmentation models when access is available under the dataset terms.

### OPTIMA

OPTIMA is referenced in intraretinal cyst segmentation literature and can be used when the data and annotation terms permit research use.

## Reproducibility Requirements

For each experiment, report:

- Dataset name and version.
- Patient-level train/validation/test split.
- Number of labeled and unlabeled training slices.
- Intensity normalization mode.
- Patch size.
- Annotation source used to form the binary mask.

