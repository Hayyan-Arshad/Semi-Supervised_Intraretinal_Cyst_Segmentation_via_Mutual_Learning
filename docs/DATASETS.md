# Datasets

This repository does not track datasets or checkpoints.

## Expected HDF5 Layout

The default loader expects SSL4MIS-style HDF5 files:

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

Validation/test volume files contain:

- `image`: 3D stack shaped like `(num_slices, height, width)`.
- `label`: 3D binary mask stack with the same shape.

## Patient-Wise Splitting

For real experiments, the split must be patient-wise, not slice-wise.

Bad:

```text
Subject_01 slices in train
Subject_01 slices in validation
```

Good:

```text
train: Subject_01 ... Subject_07
val:   Subject_08
test:  Subject_09 ... Subject_10
```

No patient should appear in more than one split.

## Dataset Status

### Duke DME / Chiu BOE 2014

Available locally outside the repo:

```text
/Users/hayyan/SSL/datasets/duke_dme/2015_BOE_Chiu/
```

The `.mat` files contain OCT images and fluid annotations. This can be converted into the expected HDF5 layout.

The current `data/DUKE_TINY` folder, if present locally, is only a tiny ignored sanity-check dataset. It is not a valid patient-wise train/validation/test split.

### Kermany OCT 2017

Available locally outside the repo:

```text
/Users/hayyan/SSL/datasets/kermany_oct/
```

This is a classification dataset. It can be useful for unlabeled SSL or pretraining, but it does not provide cyst segmentation masks.

### RETOUCH

Relevant for intraretinal fluid segmentation. Access usually requires registration or a data agreement.

### OPTIMA

Referenced by the paper, but no direct public bulk download was found during setup.

## What Still Needs To Be Added

Recommended next step:

```text
scripts/prepare_duke_h5.py
```

The script should:

- Read Duke `.mat` files.
- Split patients into train/validation/test.
- Generate `train_slices.list`, `val.list`, and `test.list`.
- Save training slices under `data/slices/`.
- Save validation/test patient volumes under `data/`.
- Binarize cyst/fluid masks.
- Avoid patient leakage.

