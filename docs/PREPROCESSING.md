# Preprocessing

Preprocessing is applied consistently through the OCT HDF5 dataloader.

Relevant module:

- `code/dataloaders/oct_h5.py`

## Intensity Normalization

The dataloader supports four image normalization modes:

```text
none
minmax
zscore
clip_zscore
```

Default:

```text
intensity_norm: zscore
```

Labels are treated as binary masks and are not intensity-normalized.

### none

Converts images to `float32` and leaves intensity values unchanged.

### minmax

Scales each loaded image or volume to `[0, 1]`:

```text
(x - min) / (max - min)
```

### zscore

Standardizes each loaded image or volume:

```text
(x - mean) / std
```

### clip_zscore

Applies z-score normalization, clips values to `[-5, 5]`, and rescales to approximately `[-1, 1]`.

## Spatial Processing

All training images and labels are resized to the configured patch size:

```text
patch_size: [512, 512]
```

Validation and test volumes are evaluated slice by slice and resized internally for model inference.

## Data Augmentation

Training-time augmentation includes:

- Random rotation and flip.
- Random in-plane rotation between approximately `-20` and `+20` degrees.

Augmentation is applied only to training samples. Validation and test data are not augmented.

## Mask Handling

Segmentation masks are represented as binary foreground/background labels. Foreground values greater than zero are treated as cyst/fluid during training.

