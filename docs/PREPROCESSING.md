# Preprocessing

## Implemented

Preprocessing currently happens in the dataloader:

Code:

- `code/dataloaders/oct_h5.py`

Implemented image intensity options:

```bash
--intensity_norm none
--intensity_norm minmax
--intensity_norm zscore
--intensity_norm clip_zscore
```

Default:

```bash
--intensity_norm zscore
```

Labels are not intensity-normalized.

## Normalization Modes

### none

Converts image to `float32` and leaves values unchanged.

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

Applies z-score normalization, clips to `[-5, 5]`, then scales to approximately `[-1, 1]`.

## Augmentation

Training augmentation currently includes:

- Random rotation and flip.
- Random rotation from approximately `-20` to `+20` degrees.
- Resize to `--patch_size`.

Validation/test samples are not augmented.

## Important Caveats

The current transform uses nearest-neighbor resizing for both image and label. That is safe for labels but not ideal for OCT images. A future improvement should use:

- bilinear interpolation for images.
- nearest-neighbor interpolation for labels.

The repo now supports normalization, but it does not yet include a full raw-dataset conversion pipeline. Patient-wise preprocessing should be implemented in a separate preparation script.

