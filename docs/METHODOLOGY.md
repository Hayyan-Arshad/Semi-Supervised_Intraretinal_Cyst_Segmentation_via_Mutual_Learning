# Methodology

This project studies semi-supervised intraretinal cyst/fluid segmentation in OCT B-scans through mutual learning between a compact convolutional segmentation model and a promptable foundation model.

## Objective

The objective is to learn a deployable OCT segmentation model from limited labeled data while leveraging additional unlabeled scans. During training, predictions from a CNN branch and a SAM/MedSAM branch regularize one another. During inference, only the CNN branch is required, keeping deployment lightweight.

## Model Architecture

### CNN Segmentation Branch

The primary segmentation branch is an EfficientNet-B2 U-Net implemented with `segmentation-models-pytorch`. It accepts single-channel OCT B-scans and outputs a binary foreground logit map for cyst/fluid segmentation.

Relevant modules:

- `code/networks/efficient_unet.py`
- `code/networks/net_factory.py`

Default configuration:

```text
cnn_model: efficient_unet
encoder_name: efficientnet-b2
in_channels: 1
out_channels: 1
```

### Promptable SAM/MedSAM Branch

The second branch wraps SAM or MedSAM ViT-B. OCT slices are adapted to the expected SAM input format, encoded by the image encoder, and decoded using prompts derived from the CNN prediction.

Relevant modules:

- `code/networks/sam_adapter.py`
- `code/networks/net_factory.py`

Default configuration:

```text
prompt_model: sam
sam_model_type: vit_b
```

The SAM image encoder can optionally be frozen to reduce training cost and stabilize optimization.

## Prompt Generation

The CNN branch provides weak spatial guidance for the promptable branch. Its foreground probability map is thresholded, converted into a bounding box, and paired with a positive point sampled from the predicted foreground. If no foreground is detected, the full image box and a negative center point are used.

Relevant modules:

- `code/prompts/mask_prompts.py`
- `code/prompts/factory.py`

Default configuration:

```text
prompt_generator: mask_box_point
prompt_threshold: 0.5
prompt_margin: 4
```

## Training Loss

### Supervised Segmentation Loss

For labeled samples, both branches are trained with Dice loss and binary cross-entropy with logits:

```text
L_sup = 0.5 * (L_cnn_sup + L_prompt_sup)
```

### Mutual Consistency Loss

For both labeled and unlabeled samples, the two branches are encouraged to agree:

```text
L_mutual = 0.5 * (
  Dice(cnn_logits, stopgrad(prompt_logits)) +
  Dice(prompt_logits, stopgrad(cnn_logits))
)
```

The final objective is:

```text
L_total = L_sup + w(t) * L_mutual
```

where `w(t)` is a sigmoid ramp-up schedule. This delays the consistency objective until the supervised signal has begun to shape stable predictions.

Default configuration:

```text
warmup_iterations: 1000
consistency: 1.0
consistency_rampup: 200
```

## Semi-Supervised Sampling

Training batches are formed with a two-stream sampler. The first `labeled_num` samples in `train_slices.list` are treated as labeled, and remaining samples are treated as unlabeled. Each batch contains `labeled_bs` labeled examples, with the rest drawn from the unlabeled pool.

Relevant modules:

- `code/dataloaders/oct_h5.py`
- `code/trainers/semisam_trainer.py`

## Preprocessing and Augmentation

OCT images are intensity-normalized in the dataloader. Available modes are:

```text
none
minmax
zscore
clip_zscore
```

The default is z-score normalization. Training-time augmentation includes random rotation, random flip, and resizing to the configured patch size. Validation and test samples are not augmented.

## Inference Protocol

The inference protocol uses the CNN branch only. The promptable SAM/MedSAM branch is used during training as a mutual-learning signal and is not required for deployment.

This design keeps the deployed model efficient while still benefiting from promptable foundation-model supervision during training.

