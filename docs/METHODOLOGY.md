# Methodology

This project implements a semi-supervised mutual-learning pipeline for intraretinal cyst/fluid segmentation in OCT B-scans.

## Objective

The goal is to train a deployable OCT cyst segmentation model using a small labeled set and a larger unlabeled set. The training stage uses two branches:

- A CNN segmentation branch based on EfficientNet-B2 U-Net.
- A promptable SAM/MedSAM branch guided by prompts generated from the CNN prediction.

At inference time, only the CNN branch is needed.

## Architecture

### CNN Branch

The default CNN is an EfficientNet-B2 U-Net implemented through `segmentation-models-pytorch`.

Code:

- `code/networks/efficient_unet.py`
- `code/networks/net_factory.py`

Default settings:

- `--cnn_model efficient_unet`
- `--encoder_name efficientnet-b2`
- `--in_channels 1`
- `--out_channels 1`

The CNN outputs one foreground logit map for binary cyst/fluid segmentation.

### SAM/MedSAM Branch

The promptable branch wraps Meta SAM or MedSAM ViT-B checkpoints. OCT slices are converted from one channel to three channels, resized to SAM input size, and passed through SAM image encoder, prompt encoder, and mask decoder.

Code:

- `code/networks/sam_adapter.py`
- `code/networks/net_factory.py`

Default settings:

- `--prompt_model sam`
- `--sam_model_type vit_b`
- `--sam_checkpoint <path-to-sam-or-medsam-checkpoint>`

The SAM image encoder can be frozen with:

```bash
--sam_freeze_image_encoder
```

## Prompt Generation

The CNN prediction is converted into SAM prompts.

Code:

- `code/prompts/mask_prompts.py`
- `code/prompts/factory.py`

Current prompt strategy:

- Threshold CNN foreground probability.
- Find the foreground bounding box.
- Add a small configurable margin.
- Sample one positive point inside the predicted foreground.
- If no foreground exists, use the full image box and a negative center point.

Default settings:

- `--prompt_generator mask_box_point`
- `--prompt_threshold 0.5`
- `--prompt_margin 4`

## Losses

Code:

- `code/utils/losses.py`
- `code/trainers/semisam_trainer.py`

### Supervised Loss

For labeled samples, both branches receive supervised segmentation loss:

```text
L_sup = 0.5 * (L_cnn_sup + L_prompt_sup)
```

Each branch uses:

```text
Dice loss + BCEWithLogits loss
```

### Mutual Consistency Loss

For labeled and unlabeled samples, the CNN and SAM/MedSAM predictions are encouraged to agree:

```text
L_mutual = 0.5 * (
  Dice(cnn_logits, stopgrad(prompt_logits)) +
  Dice(prompt_logits, stopgrad(cnn_logits))
)
```

The total loss is:

```text
L_total = L_sup + w(t) * L_mutual
```

The consistency weight is zero during warmup, then ramps up using a sigmoid ramp:

```bash
--warmup_iterations 1000
--consistency 1.0
--consistency_rampup 200
```

## Semi-Supervised Sampling

Training uses a two-stream batch sampler:

- First `--labeled_num` entries in `train_slices.list` are treated as labeled.
- Remaining training entries are treated as unlabeled.
- Each batch contains `--labeled_bs` labeled samples and the rest unlabeled samples.

Code:

- `code/dataloaders/oct_h5.py`
- `code/trainers/semisam_trainer.py`

## Inference

The intended inference path uses only the CNN branch. SAM/MedSAM is a training-time mutual-learning teacher/peer branch and is not required for deployment.

An inference script is not yet implemented. It should load `cnn_best.pth` or another CNN checkpoint and run only the EfficientNet U-Net model on preprocessed OCT slices/volumes.

## Current Implementation Status

Implemented:

- Modular SSL4MIS-style project layout.
- EfficientNet U-Net branch.
- SAM/MedSAM prompt branch.
- Mask-to-box/point prompt generation.
- Dice + BCE supervised loss.
- Mutual consistency loss.
- Consistency ramp-up.
- Two-stream labeled/unlabeled sampler.
- Configurable intensity normalization.
- Rotation/flip augmentation.
- Synthetic smoke test.
- One-iteration real training-path test with Duke OCT and SAM ViT-B checkpoint.

Still needed before serious experiments:

- Patient-wise data preparation script.
- Proper train/validation/test split files.
- Full RETOUCH/OPTIMA conversion if access is available.
- Inference/evaluation script.
- Experiment logging table for final paper-style results.

