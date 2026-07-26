# Training

## Quick Smoke Test

```bash
python -m code.tests.test_normalization
python -m code.tests.smoke_train
```

The smoke trainer uses synthetic HDF5 data and tiny mock models. It does not require real OCT data or SAM checkpoints.

## Real Training Command

Example:

```bash
python train.py \
  --trainer semisam \
  --dataset oct_h5 \
  --root_path data/OCT_IRF \
  --intensity_norm zscore \
  --cnn_model efficient_unet \
  --encoder_name efficientnet-b2 \
  --encoder_weights imagenet \
  --prompt_model sam \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --sam_freeze_image_encoder \
  --batch_size 9 \
  --labeled_bs 6 \
  --labeled_num 85 \
  --patch_size 512 512 \
  --max_iterations 30000 \
  --warmup_iterations 1000 \
  --consistency 1.0 \
  --consistency_rampup 200 \
  --num_workers 4
```

## CPU Sanity Run

A one-iteration CPU run with SAM ViT-B is slow but useful for verifying the real code path:

```bash
python train.py \
  --root_path data/DUKE_TINY \
  --intensity_norm zscore \
  --sam_checkpoint /Users/hayyan/SSL4MIS/code/pretrained_ckpt/sam_vit_b_01ec64.pth \
  --encoder_weights None \
  --batch_size 2 \
  --labeled_bs 1 \
  --labeled_num 2 \
  --patch_size 64 64 \
  --max_iterations 1 \
  --num_workers 0 \
  --val_every 999999 \
  --save_every 999999 \
  --sam_freeze_image_encoder \
  --snapshot_path model \
  --exp duke_tiny_real_cpu_norm
```

This was used only to verify execution. It is not a valid experiment split.

## Outputs

Training outputs are written under:

```text
model/<exp>/
```

Typical files:

- `log.txt`
- TensorBoard event files.
- `cnn_iter_<n>.pth`
- `prompt_iter_<n>.pth`
- `cnn_best.pth`

The `model/` folder is ignored by Git.

