# Training

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a SAM or MedSAM ViT-B checkpoint separately and provide its path with `--sam_checkpoint`.

## Training Command

```bash
python train.py \
  --config code/configs/semisam_oct.yaml
```

The YAML file defines the complete reference experiment. Explicit command-line arguments take precedence, which makes controlled variants concise:

```bash
python train.py \
  --config code/configs/semisam_oct.yaml \
  --root_path data/OCT_IRF \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --labeled_num 40 \
  --exp SemiSAM_40L
```

An experiment can also be specified entirely through command-line arguments:

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

## Key Parameters

- `config`: YAML experiment configuration. Command-line values override matching YAML values.
- `trainer`: training algorithm. Default: `semisam`.
- `dataset`: dataset loader. Default: `oct_h5`.
- `intensity_norm`: image normalization mode.
- `cnn_model`: CNN segmentation branch.
- `encoder_name`: CNN encoder backbone.
- `prompt_model`: SAM/MedSAM promptable branch.
- `labeled_num`: number of labeled training entries.
- `labeled_bs`: number of labeled samples per batch.
- `warmup_iterations`: iterations before consistency regularization begins.
- `consistency`: maximum consistency-loss weight.

## Verification

Lightweight checks are provided for development:

```bash
python -m code.tests.test_config
python -m code.tests.test_normalization
python -m code.tests.smoke_train
```

These checks validate YAML loading and overrides, preprocessing, and trainer wiring without requiring protected datasets.

## Outputs

Training outputs are written under:

```text
model/<exp>/
```

Typical outputs include:

- `log.txt`
- TensorBoard event files.
- `cnn_iter_<n>.pth`
- `prompt_iter_<n>.pth`
- `cnn_best.pth`

The `model/` directory is ignored by Git.
