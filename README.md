# SemiSAM Intraretinal Cyst Segmentation

Compact implementation of the paper-style semi-supervised intraretinal cyst segmentation pipeline:

- EfficientNet-B2 U-Net predicts cyst masks.
- The EfficientNet mask is converted into a box and point prompt.
- SAM or MedSAM ViT-B predicts a second mask from that prompt.
- Labeled OCT slices use Dice + BCE supervision for both branches.
- Labeled and unlabeled slices use mutual Dice consistency after a warmup.
- Inference uses only the EfficientNet-B2 U-Net branch.

This repository intentionally does not include datasets, checkpoints, model weights, logs, or HDF5 volumes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a SAM or MedSAM ViT-B checkpoint separately and pass it with `--sam_checkpoint`.

## Expected Data Layout

The loader expects an SSL4MIS-style HDF5 structure:

```text
data/OCT_IRF/
  train_slices.list
  val.list
  data/
    slices/
      case_or_slice_name.h5
    validation_volume_name.h5
```

Each `.h5` file should contain:

- `image`: 2D OCT slice for training samples, or 3D volume for validation.
- `label`: binary intraretinal cyst mask with foreground values greater than zero.

`train_slices.list` should list training slice names without `.h5`. `val.list` should list validation volume names without `.h5`.

The first `--labeled_num` entries in `train_slices.list` are treated as labeled. Remaining entries are treated as unlabeled and contribute only to consistency loss.

## Train

```bash
python train.py \
  --root_path data/OCT_IRF \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --batch_size 9 \
  --labeled_bs 6 \
  --labeled_num 85 \
  --patch_size 512 512 \
  --max_iterations 30000 \
  --num_workers 4
```

Useful switches:

- `--sam_freeze_image_encoder`: train prompt encoder and mask decoder while freezing SAM image encoder.
- `--warmup_iterations`: number of iterations before consistency loss starts. Default: `1000`.
- `--consistency_rampup`: ramp length for consistency weight.
- `--snapshot_path`: local output directory for logs and weights. Ignored by Git.
- `--num_workers`: data-loading workers. Use `0` for debugging on macOS or notebooks.

## OCT Dataset Notes

Datasets checked/downloaded locally during project setup:

- Duke DME / Chiu BOE 2014: publicly downloadable OCT volumes with layer/cyst annotations in `.mat` files. Local copy was downloaded outside this repo.
- Kermany OCT 2017 mirror on Hugging Face: publicly downloadable classification OCT images in parquet shards. Useful for unlabeled SSL or pretraining, but it does not provide cyst segmentation masks.

Datasets that need manual access:

- RETOUCH: relevant for intraretinal fluid/cyst segmentation, but access requires challenge/registration/data-agreement steps.
- OPTIMA cyst segmentation data: referenced by the paper, but a direct public bulk download link was not available from the public page scan.

Recommended practice:

- Keep raw datasets outside Git.
- Convert approved segmentation datasets into the HDF5 layout above.
- Use classification-only OCT data only as unlabeled samples unless segmentation masks are created.

## Local Dataset Helpers

The `scripts/` folder contains download helpers for public datasets. They write into `datasets/`, which is ignored by Git.

```bash
bash scripts/download_duke_dme.sh
bash scripts/download_kermany_hf.sh
```

## Repository Contents

```text
train.py                  Semi-supervised training entrypoint
semisam/dataset.py        HDF5 OCT dataset and two-stream sampler
semisam/losses.py         Dice + BCE and consistency Dice losses
semisam/prompts.py        Mask-to-box/point prompt generation
semisam/sam_adapter.py    SAM/MedSAM promptable segmentation branch
semisam/validation.py     Dice/HD95 validation helper
requirements.txt          Python dependencies
```

## Attribution

This project was prepared from the SSL4MIS code structure and adapted for the semi-supervised SAM + EfficientNet-B2 U-Net intraretinal cyst workflow described in the provided paper. Respect the original licenses and dataset terms before redistribution or publication.
