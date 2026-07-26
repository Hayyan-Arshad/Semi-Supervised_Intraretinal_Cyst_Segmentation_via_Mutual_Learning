# Semi-Supervised Intraretinal Cyst Segmentation via Mutual Learning

SSL4MIS-style PyTorch project for semi-supervised intraretinal cyst segmentation with a CNN branch and a promptable SAM/MedSAM branch.

The default experiment follows the provided paper:

- EfficientNet-B2 U-Net predicts intraretinal cyst masks.
- The CNN mask becomes a box + point prompt.
- SAM or MedSAM ViT-B predicts a second mask from that prompt.
- Labeled slices use Dice + BCE supervision for both branches.
- Labeled and unlabeled slices use mutual Dice consistency after warmup.
- Inference uses the CNN branch.

No datasets, checkpoints, logs, HDF5 volumes, or model weights are tracked in Git.

## Layout

```text
code/
  configs/       argparse defaults and reference experiment config
  dataloaders/   OCT HDF5 dataset, transforms, two-stream sampler, dataset factory
  networks/      CNN/SAM adapters and model factory
  prompts/       prompt-generation strategies and prompt factory
  trainers/      train-loop implementations and trainer factory
  utils/         losses, ramps, validation metrics
  tests/         lightweight smoke tests
scripts/         public dataset download helpers
train.py         thin entrypoint that builds the selected trainer
```

## Documentation

- [Architecture and algorithm](docs/ARCHITECTURE.md)
- [Methodology](docs/METHODOLOGY.md)
- [Datasets](docs/DATASETS.md)
- [Preprocessing](docs/PREPROCESSING.md)
- [Training](docs/TRAINING.md)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a SAM or MedSAM ViT-B checkpoint separately and pass it with `--sam_checkpoint`.

## Train

```bash
python train.py \
  --config code/configs/semisam_oct.yaml
```

Values supplied explicitly on the command line override the YAML configuration. For example:

```bash
python train.py \
  --config code/configs/semisam_oct.yaml \
  --root_path data/OCT_IRF \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --batch_size 6
```

The same experiment can be specified entirely through command-line options:

```bash
python train.py \
  --trainer semisam \
  --dataset oct_h5 \
  --cnn_model efficient_unet \
  --encoder_name efficientnet-b2 \
  --prompt_model sam \
  --prompt_generator mask_box_point \
  --root_path data/OCT_IRF \
  --intensity_norm zscore \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --batch_size 9 \
  --labeled_bs 6 \
  --labeled_num 85 \
  --patch_size 512 512 \
  --max_iterations 30000 \
  --num_workers 4
```

Useful switches:

- `--encoder_name`: swap the CNN encoder, for example `efficientnet-b0`, `efficientnet-b2`, or another encoder supported by `segmentation-models-pytorch`.
- `--encoder_weights`: use `imagenet` or `None`.
- `--intensity_norm`: OCT intensity normalization. Options: `none`, `minmax`, `zscore`, `clip_zscore`.
- `--prompt_model`: promptable segmentation branch. Current option: `sam`.
- `--prompt_generator`: prompt strategy. Current option: `mask_box_point`.
- `--sam_freeze_image_encoder`: freeze the SAM image encoder.
- `--warmup_iterations`: iterations before consistency loss starts.
- `--val_every`: validation interval.
- `--num_workers`: data-loading workers. Use `0` for debugging on macOS or notebooks.

The loadable reference experiment is defined in [code/configs/semisam_oct.yaml](code/configs/semisam_oct.yaml).

## Swapping Components

Add new components by following the existing factories:

- Dataset: add a loader under `code/dataloaders/`, then register it in `code/dataloaders/factory.py`.
- CNN model: add a builder under `code/networks/`, then register it in `code/networks/net_factory.py`.
- Prompt model: add an adapter under `code/networks/`, then register it in `build_prompt_segmenter`.
- Prompt strategy: add a function under `code/prompts/`, then register it in `code/prompts/factory.py`.
- Training method: add a trainer under `code/trainers/`, then register it in `code/trainers/trainer_factory.py`.

That is the main SSL4MIS-style extension point: keep scripts thin, keep modules replaceable, and switch experiments with command-line flags.

## Expected Data Layout

The default loader expects SSL4MIS-style HDF5 data:

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

Images are normalized by the dataloader before augmentation/resizing. The default is `--intensity_norm zscore`.
Labels are not intensity-normalized.

The first `--labeled_num` entries in `train_slices.list` are treated as labeled. Remaining entries are treated as unlabeled and contribute only to consistency loss.

## Verification

```bash
python -m py_compile train.py code/**/*.py
python -m code.tests.test_config
python -m code.tests.smoke_train
```

The smoke test verifies trainer wiring without requiring protected datasets or SAM weights.

## OCT Dataset Notes

Supported dataset sources:

- Duke DME / Chiu BOE 2014: publicly downloadable OCT volumes with layer/cyst annotations in `.mat` files.
- Kermany OCT 2017 mirror on Hugging Face: classification OCT images in parquet shards. Useful for unlabeled SSL or pretraining, but it does not provide cyst segmentation masks.

Datasets that need manual access:

- RETOUCH: relevant for intraretinal fluid/cyst segmentation, but access requires registration/data-agreement steps.
- OPTIMA cyst segmentation data: referenced by the paper, but a direct public bulk download link was not available from the public page scan.

## Dataset Helpers

```bash
bash scripts/download_duke_dme.sh
bash scripts/download_kermany_hf.sh
```

The helpers write into `datasets/`, which is ignored by Git.

## Attribution

This project was prepared from the SSL4MIS code structure and adapted for the semi-supervised SAM + EfficientNet-B2 U-Net intraretinal cyst workflow described in the provided paper. Respect the original licenses and dataset terms before redistribution or publication.
