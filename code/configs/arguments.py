import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainer", type=str, default="semisam")
    parser.add_argument("--dataset", type=str, default="oct_h5")
    parser.add_argument("--root_path", type=str, default="data/OCT_IRF")
    parser.add_argument("--intensity_norm", type=str, default="zscore", choices=["none", "minmax", "zscore", "clip_zscore"])
    parser.add_argument("--exp", type=str, default="SemiSAM_EffiB2")
    parser.add_argument("--snapshot_path", type=str, default="model")

    parser.add_argument("--cnn_model", type=str, default="efficient_unet")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b2")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)

    parser.add_argument("--prompt_model", type=str, default="sam")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--sam_freeze_image_encoder", action="store_true")
    parser.add_argument("--prompt_generator", type=str, default="mask_box_point")
    parser.add_argument("--prompt_threshold", type=float, default=0.5)
    parser.add_argument("--prompt_margin", type=int, default=4)

    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--labeled_bs", type=int, default=6)
    parser.add_argument("--labeled_num", type=int, default=85)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--prompt_lr", type=float, default=1e-5)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--consistency", type=float, default=1.0)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=3000)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args(argv)


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
