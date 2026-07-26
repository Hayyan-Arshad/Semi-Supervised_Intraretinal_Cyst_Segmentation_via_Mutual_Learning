import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="YAML experiment configuration file")

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
    parser.add_argument("--sam_checkpoint", type=str)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--sam_freeze_image_encoder", dest="sam_freeze_image_encoder", action="store_true")
    freeze_group.add_argument("--no_sam_freeze_image_encoder", dest="sam_freeze_image_encoder", action="store_false")
    parser.set_defaults(sam_freeze_image_encoder=False)
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

    return parser


def load_config(path, parser):
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError as exc:
        parser.error(f"could not read config file '{config_path}': {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"invalid YAML in config file '{config_path}': {exc}")

    if not isinstance(config, dict):
        parser.error(f"config file '{config_path}' must contain a top-level mapping")

    valid_keys = {action.dest for action in parser._actions if action.dest not in {"help", "config"}}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        parser.error(f"unknown config option(s): {', '.join(unknown_keys)}")

    actions = {action.dest: action for action in parser._actions if action.dest in valid_keys}
    normalized = {}
    for key, value in config.items():
        action = actions[key]
        try:
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    raise ValueError("must be true or false")
            elif action.nargs is not None:
                if not isinstance(value, (list, tuple)):
                    raise ValueError("must be a YAML list")
                if isinstance(action.nargs, int) and len(value) != action.nargs:
                    raise ValueError(f"must contain exactly {action.nargs} values")
                value = [action.type(item) if action.type else item for item in value]
            elif action.type is not None and value is not None:
                value = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid value for '{key}' in '{config_path}': {exc}")

        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            parser.error(f"invalid value for '{key}' in '{config_path}': choose from {choices}")
        normalized[key] = value

    return normalized


def parse_args(argv=None):
    parser = build_parser()
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config is not None:
        parser.set_defaults(**load_config(config_args.config, parser))

    args = parser.parse_args(argv)
    if not args.sam_checkpoint:
        parser.error("--sam_checkpoint must be provided on the command line or in the YAML config")
    if args.labeled_bs > args.batch_size:
        parser.error("--labeled_bs cannot exceed --batch_size")
    return args


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
