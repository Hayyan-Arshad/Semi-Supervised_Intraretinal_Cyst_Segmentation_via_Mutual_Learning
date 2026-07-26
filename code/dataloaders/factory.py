from .oct_h5 import OCTH5Dataset, RandomGenerator


def build_datasets(args):
    if args.dataset != "oct_h5":
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Available: oct_h5")
    train_set = OCTH5Dataset(
        args.root_path,
        split="train",
        transform=RandomGenerator(args.patch_size),
        intensity_norm=args.intensity_norm,
    )
    val_set = OCTH5Dataset(args.root_path, split="val", intensity_norm=args.intensity_norm)
    return train_set, val_set
