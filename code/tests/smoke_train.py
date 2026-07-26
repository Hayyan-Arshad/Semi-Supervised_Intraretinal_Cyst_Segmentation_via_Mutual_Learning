from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch.nn as nn

from code.configs import parse_args, set_reproducibility
from code.trainers.semisam_trainer import SemiSAMTrainer


def make_tiny_h5_dataset():
    root = Path(tempfile.mkdtemp(prefix="semisam_smoke_", dir="/private/tmp"))
    (root / "data" / "slices").mkdir(parents=True)
    (root / "data").mkdir(exist_ok=True)
    train_names = [f"slice_{i}" for i in range(4)]
    (root / "train_slices.list").write_text("\n".join(train_names) + "\n")
    (root / "val.list").write_text("val_case\n")

    for i, name in enumerate(train_names):
        image = np.random.rand(32, 32).astype(np.float32)
        label = np.zeros((32, 32), dtype=np.uint8)
        label[8:16, 8:16] = 1 if i < 2 else 0
        with h5py.File(root / "data" / "slices" / f"{name}.h5", "w") as handle:
            handle["image"] = image
            handle["label"] = label

    with h5py.File(root / "data" / "val_case.h5", "w") as handle:
        handle["image"] = np.random.rand(1, 32, 32).astype(np.float32)
        handle["label"] = np.zeros((1, 32, 32), dtype=np.uint8)
    return root


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TinyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, images, boxes, points, point_labels):
        return self.conv(images)


def main():
    root = make_tiny_h5_dataset()
    args = parse_args(
        [
            "--config",
            "code/configs/semisam_oct.yaml",
            "--root_path",
            str(root),
            "--sam_checkpoint",
            "dummy.pth",
            "--batch_size",
            "4",
            "--labeled_bs",
            "2",
            "--labeled_num",
            "2",
            "--patch_size",
            "32",
            "32",
            "--max_iterations",
            "1",
            "--snapshot_path",
            str(root / "model"),
            "--exp",
            "smoke",
            "--num_workers",
            "0",
        ]
    )
    set_reproducibility(args.seed)
    trainer = SemiSAMTrainer(
        args,
        cnn_builder=lambda _args, device: TinyCNN().to(device),
        prompt_model_builder=lambda _args, device: TinyPromptModel().to(device),
    )
    trainer.run()
    print(f"SMOKE_OK {root}")


if __name__ == "__main__":
    main()
