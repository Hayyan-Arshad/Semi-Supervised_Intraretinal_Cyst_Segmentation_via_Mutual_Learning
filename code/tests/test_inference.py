from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from code.inference import CNNPredictor
from code.networks.efficient_unet import build_efficient_unet


def main():
    root = Path(tempfile.mkdtemp(prefix="semisam_inference_", dir="/private/tmp"))
    checkpoint = root / "cnn.pth"
    args = SimpleNamespace(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        out_channels=1,
    )
    model = build_efficient_unet(args, torch.device("cpu"))
    torch.save(model.state_dict(), checkpoint)

    predictor = CNNPredictor(
        checkpoint=checkpoint,
        encoder_name="efficientnet-b0",
        patch_size=(32, 32),
        device="cpu",
    )
    probability_map, mask = predictor.predict(np.random.rand(24, 40).astype(np.float32))
    assert probability_map.shape == (24, 40)
    assert mask.shape == (24, 40)
    assert mask.dtype == np.uint8
    assert np.isfinite(probability_map).all()
    print("INFERENCE_OK")


if __name__ == "__main__":
    main()
