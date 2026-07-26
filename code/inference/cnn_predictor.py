from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import zoom

from code.dataloaders.oct_h5 import normalize_image
from code.networks.efficient_unet import build_efficient_unet


class CNNPredictor:
    """Load the deployable CNN branch and predict one OCT B-scan at a time."""

    def __init__(
        self,
        checkpoint,
        encoder_name="efficientnet-b2",
        intensity_norm="zscore",
        patch_size=(512, 512),
        device="auto",
    ):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"CNN checkpoint not found: {self.checkpoint}")

        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        self.intensity_norm = intensity_norm
        self.patch_size = tuple(patch_size)
        args = SimpleNamespace(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,
            out_channels=1,
        )
        self.model = build_efficient_unet(args, self.device)
        state = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image, threshold=0.5):
        image = self._to_grayscale(image)
        height, width = image.shape
        normalized = normalize_image(image, self.intensity_norm)
        resized = zoom(
            normalized,
            (self.patch_size[0] / height, self.patch_size[1] / width),
            order=1,
        ).astype(np.float32)
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()

        probability_map = zoom(
            probabilities,
            (height / self.patch_size[0], width / self.patch_size[1]),
            order=1,
        )
        mask = (probability_map >= threshold).astype(np.uint8)
        return probability_map, mask

    @staticmethod
    def _to_grayscale(image):
        array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., :3].mean(axis=2)
        if array.ndim != 2:
            raise ValueError("Upload a 2D grayscale or RGB OCT B-scan.")
        return array.astype(np.float32)
