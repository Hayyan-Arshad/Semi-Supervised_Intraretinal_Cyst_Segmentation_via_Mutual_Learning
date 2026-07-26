import os

import numpy as np
import streamlit as st
from PIL import Image

from code.inference import CNNPredictor


st.set_page_config(page_title="SemiSAM OCT Segmentation", layout="wide")
st.title("SemiSAM OCT Segmentation")
st.caption("CNN inference demo for intraretinal cyst segmentation.")


def make_overlay(image, mask):
    source = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(source, (1, 99))
    source = np.clip((source - low) / max(high - low, 1e-6), 0, 1)
    base = np.repeat((source[..., None] * 255).astype(np.uint8), 3, axis=2)
    overlay = base.astype(np.float32)
    overlay[mask > 0] = 0.45 * overlay[mask > 0] + 0.55 * np.array([225, 95, 75])
    return overlay.astype(np.uint8)


@st.cache_resource
def load_predictor():
    return CNNPredictor(
        checkpoint=os.getenv("MODEL_CHECKPOINT", "checkpoints/cnn_best.pth"),
        encoder_name=os.getenv("MODEL_ENCODER_NAME", "efficientnet-b2"),
        intensity_norm=os.getenv("MODEL_INTENSITY_NORM", "zscore"),
        device=os.getenv("MODEL_DEVICE", "auto"),
    )


uploaded = st.file_uploader("Upload an OCT B-scan", type=["png", "jpg", "jpeg", "tif", "tiff"])
threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded is not None:
    image = np.asarray(Image.open(uploaded).convert("L"), dtype=np.float32)
    if st.button("Segment OCT scan", type="primary"):
        try:
            predictor = load_predictor()
            _, mask = predictor.predict(image, threshold=threshold)
            overlay = make_overlay(image, mask)
            left, right = st.columns(2)
            with left:
                st.image(mask * 255, caption="Predicted cyst mask", clamp=True)
            with right:
                st.image(overlay, caption="Prediction overlay")
            foreground = int(mask.sum())
            coverage = 100 * foreground / mask.size
            st.success(f"Foreground pixels: {foreground:,} | Estimated coverage: {coverage:.2f}%")
        except Exception as exc:
            st.error(str(exc))
