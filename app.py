import os

import gradio as gr
import numpy as np

from code.inference import CNNPredictor


CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/cnn_best.pth")
ENCODER_NAME = os.getenv("MODEL_ENCODER_NAME", "efficientnet-b2")
INTENSITY_NORM = os.getenv("MODEL_INTENSITY_NORM", "zscore")
DEVICE = os.getenv("MODEL_DEVICE", "auto")

try:
    predictor = CNNPredictor(
        checkpoint=CHECKPOINT,
        encoder_name=ENCODER_NAME,
        intensity_norm=INTENSITY_NORM,
        device=DEVICE,
    )
    startup_message = f"Model ready: `{CHECKPOINT}` | device: `{predictor.device}`"
except Exception as exc:
    predictor = None
    startup_message = f"Model unavailable: `{exc}`"


def make_overlay(image, mask):
    source = np.asarray(image)
    if source.ndim == 3:
        source = source[..., :3].mean(axis=2)
    source = source.astype(np.float32)
    low, high = np.percentile(source, (1, 99))
    source = np.clip((source - low) / max(high - low, 1e-6), 0, 1)
    base = np.repeat((source[..., None] * 255).astype(np.uint8), 3, axis=2)
    overlay = base.astype(np.float32)
    overlay[mask > 0] = 0.45 * overlay[mask > 0] + 0.55 * np.array([225, 95, 75])
    return overlay.astype(np.uint8)


def predict(image, threshold):
    if image is None:
        raise gr.Error("Upload an OCT B-scan to begin.")
    if predictor is None:
        raise gr.Error(startup_message)
    try:
        probability_map, mask = predictor.predict(image, threshold=threshold)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    overlay = make_overlay(image, mask)
    mask_image = (mask * 255).astype(np.uint8)
    foreground = int(mask.sum())
    coverage = 100 * foreground / mask.size
    status = f"Foreground pixels: `{foreground:,}`  |  Estimated coverage: `{coverage:.2f}%`  |  Threshold: `{threshold:.2f}`"
    return mask_image, overlay, status


CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
.hero { padding: 18px 0 8px; }
.hero h1 { letter-spacing: -0.02em; margin-bottom: 8px; }
.panel { border: 1px solid #d0d5dd; border-radius: 10px; padding: 16px; }
"""

with gr.Blocks(title="SemiSAM OCT Segmentation", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div class="hero">
        <h1>SemiSAM OCT Segmentation</h1>
        <p>Research demonstration of the deployable CNN branch for intraretinal cyst segmentation.</p>
        </div>
        """
    )
    gr.Markdown(startup_message)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            image = gr.Image(label="OCT B-scan", type="numpy", image_mode="L")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Segmentation threshold")
            run = gr.Button("Segment OCT scan", variant="primary")
        with gr.Column(scale=1, elem_classes="panel"):
            mask = gr.Image(label="Predicted cyst mask", type="numpy")
            overlay = gr.Image(label="Prediction overlay", type="numpy")
    status = gr.Markdown("Upload an image and run segmentation.")
    run.click(predict, inputs=[image, threshold], outputs=[mask, overlay, status])


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "10000")),
        show_api=False,
    )
