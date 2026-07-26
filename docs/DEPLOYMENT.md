# Deployment

The repository includes a browser demo for the deployable CNN branch. It accepts a single OCT B-scan and returns a binary cyst mask and an overlay. SAM is intentionally excluded from the serving path because it is used as a training-time collaborator.

## Local Host

Place the trained checkpoint at `checkpoints/cnn_best.pth`, then run:

```bash
pip install -r requirements.txt
PORT=7860 python app.py
```

Open `http://localhost:7860` in a browser. A different checkpoint can be selected with `MODEL_CHECKPOINT`:

```bash
MODEL_CHECKPOINT=/path/to/cnn_best.pth python app.py
```

The app uses the same default preprocessing as training: z-score intensity normalization, followed by resizing to the configured model input size.

## Hugging Face Spaces

Create a Gradio Space, push the repository, and add the trained CNN checkpoint as `checkpoints/cnn_best.pth` using Git LFS or the Space storage mechanism. The included `app.py`, `requirements.txt`, and `Dockerfile` provide the serving entrypoint and dependencies. The Docker image uses the smaller `requirements-host.txt` set and excludes training-only SAM dependencies.

For a private checkpoint, keep the repository public and attach the model through a private Space or a mounted storage volume. Model weights are intentionally excluded from Git.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT` | `checkpoints/cnn_best.pth` | CNN state-dict checkpoint |
| `MODEL_ENCODER_NAME` | `efficientnet-b2` | Encoder used during training |
| `MODEL_INTENSITY_NORM` | `zscore` | Input normalization mode |
| `MODEL_DEVICE` | `auto` | `cpu`, `cuda`, or automatic selection |
| `PORT` | `10000` in hosted Docker, `7860` for local use | Web server port |

The serving checkpoint must match the encoder and output-channel configuration used during training.
