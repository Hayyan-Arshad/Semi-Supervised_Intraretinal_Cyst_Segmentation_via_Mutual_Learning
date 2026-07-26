FROM python:3.11-slim

WORKDIR /app

COPY requirements-host.txt .
RUN pip install --no-cache-dir -r requirements-host.txt

COPY . .

ENV MODEL_CHECKPOINT=/app/checkpoints/cnn_best.pth
ENV MODEL_DEVICE=cpu
ENV PORT=10000

EXPOSE 10000
CMD ["python", "app.py"]
