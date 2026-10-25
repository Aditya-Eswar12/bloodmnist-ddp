# Inference service for the blood-cell classifier.
#   docker build -t bloodcell-api .
#   docker run -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" bloodcell-api
#   (add -e ANTHROPIC_API_KEY=... to let the report step use Claude; without it a template is used)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_CKPT=/app/checkpoints/baseline_cnn_s0.pt DB_PATH=/app/data/predictions.sqlite

# CPU-only PyTorch (+ torchvision, used by src/models.py) keeps the image around 1 GB instead of 5+ GB with CUDA
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY serve/ serve/
RUN mkdir -p /app/data /app/checkpoints && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
