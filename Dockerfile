FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libxcb1 \
      libgl1 \
      libglib2.0-0  \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch for YOLO
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

# Google Cloud Run will override the PORT environment variable if specified in the service configuration.
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\""]
