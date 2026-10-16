FROM python:3.12-slim
#-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY "data/raw_data/A02_original_aslmnist/user_7/A0.jpg" "data/raw_data/A02_original_aslmnist/user_7/A0.jpg"

ENV UV_NO_DEV=1 \
UV_FROZEN=1
ENV HAND_DETECTION_MODEL_PATH=models/models/yolo26m-seg-hand.pt \
LETTER_CLASSIFICATION_MODEL_PATH=models/models/mnist_baseline_model.keras


RUN uv sync

RUN uv run src/handspell/model.py

# Google Cloud Run will override the PORT environment variable if specified in the service configuration.
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
