FROM python:3.12-slim
#-trixie

# mediapipe's native library links against EGL and OpenGL ES (for its GPU
# delegate) even when running on CPU. python:3.12-slim doesn't include them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY api/ ./api/
COPY models/hand_landmarker.task ./models/hand_landmarker_model.task
COPY models/handspell_xgboost_megadata.joblib ./models/asl_model.joblib
COPY src/ ./src/
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY "data/mapping" "data/mapping"

ENV UV_NO_DEV=1 \
UV_FROZEN=1
ENV HAND_LANDMARKER_PATH=models/hand_landmarker_model.task \
ASL_MODEL_LANDMARKS_PATH=models/asl_model.joblib
# Reduce MediaPipe logging to only errors
ENV GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2

RUN uv sync

# Google Cloud Run will override the PORT environment variable if specified in the service configuration.
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
