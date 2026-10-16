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


RUN uv sync --no-dev

CMD ["uv", "run", "--no-project", "uvicorn", "src/handspell/model.py"]

# Google Cloud Run will override the PORT environment variable if specified in the service configuration.
CMD ["uv", "run", "--no-project", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
