FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/

COPY models/kazim_handspell_mnist_custom_cnn.keras ./models/kazim_handspell_mnist_custom_cnn.keras


EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\""]
