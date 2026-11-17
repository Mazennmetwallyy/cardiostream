# --- 1. Build the React dashboard -------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- 2. Serving image: Python + ONNX Runtime, no PyTorch --------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CARDIO_ARTIFACTS=/app/artifacts CARDIO_STATIC=/app/static PORT=8000
WORKDIR /app

COPY pyproject.toml README.md ./
COPY cardiostream/ cardiostream/
RUN pip install --no-cache-dir .

COPY artifacts/model.onnx artifacts/metrics.json artifacts/
COPY artifacts/records/ artifacts/records/
COPY --from=web /web/dist static/

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn cardiostream.api:app --host 0.0.0.0 --port ${PORT}"]
