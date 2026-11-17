# Hugging Face Space image for the full Veritas backend.
#
# The detection components that need torch are optional: this image deliberately does not
# install it, so the Space runs lite/fast and reports standard/deep unavailable with the
# reason. To enable them, build with --build-arg WITH_TORCH=1 (see docs/DEPLOY.md) and
# check the Space's disk quota first — CPU torch alone is ~800 MB.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    VERITAS_ARTIFACT_DIR=/data/artifacts \
    VERITAS_DATA_DIR=/data/state

WORKDIR /app

# libgl/libgomp are needed by some document libraries; tesseract is opt-in via OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY ml/bundled /app/ml/bundled

ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            torch==2.4.1 transformers==4.45.2 sentence-transformers==3.2.1 faiss-cpu==1.8.0; \
    fi

# Non-root, per Hugging Face's guidance for Docker Spaces.
RUN useradd -m -u 1000 space && mkdir -p /data/artifacts /data/state && chown -R space:space /data /app
USER space

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
