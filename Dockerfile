# FinSight API — multi-stage Dockerfile
# Stage 1: builder (installs deps)
# Stage 2: runtime (lean image)

# ---- Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for transformers/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Model cache dir (mount a volume here to persist downloaded models)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
