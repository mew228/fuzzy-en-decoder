# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for TF CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Railway injects PORT env variable
ENV PORT=8080
EXPOSE 8080

# Single worker (TF is memory-heavy), 120s timeout for training requests
CMD gunicorn main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --graceful-timeout 30
