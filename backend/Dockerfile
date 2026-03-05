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

# Railway injects PORT env variable at runtime
ENV PORT=8080
EXPOSE 8080

# MUST use shell form (not exec form) so $PORT expands correctly
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
