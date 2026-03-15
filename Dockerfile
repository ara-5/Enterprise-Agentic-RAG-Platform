# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data vectorstore/index evaluation

# Expose FastAPI port
EXPOSE 8000

# Health check (used by cloud run, kubernetes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Start FastAPI
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
