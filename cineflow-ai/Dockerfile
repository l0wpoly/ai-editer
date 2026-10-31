# ==============================================================================
# Stage 1: Frontend Asset Builder (Vite + Modern ES Modules / CSS Minification)
# ==============================================================================
FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Production Multimedia Runtime (Debian Bookworm + FFmpeg + Python 3.12)
# ==============================================================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000 \
    STORAGE_DIR=/app/storage \
    STATIC_DIR=/app/static

# Install FFmpeg (includes ffprobe, libx264, aac, libavfilter/sidechaincompress),
# tini (clean zombie process reaping for FFmpeg workers), and curl (healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        tini \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ffmpeg -version && \
    ffprobe -version

# Create unprivileged runtime user and workspace directories
RUN groupadd --gid 10001 cineflow && \
    useradd --uid 10001 --gid cineflow --create-home --shell /bin/bash cineflow && \
    mkdir -p /app/storage/uploads /app/storage/workdir /app/storage/outputs /app/static

WORKDIR /app

# Install locked Python dependencies first for optimal Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY services/ ./services/
COPY main.py ./
COPY frontend/ ./frontend/

# Copy compiled & minified frontend bundle from Stage 1
COPY --from=frontend-builder /build/static ./static/

# Grant ownership to non-root user
RUN chown -R cineflow:cineflow /app

USER cineflow

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=8s --start-period=20s --retries=3 \
    CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-10000}/api/health" || exit 1'

ENTRYPOINT ["/usr/bin/tini", "--"]

# Single worker ensures SQLite + in-memory job semaphore + background asyncio tasks
# share one unified event loop without multi-process contention on Render instances.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --timeout-keep-alive 120"]
