# ==============================================================================
# Universal Multi-Modal Agentic RAG System - Production Dockerfile
# ==============================================================================
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (C/C++ toolchain, OpenGL, glib for image/PDF handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy package definitions and source tree
COPY pyproject.toml /app/
COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data

# Install dependencies into system Python
RUN uv pip install --system -e .

# Prepare persistent data mount points with full write permissions (for Hugging Face Spaces user 1000)
RUN mkdir -p /app/data/raw \
             /app/data/rendered_pages \
             /app/data/chunks \
             /app/data/figures \
             /app/data/tables \
             /app/data/embeddings \
             /app/data/retrieval_runs \
    && chmod -R 777 /app/data

ENV PYTHONPATH="/app/backend:/app/backend/src:/app:$PYTHONPATH"

# Expose ports for both Hugging Face Spaces (7860) and Render/Local (8000)
EXPOSE 7860
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
