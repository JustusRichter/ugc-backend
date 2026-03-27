FROM python:3.11-slim

# v4 — explicit COPY per file to bypass Railway layer cache
# System dependencies: ffmpeg (required by yt-dlp + whisper) + curl for healthcheck
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model so first request is fast
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Copy backend
COPY main.py .

# Copy frontend explicitly (own layer — always refreshed)
COPY frontend/ ./frontend/

# Copy config files
COPY railway.json .

# Expose default port (Railway injects $PORT at runtime)
EXPOSE 8000

# Start server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
