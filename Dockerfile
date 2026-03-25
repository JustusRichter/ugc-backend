FROM python:3.11-slim

# System dependencies: ffmpeg (required by yt-dlp + whisper) + curl for healthcheck
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose default port (Railway injects $PORT at runtime)
EXPOSE 8000

# Start server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
