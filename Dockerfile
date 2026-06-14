FROM python:3.12-slim

# ffmpeg: hardsub + audio extraction
# nodejs: yt-dlp JS challenge bypass
# tini: PID 1 init so killed worker process groups don't leave zombies
# (ctranslate2/faster-whisper/sentencepiece ship prebuilt wheels -- no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    npm \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models /app/jobs

ENV TRANSLATE_MODEL_DIR=/app/models \
    HF_HOME=/app/models/hf_cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["tini", "--", "/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
