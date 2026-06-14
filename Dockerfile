FROM python:3.12-slim

# ffmpeg: hardsub + audio extraction
# tini: PID 1 init so killed worker process groups don't leave zombies
# ca-certificates/curl/unzip: fetch the Deno binary below
# (ctranslate2/faster-whisper/sentencepiece ship prebuilt wheels -- no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tini \
    ca-certificates \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno: JS runtime yt-dlp uses to solve YouTube's JS challenges (the only
# runtime enabled by default; auto-detected once it's on PATH).
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno

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
