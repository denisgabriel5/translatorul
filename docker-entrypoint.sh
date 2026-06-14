#!/bin/sh
set -e

MODEL_DIR="${TRANSLATE_MODEL_DIR:-/app/models}"
MODEL_FILE="${TRANSLATE_MODEL_FILE:-Qwen2.5-14B-Instruct-Q8_0.gguf}"
MODEL_REPO="${TRANSLATE_MODEL_REPO:-bartowski/Qwen2.5-14B-Instruct-GGUF}"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Downloading translation model ($MODEL_FILE from $MODEL_REPO)..."
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$MODEL_REPO',
    filename='$MODEL_FILE',
    local_dir='$MODEL_DIR',
)
"
fi

exec "$@"
