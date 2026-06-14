#!/bin/sh
set -e

MODEL_DIR="${TRANSLATE_MODEL_DIR:-/app/models}"
MODEL_FILE="${TRANSLATE_MODEL_FILE:-Qwen2.5-7B-Instruct-Q4_K_M.gguf}"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Downloading translation model ($MODEL_FILE)..."
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='lmstudio-community/Qwen2.5-7B-Instruct-GGUF',
    filename='$MODEL_FILE',
    local_dir='$MODEL_DIR',
)
"
fi

exec "$@"
