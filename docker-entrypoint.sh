#!/bin/sh
set -e

MODEL_DIR="${TRANSLATE_MODEL_DIR:-/app/models}"
MODEL_SUBDIR="${TRANSLATE_MODEL_SUBDIR:-madlad}"
MODEL_REPO="${TRANSLATE_MODEL_REPO:-jbochi/madlad400-3b-mt}"
MODEL_PATH="$MODEL_DIR/$MODEL_SUBDIR"

if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
  echo "Downloading translation model ($MODEL_REPO)..."
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_REPO',
    local_dir='$MODEL_PATH',
    ignore_patterns=['*.safetensors', '*.h5', '*.msgpack'],
)
"
fi

exec "$@"
