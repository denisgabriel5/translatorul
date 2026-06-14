#!/bin/sh
set -e

MODEL_DIR="${TRANSLATE_MODEL_DIR:-/app/models}"
MODEL_SUBDIR="${TRANSLATE_MODEL_SUBDIR:-madlad-ct2}"
MODEL_REPO="${TRANSLATE_MODEL_REPO:-santhosh/madlad400-3b-ct2}"
MODEL_PATH="$MODEL_DIR/$MODEL_SUBDIR"

# Re-download whenever the CTranslate2 weights (model.bin) are missing, so a
# partial/incompatible model self-heals on restart.
if [ ! -f "$MODEL_PATH/model.bin" ]; then
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
