#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets/kermany_oct

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zacharielegault/Kermany2017-OCT",
    repo_type="dataset",
    local_dir="datasets/kermany_oct",
    local_dir_use_symlinks=False,
)
PY

