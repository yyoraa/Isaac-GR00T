#!/usr/bin/env bash
set -euo pipefail

quest_root="${ROBOTTT_QUEST_ROOT:-/gpfs/home/qge0476/project/yiyun/robottt-quest}"
repo_dir="${ROBOTTT_REPO:-${quest_root}/Isaac-GR00T}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${quest_root}/uv-cache}"
export UV_LINK_MODE=copy

cd "${repo_dir}"
/home/qge0476/.local/bin/uv sync --frozen

test -x .venv/bin/python
test -x .venv/bin/torchrun
.venv/bin/python -c 'import torch; print(torch.__version__)'
