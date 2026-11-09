#!/usr/bin/env bash
# One-time setup on the EC2 GPU box (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu). Mirrors kev_modal.py's image.
set -euo pipefail
KEV_REF=fc4a17e194bca35a48c57211392ec49ea54a94ac  # keep in sync with SETTINGS["KEV_REF"] in kev_modal.py

nvidia-smi --query-gpu=name,memory.total --format=csv
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

[ -d /kev ] || { sudo git clone https://github.com/jaredpalmer/kev.git /kev; sudo chown -R "$USER" /kev; }
git -C /kev checkout --quiet "$KEV_REF"

uv venv --python 3.13 ~/kev-env
source ~/kev-env/bin/activate
uv pip install "kev[serve] @ file:///kev" modal
# separate install on purpose (as in the Modal image): fla's kernels need a newer triton than torch pins
uv pip install "flash-linear-attention==0.5.2" "triton==3.7.1"  # 3.9 breaks fla 0.5.2 import (autotune key BT) -> silent slow fallback
python -c "import torch, kev; print('cuda ok:', torch.cuda.get_device_name(0))"
echo "done. next: source ~/kev-env/bin/activate"
