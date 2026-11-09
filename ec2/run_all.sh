#!/usr/bin/env bash
# 0.8b then 4b; 4b runs only if 0.8b succeeded
set -eo pipefail
source ~/kev-env/bin/activate && cd ~/ec2 && mkdir -p logs
python kev_ec2.py train --data data --name jigsaw-0.8b --init-from jaredpalmer/kev-0.8b 2>&1 | tee logs/jigsaw-0.8b.log
python kev_ec2.py train --data data --name jigsaw-4b --init-from jaredpalmer/kev-4b 2>&1 | tee logs/jigsaw-4b.log
echo ALL_DONE | tee logs/status
