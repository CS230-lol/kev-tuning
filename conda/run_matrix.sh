#!/usr/bin/env bash
# One worker per GPU; jobs balanced so each GPU gets one 4b job (+ one 0.8b job). Resumable: rerun to continue.
set -uo pipefail
source ~/kev-env/bin/activate && cd ~/conda && mkdir -p logs
T=CS230-lol; B=jaredpalmer
JOBS=("$T/kev-4b-jigsaw-toxicity:ctx $T/kev-0.8b-jigsaw-toxicity:ctx"
      "$T/kev-4b-jigsaw-toxicity:noctx $T/kev-0.8b-jigsaw-toxicity:noctx"
      "$B/kev-4b:ctx $B/kev-0.8b:ctx"
      "$B/kev-4b:noctx $B/kev-0.8b:noctx")
N=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((N - 1))); do
  args=""; for i in $(seq $g $N 3); do for j in ${JOBS[$i]}; do args+=" --job $j"; done; done
  CUDA_VISIBLE_DEVICES=$g python kev_conda_eval.py run $args > logs/gpu$g.log 2>&1 &
done
wait
python kev_conda_eval.py matrix > logs/matrix.log 2>&1 && echo ALL_DONE > logs/status || echo FAILED > logs/status
