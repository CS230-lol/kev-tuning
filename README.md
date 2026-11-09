# Kev × Jigsaw toxicity: context-aware fine-tuning and zero-shot CONDA eval

Fine-tunes [Kev](https://github.com/jaredpalmer/kev) (an open System One-style decision model) on Jigsaw Unintended
Bias comments with their **reply threads reconstructed** as context, then evaluates it zero-shot on
[CONDA](https://github.com/usydnlp/CONDA) (Dota 2 in-game chat).

| What | Where |
| --- | --- |
| Code, logs, metrics, eval matrix | this repo (~2 MB) |
| Jigsaw data: raw, filtered, threads, Kev training splits | [datasets/CS230-lol/JigsawToxicity](https://huggingface.co/datasets/CS230-lol/JigsawToxicity) |
| Fine-tuned models | [CS230-lol/kev-0.8b-jigsaw-toxicity](https://huggingface.co/CS230-lol/kev-0.8b-jigsaw-toxicity), [CS230-lol/kev-4b-jigsaw-toxicity](https://huggingface.co/CS230-lol/kev-4b-jigsaw-toxicity) |
| Raw per-question predictions, CONDA Kev records | [datasets/CS230-lol/kev-jigsaw-toxicity-eval](https://huggingface.co/datasets/CS230-lol/kev-jigsaw-toxicity-eval) |

## Results

### Jigsaw (held-out articles, 1,000 records × 7 questions)

Kev answers `toxic` (yes/no), `severe_toxicity` (2 levels) and five subtypes (obscene, sexual_explicit,
identity_attack, insult, threat; 3 levels each). Calibrated probabilities; base → fine-tuned.

| Model | Accuracy | Brier | ECE | Coverage @ 5% error | Regression check (decision-v7) |
| --- | --- | --- | --- | --- | --- |
| kev-0.8b | 0.586 → **0.814** | 0.527 → 0.267 | 0.050 → 0.010 | 0.001 → 0.487 | 0.813 → 0.813 |
| kev-4b | 0.759 → **0.838** | 0.335 → 0.231 | 0.029 → 0.008 | 0.278 → 0.624 | 0.860 → 0.860 |

Details: `runs/jigsaw-*/result.json` (bootstrap CIs, per-question metrics), `runs/jigsaw-*/model-card.md`.

### CONDA zero-shot (8,974 utterances = `CONDA_valid.csv`)

Intent asked as a new 4-way question (E explicit, I implicit, A action, O other). Without context = the utterance alone.

| Model | Context | UCA | U-F1 E | U-F1 I | U-F1 A | U-F1 O | T-F1 | JSA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always O (majority) | – | 0.739 | 0 | 0 | 0 | 0.850 | – | – |
| Joint BERT (trained on CONDA) | – | 0.924 | 0.876 | 0.751 | 0.804 | 0.956 | 0.983 | 0.894 |
| **kev-4b-jigsaw** | no | **0.853** | **0.725** | **0.531** | **0.607** | **0.921** | **0.671** | **0.443** |
| kev-4b-jigsaw | yes | 0.835 | 0.703 | 0.475 | 0.597 | 0.911 | 0.671 | 0.435 |
| kev-4b (base) | no | 0.811 | 0.629 | 0.450 | 0.582 | 0.899 | 0.614 | 0.353 |
| kev-4b (base) | yes | 0.741 | 0.570 | 0.350 | 0.553 | 0.853 | 0.598 | 0.328 |
| kev-0.8b-jigsaw | no | 0.797 | 0.590 | 0.467 | 0.519 | 0.890 | 0.421 | 0.412 |
| kev-0.8b-jigsaw | yes | 0.617 | 0.479 | 0.286 | 0.389 | 0.749 | 0.336 | 0.273 |
| kev-0.8b (base) | no | 0.797 | 0.484 | 0.487 | 0.495 | 0.892 | 0.434 | 0.426 |
| kev-0.8b (base) | yes | 0.583 | 0.340 | 0.269 | 0.313 | 0.734 | 0.290 | 0.251 |

Full 16-row matrix (incl. per-slot T-F1 and the Jigsaw-questions → intent mapping): [`conda/results/matrix.md`](conda/results/matrix.md).

- The Jigsaw fine-tune improves kev-4b on every metric; kev-0.8b gains only with context.
- CONDA conversation context hurts every model (a match's whole chat, unlike Jigsaw reply chains).
- Mapping the 7 Jigsaw answers to intent gets the best explicit F1 (0.785) but almost no implicit (≈0.07); A is 0 by construction.
- CONDA token labels are exact lexicon matches: a lexicon lookup scores T-F1 1.000, so token scores measure lexicon guessing.
- Zero-shot vs CONDA-trained baselines is a reference, not a like-for-like comparison.

## Layout

```
filter.py            all_data.csv -> filtered.csv (14 columns)
build_threads.py     filtered.csv -> threads.jsonl (parent_id chains, 5 nearest ancestors)
split_threads.py     threads.jsonl -> training.jsonl (3,000) + test.jsonl (1,000), balanced, split by article
to_kev.py            thread records -> Kev records (7 questions, THREAD/REPLY state)
ec2/                 training on a plain GPU box: setup_ec2.sh, kev_ec2.py (train/publish), run_all.sh, logs/
runs/jigsaw-*/       training reports: result.json, config.json, train.log, errors.jsonl, */report.json
conda/               CONDA eval: conda_to_kev.py, kev_conda_eval.py (check/run/matrix), run_matrix.sh, results/, logs/
hf/                  Hugging Face dataset card sources
```

## Reproduce

Python 3.13 on a CUDA GPU (we used one L40S, `g6e.2xlarge`). Data prep scripts are stdlib-only.

```bash
# 1. Data (or download from datasets/CS230-lol/JigsawToxicity)
python3 filter.py && python3 build_threads.py
python3 split_threads.py 3000 1000 && python3 to_kev.py training.jsonl test.jsonl
#    Kev partitions (train/calibration/development) are published under kev/ in the HF dataset.

# 2. Train (on the GPU box)
bash ec2/setup_ec2.sh                # uv, kev @ pinned ref, flash-linear-attention + triton 3.7.1
cd ec2 && mkdir -p data && cp ../data/*.jsonl data/ && bash run_all.sh   # 0.8b then 4b (~2 h on L40S)
HF_TOKEN=... python kev_ec2.py publish --name jigsaw-4b --repo <org>/kev-4b-jigsaw-toxicity

# 3. CONDA eval
git clone https://github.com/usydnlp/CONDA
python3 conda/conda_to_kev.py CONDA/data conda/data      # or download conda/data from the eval dataset
rsync -aL conda/ <box>:conda/                             # -L: conda/kev_modal.py is a symlink
ssh <box> 'cd conda && ~/kev-env/bin/python kev_conda_eval.py check && tmux new -d "bash run_matrix.sh"'  # ~8.5 h on 1 L40S
```

`run_matrix.sh` spreads the 8 jobs over all visible GPUs; finished jobs are skipped on rerun.

## Credits and licenses

- Jigsaw / Civil Comments data: CC0 1.0.
- `ec2/kev_modal.py` is copied unmodified from [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0),
  as are the base models `jaredpalmer/kev-{0.8b,4b}`.
- CONDA: Weld et al., Findings of ACL-IJCNLP 2021 ([paper](https://arxiv.org/abs/2106.06213), [data](https://github.com/usydnlp/CONDA)).
  `conda/conda_eval.py` re-implements its Table 5 metrics.
