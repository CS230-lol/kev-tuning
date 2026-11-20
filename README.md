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
| RNN-NLU (trained on CONDA) | – | 0.902 | 0.815 | 0.705 | 0.734 | 0.942 | 0.961 | 0.850 |
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
- Zero-shot vs CONDA-trained baselines is a reference, not a like-for-like comparison. Even the small RNN-NLU beats
  every Kev setup (UCA 0.902 vs 0.853, JSA 0.850 vs 0.443), so the gap comes from training on CONDA labels.
- RNN-NLU and Joint BERT rows are our own reproduction (one seed, same 8,974-utterance test set), not the paper's numbers.

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

Every step has a download shortcut, so you can start at any stage. All downloads below are public and need no token.
Each "build" path below was re-run from scratch and gives files byte-identical to the published ones.

Tools: Python 3 (data prep is standard-library only), [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
(`pip install -U huggingface_hub`, or prefix commands with `uvx --from huggingface_hub`), and for training/eval a Linux
box with one CUDA GPU. We used an AWS `g6e.2xlarge` (1× L40S 48 GB, Deep Learning Base OSS Nvidia Driver GPU AMI,
Amazon Linux 2023, 150 GB disk).

```bash
git clone https://github.com/CS230-lol/kev-tuning && cd kev-tuning
```

### 1. Jigsaw training data → `data/{train,calibration,development}.jsonl`

**Download** (what the models were trained on):

```bash
hf download CS230-lol/JigsawToxicity --repo-type dataset --include "kev/*" --local-dir hf-jigsaw
mkdir -p data && cp hf-jigsaw/kev/*.jsonl data/
```

**Or build from the raw Kaggle file.** Get `all_data.csv` (916 MB) from either source:

```bash
# Kaggle: accept the competition rules first, needs ~/.kaggle/kaggle.json
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f all_data.csv && unzip all_data.csv.zip
# or the unmodified copy on Hugging Face
hf download CS230-lol/JigsawToxicity all_data.csv --repo-type dataset --local-dir .
```

Then (about 4 GB free disk; ran on an 8 GB laptop):

```bash
python3 filter.py                          # all_data.csv  -> filtered.csv  (14 columns)
python3 build_threads.py filtered.csv threads.jsonl   # -> threads.jsonl (1,999,516 records with reply chains)
python3 split_threads.py 3000 1000         # -> training.jsonl + test.jsonl (balanced, split by article; seeded)
python3 to_kev.py training.jsonl test.jsonl            # -> training.kev.jsonl + test.kev.jsonl
# Kev's own splitter, at the version we used (jaredpalmer/kev@ac8e245)
curl -sLO https://raw.githubusercontent.com/jaredpalmer/kev/ac8e24524695cfb2d5147bf8063c3db3ef9276e6/skills/kev-finetune/scripts/split_data.py
python3 split_data.py training.kev.jsonl --out data --calibration 0.15 --development 0.01
cat data/development.jsonl >> data/train.jsonl   # splitter needs a development share; fold it back into train
cp test.kev.jsonl data/development.jsonl         # development = the 1,000 held-out-article test records
```

Intermediate files are also downloadable: `filtered.csv`, `threads.jsonl` and `splits/{training,test}.jsonl` from
[CS230-lol/JigsawToxicity](https://huggingface.co/datasets/CS230-lol/JigsawToxicity) (same `hf download ... <file>` form).

### 2. Train (optional: the models are on Hugging Face)

From your machine, with `<box>` the GPU host:

```bash
rsync -a ec2/ <box>:ec2/ && rsync -a data/ <box>:ec2/data/
ssh <box> 'bash ~/ec2/setup_ec2.sh'          # uv, Python 3.13 venv ~/kev-env, kev @ pinned ref in /kev (needs sudo, git)
ssh <box> 'tmux new -d -s train "bash ~/ec2/run_all.sh"'   # 0.8b then 4b, ~2 h on one L40S
ssh <box> 'cat ~/ec2/logs/status'            # ALL_DONE when finished; reports in ~/ec2/runs/jigsaw-*/
rsync -a --exclude checkpoint --exclude data <box>:ec2/runs/ runs/   # reports only; weights are on HF
# publish (needs a write token)
ssh <box> 'cd ~/ec2 && source ~/kev-env/bin/activate && HF_TOKEN=<token> python kev_ec2.py publish --name jigsaw-4b --repo <org>/kev-4b-jigsaw-toxicity'
```

### 3. CONDA zero-shot eval

**CONDA records → `conda/data/`**, download:

```bash
hf download CS230-lol/kev-jigsaw-toxicity-eval --repo-type dataset --include "conda/data/*" --local-dir .
```

or build from the CONDA repo (we used commit `3fb9067`; takes seconds):

```bash
git clone https://github.com/usydnlp/CONDA && git -C CONDA checkout 3fb90675f881079c85c1de1f25df1bf8d521194f
python3 conda/conda_to_kev.py CONDA/data conda/data    # test set = CONDA_valid.csv (official test labels are not public)
```

**Run** (all 8 jobs, ~8.5 h on one L40S; models download from Hugging Face automatically):

```bash
rsync -aL conda/ <box>:conda/                           # -L: conda/kev_modal.py is a symlink
# skip the next line if the box was already set up in step 2
rsync -a ec2/setup_ec2.sh <box>:ec2/ && ssh <box> 'bash ~/ec2/setup_ec2.sh'
ssh <box> 'cd ~/conda && ~/kev-env/bin/python kev_conda_eval.py check'   # each line must end "0 still over Kev's limits"
ssh <box> 'tmux new -d -s conda "bash ~/conda/run_matrix.sh"'
ssh <box> 'cat ~/conda/logs/status'                     # ALL_DONE when finished
rsync -a <box>:conda/results <box>:conda/logs conda/    # results/matrix.md is the table above
```

`run_matrix.sh` spreads the 8 jobs over all visible GPUs; a finished job is skipped on rerun.

### Raw predictions

Per-question probabilities for every Jigsaw and CONDA eval (1.3 GB). Paths inside mirror this repo (`runs/`, `conda/`):

```bash
hf download CS230-lol/kev-jigsaw-toxicity-eval --repo-type dataset --local-dir hf-eval
```

## Credits and licenses

- Jigsaw / Civil Comments data: CC0 1.0.
- `ec2/kev_modal.py` is copied unmodified from [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0),
  as are the base models `jaredpalmer/kev-{0.8b,4b}`.
- CONDA: Weld et al., Findings of ACL-IJCNLP 2021 ([paper](https://arxiv.org/abs/2106.06213), [data](https://github.com/usydnlp/CONDA)).
  `conda/conda_eval.py` re-implements its Table 5 metrics.
