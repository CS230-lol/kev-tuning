---
license: other
license_name: mixed
language:
- en
pretty_name: Kev Jigsaw toxicity — raw evaluation outputs
viewer: false
tags:
- toxicity
- evaluation
- predictions
- kev
---

# Kev Jigsaw toxicity — raw evaluation outputs

Per-question predictions behind the results in [github.com/CS230-lol/kev-tuning](https://github.com/CS230-lol/kev-tuning),
plus the CONDA records Kev was scored on. Metrics, reports and the eval matrix live in the GitHub repo; this dataset
holds the large files. Paths mirror the GitHub repo.

Models: [CS230-lol/kev-0.8b-jigsaw-toxicity](https://huggingface.co/CS230-lol/kev-0.8b-jigsaw-toxicity),
[CS230-lol/kev-4b-jigsaw-toxicity](https://huggingface.co/CS230-lol/kev-4b-jigsaw-toxicity), and base
`jaredpalmer/kev-{0.8b,4b}`. Training data: [CS230-lol/JigsawToxicity](https://huggingface.co/datasets/CS230-lol/JigsawToxicity) (`kev` config).

## Files

| Path | What |
| --- | --- |
| `runs/jigsaw-{0.8b,4b}/development/` | Fine-tuned model on the 1,000 held-out Jigsaw records (7,000 questions) |
| `runs/jigsaw-*/calibration/` | Same on the 449 calibration records (temperature fit) |
| `runs/jigsaw-*/baseline/{development,calibration}/` | Base Kev on the same records |
| `runs/jigsaw-*/regression/{finetuned,baseline}/` | 300 groups of Kev's public decision-v7 suite (forgetting check) |
| `conda/results/<model>/<ctx\|noctx>/` | Zero-shot CONDA eval, 4 models × with/without context |
| `conda/data/{ctx,noctx}.jsonl` | CONDA `CONDA_valid.csv` (8,974 utterances) as Kev records: 18,380 records, 101,544 questions each |
| `conda/data/index.jsonl` | Record line → utterance index and question ids (same for both modes) |
| `conda/data/gold.{label,seq.out}` | CONDA gold intent and token tags, one utterance per line |

Each eval directory has:

- `predictions.jsonl`: one line per record, `{"id": "custom/<line>", "request_sha256", "prediction": {"probabilities": {question: {option: p}}}}`, raw (temperature 1).
- `rows.json`: one entry per question, `{id, question, keys, label, p, variant, ...}` as written by `kev.benchmark.evaluate_records`.
- `report.json`: Kev's metric report.
- CONDA only: `pred.label`, `pred.jigsaw.label`, `pred.seq.out` (argmax predictions in CONDA's format) and `result.json` (UCA, U-F1, T-F1, JSA).

No Jigsaw text is stored here (record ids are line numbers in the `kev` config files). `conda/data/` contains CONDA utterance text.

## CONDA records

Built by `conda/conda_to_kev.py` from the [CONDA repo](https://github.com/usydnlp/CONDA). Per utterance:

- Record 1: a 4-way `intent` choice (explicit / implicit / action / other, from the paper's annotator guidelines) plus
  the 7 Jigsaw questions (their labels are placeholders: CONDA has no Jigsaw gold).
- Then one `tok_<i>` choice per token (toxic / character / dota / slang / pronoun / other), 8 per record; `[SEPA]` is
  never asked. Example words per class come from `CONDA_train` only.
- `ctx`: earlier utterances of the same conversation as `THREAD:` lines; `noctx`: the utterance alone. At run time,
  `kev_conda_eval.py` drops the oldest thread lines of 143 records (1,263 lines total) to fit Kev's 384-token state limit.

## Licenses and credits

- Jigsaw / Civil Comments: CC0 1.0.
- CONDA (`conda/data/`): Weld et al., *CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding
  and detection*, Findings of ACL-IJCNLP 2021. The CONDA repo states no license; text is redistributed here for research
  reproducibility with attribution, and will be removed on request from the authors.
- Base models and scoring code: [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0).

```bibtex
@inproceedings{weld-etal-2021-conda,
    title = "{CONDA}: a {CON}textual Dual-Annotated dataset for in-game toxicity understanding and detection",
    author = "Weld, Henry  and
      Huang, Guanghao  and
      Lee, Jean  and
      Zhang, Tongshu  and
      Wang, Kunze  and
      Guo, Xinghong  and
      Long, Siqu  and
      Poon, Josiah  and
      Han, Caren",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.213",
    doi = "10.18653/v1/2021.findings-acl.213",
    pages = "2406--2416",
}
```
