---
license: cc0-1.0
language:
- en
pretty_name: Jigsaw Unintended Bias — Threaded Toxicity
size_categories:
- 1M<n<10M
task_categories:
- text-classification
task_ids:
- multi-label-classification
tags:
- toxicity
- moderation
- civil-comments
- conversation
- threads
configs:
- config_name: threads
  default: true
  data_files: threads.jsonl
- config_name: filtered
  data_files: filtered.csv
- config_name: all_data
  data_files: all_data.csv
- config_name: balanced
  data_files:
  - split: train
    path: splits/training.jsonl
  - split: test
    path: splits/test.jsonl
- config_name: kev
  data_files:
  - split: train
    path: kev/train.jsonl
  - split: calibration
    path: kev/calibration.jsonl
  - split: development
    path: kev/development.jsonl
---

# Jigsaw Unintended Bias — Threaded Toxicity

Comments from the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)
competition (Civil Comments platform, 2015–2017), with each comment's **reply chain reconstructed** so a toxicity
classifier can see the conversation the comment was written in, not only the comment alone.

Built to fine-tune Kev (an open System One-style decision model) for context-aware toxic chat detection.

## Files / configs

| Config | File | Rows | Description |
| --- | --- | --- | --- |
| `threads` (default) | `threads.jsonl` | 1,999,516 | One record per comment with its ancestor chain as `context`. |
| `filtered` | `filtered.csv` | 1,999,516 | `all_data.csv` reduced to 14 columns (text, thread keys, toxicity labels). |
| `all_data` | `all_data.csv` | 1,999,516 | Original Kaggle `all_data.csv`, unmodified (46 columns incl. identity and reaction columns). |
| `balanced` | `splits/{training,test}.jsonl` | 3,000 / 1,000 | Balanced sample of `threads`, split by `article_id` (see below). |
| `kev` | `kev/{train,calibration,development}.jsonl` | 2,547 / 449 / 1,000 | `balanced` in Kev's labelled-request format; what [kev-0.8b/4b-jigsaw-toxicity](https://huggingface.co/CS230-lol/kev-4b-jigsaw-toxicity) were trained on. |

```python
from datasets import load_dataset

threads = load_dataset("CS230-lol/JigsawToxicity", "threads", split="train")
train = threads.filter(lambda r: r["split"] == "train")
test = threads.filter(lambda r: r["split"] == "test")
```

Every config loads as a single `train` split; the original Kaggle partition is in the `split` column (`train` 1,804,875 / `test` 194,641).

## Thread reconstruction

`parent_id` holds the `id` of the comment being replied to. Verified on the data:

- 1,134,709 comments are replies (non-empty `parent_id`); 1,005,497 of those parents exist in the dump.
- Every resolved parent shares the child's `article_id`.
- The parent is created before the child in all but 2 of 1,005,497 pairs.

For each comment, `parent_id` is followed upward until a top-level comment (no parent) or a parent missing from the
dump. The **nearest 5 ancestors** are kept, ordered root → direct parent. Sibling replies are not included; `context`
is the linear chain the comment responds to.

## `threads` schema

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Comment id. |
| `split` | string | Original Kaggle partition: `train` or `test`. |
| `article_id` | string | Article the comment thread belongs to. |
| `depth` | int | Number of resolvable ancestors (full depth, before the 5-ancestor cap). 0 = top-level or orphan. |
| `parent_missing` | bool | True if the chain breaks: some ancestor's `parent_id` points to a comment not in the dump. |
| `context` | list of `{id, text}` | Up to 5 nearest ancestors, root → direct parent. Empty for top-level comments. |
| `comment` | string | The comment to classify. |
| `toxicity` | float | Fraction of annotators rating the comment toxic (0–1). Primary target; ≥ 0.5 is the competition's positive class. |
| `severe_toxicity`, `obscene`, `sexual_explicit`, `identity_attack`, `insult`, `threat` | float | Subtype annotator fractions (0–1). |
| `toxicity_annotator_count` | int | Number of annotators for the toxicity labels. |

Example (`context` texts truncated):

```json
{"id": "5167187", "split": "train", "article_id": "328935", "depth": 3, "parent_missing": false,
 "context": [{"id": "5166075", "text": "I wonder how many people have actually been inconvenienced since this initiative took effect..."},
             {"id": "5166798", "text": "Does it make a difference? Is that the standard for right and wrong?..."},
             {"id": "5166939", "text": "So from what I gather from your comment, you do not view transgender people as normal..."}],
 "comment": "That's right. They are not normal. And I am starting from the premise that they are ABNORMAL. Proceed wth the typical racist, bigot, sexist rubbish.\nThanks!",
 "toxicity": 0.803, "severe_toxicity": 0.066, "obscene": 0.049, "sexual_explicit": 0.082,
 "identity_attack": 0.131, "insult": 0.754, "threat": 0.016, "toxicity_annotator_count": 61}
```

The reply alone reads as an insult; the context shows it is aimed at transgender people, which is the kind of signal
`context` is meant to supply.

## `balanced` and `kev` (fine-tuning splits)

`split_threads.py` samples `threads` into 3,000 train / 1,000 test records:

- Test = articles with `crc32(article_id) % 100 < 20`; no article appears on both sides.
- Equal counts per stratum (toxic ≥ 0.5) × (has context): 50% toxic, 50% with context.
- Inside the toxic strata, 6% of the file is reserved per rare subtype (threat, sexual_explicit, obscene,
  identity_attack ≥ 0.5) so each answer level occurs at least 5% of the time.
- Records with `parent_missing` are skipped, so "no context" really means top-level.

`to_kev.py` turns each record into a Kev request: state = `THREAD:` (ancestors, `> ` lines, oldest first) + `REPLY:`
(the comment), capped at 1,400 characters with the reply kept first. Seven questions per record: `toxic` (noul),
`severe_toxicity` (2-level score, cut 0.1) and `obscene`, `sexual_explicit`, `identity_attack`, `insult`, `threat`
(3-level scores: Not present / Borderline or mild / Clearly present, cuts 0.1 and 0.5). The training file was split
85/15 into `train`/`calibration` (2 conflicting states dropped); `development` is the 1,000 test records.

Code, training logs and results: [github.com/CS230-lol/kev-tuning](https://github.com/CS230-lol/kev-tuning).

## `filtered` columns

`id, comment_text, split, created_date, parent_id, article_id, toxicity, severe_toxicity, obscene, sexual_explicit,
identity_attack, insult, threat, toxicity_annotator_count`. `parent_id` is normalised from float (`"123.0"`) to an
integer string.

## Statistics (`threads`)

| | Count |
| --- | --- |
| Records | 1,999,516 |
| With non-empty `context` | 1,005,497 (50.3%) |
| `parent_missing = true` | 154,182 |
| `toxicity ≥ 0.5` | 159,782 (8.0%) |
| `toxicity ≥ 0.5` with context | 75,626 |
| Articles | 116,943 |

Depth of replies with a resolvable parent: 1 → 441,276; 2 → 239,914; 3 → 129,782; 4 → 72,367; 5 → 42,326; 6–9 → 60,823; ≥ 10 → 19,009.

## Preprocessing

Reproducible with standard-library Python scripts (no dependencies):

```bash
python3 filter.py                         # all_data.csv -> filtered.csv
python3 build_threads.py                  # filtered.csv -> threads.jsonl
python3 split_threads.py 3000 1000        # threads.jsonl -> training.jsonl + test.jsonl (balanced)
python3 to_kev.py training.jsonl test.jsonl  # -> *.kev.jsonl (Kev format)
```

## Limitations and caveats

- **Split leakage across threads.** The Kaggle `train`/`test` partition is per comment, not per thread, so a `test`
  comment's `context` can contain `train` comments (and vice versa). For leakage-free evaluation, re-split by `article_id`.
- **Incomplete chains.** About 11% of replies point to a parent absent from the dump; `parent_missing` flags these.
  Their context is partial or empty, which does not mean they are top-level comments.
- **Context is truncated** to 5 ancestors; long chains lose their earliest turns. Text length is not truncated.
- **Identity columns dropped** from `filtered`/`threads` (mostly empty). Use `all_data` to compute the competition's
  unintended-bias metrics (subgroup AUC, BPSN, BNSP).
- **Class imbalance.** Only 8% of comments are toxic at the 0.5 threshold.
- **Offensive content.** The data contains real toxic, obscene, threatening and identity-attacking text.
- Labels are crowd-annotator fractions and contain annotator disagreement and bias.

## License and citation

Civil Comments text and Jigsaw annotations are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The thread reconstruction adds no new text.

```bibtex
@misc{jigsaw-unintended-bias-in-toxicity-classification,
  author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
  title = {Jigsaw Unintended Bias in Toxicity Classification},
  year = {2019},
  howpublished = {\url{https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification}},
  publisher = {Kaggle}
}

@article{borkan2019nuanced,
  title = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification},
  author = {Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
  journal = {arXiv preprint arXiv:1903.04561},
  year = {2019}
}
```
