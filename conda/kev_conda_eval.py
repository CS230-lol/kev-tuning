"""Zero-shot CONDA eval of Kev checkpoints -> Table 5 metrics (UCA, U-F1, T-F1, per-slot, JSA).

    python kev_conda_eval.py check                                   # token limits, CPU
    python kev_conda_eval.py run --job CS230-lol/kev-4b-jigsaw-toxicity:ctx [--job ...]
    python kev_conda_eval.py matrix                                  # results/*/result.json -> results/matrix.md

Scores with kev_modal's raw predictor (fp32, temperature 1.0: argmax metrics don't depend on temperature).
A finished job (results/<model>/<mode>/result.json) is skipped, so rerunning resumes.
Needs kev_modal.py + conda_eval.py next to this file and data/ from conda_to_kev.py.
"""
import argparse
import gc
import json
import shutil
import sys
import types
from pathlib import Path

import kev_modal as km
from conda_eval import INTENTS, SLOTS, evaluate

km.runs = km.hf_cache = types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
DATA, RESULTS = Path("data"), Path("results")
MODELS = ["CS230-lol/kev-0.8b-jigsaw-toxicity", "CS230-lol/kev-4b-jigsaw-toxicity", "jaredpalmer/kev-0.8b", "jaredpalmer/kev-4b"]
INTENT_CODE = {"explicit": "E", "implicit": "I", "action": "A", "other": "O"}
SLOT_CODE = {"toxic": "T", "character": "C", "dota": "D", "slang": "S", "pronoun": "P", "other": "O"}
EXPLICIT = ["obscene", "insult", "identity_attack", "threat", "sexual_explicit"]


def jigsaw_intent(a):
    """Fixed rule, not tuned on CONDA: not toxic -> O; toxic and any subtype 'Clearly present' or severe -> E; else I."""
    if a["toxic"] != "true":
        return "O"
    return "E" if a["severe_toxicity"] == "1" or any(a[k] == "2" for k in EXPLICIT) else "I"


def assemble(rows, index, gold_seq):
    """Kev rows -> (pred intent, pred jigsaw intent, pred tags) per utterance, checking every row maps to its record."""
    answers = {}
    for r in rows:
        if r["variant"] != "clean":
            continue
        i = int(r["id"].split("/")[1])
        assert r["question"] in index[i]["qids"], (r["id"], r["question"])
        p = r["p"]
        answers.setdefault(i, {})[r["question"]] = r["keys"][max(range(len(p)), key=p.__getitem__)]
    per_utt = [{} for _ in gold_seq]
    asked = [set() for _ in gold_seq]
    for i, x in enumerate(index):
        per_utt[x["utt"]].update(answers.get(i, {}))
        asked[x["utt"]].update(x["qids"])
    labels, jig, seqs = [], [], []
    for a, q, gold in zip(per_utt, asked, gold_seq):
        labels.append(INTENT_CODE.get(a.get("intent"), "<none>"))  # <none> = record dropped -> counted wrong
        jig.append(jigsaw_intent(a) if "toxic" in a else "<none>")
        seq = []
        for j, g in enumerate(gold):
            if g == "SEPA" or f"tok_{j}" not in q:  # never asked: [SEPA] or the empty-row placeholder (O)
                seq.append(g)
            else:
                seq.append(SLOT_CODE.get(a.get(f"tok_{j}"), "<none>"))
        seqs.append(seq)
    return labels, jig, seqs


def gold():
    return [l.strip() for l in open(DATA / "gold.label")], [l.split() for l in open(DATA / "gold.seq.out")]


def trim(records, tok):
    """Chat is token-dense (1400 chars can exceed the 384-token state cap): drop oldest THREAD lines until a record fits."""
    from kev.data import materialize
    from kev.model import fits
    n = 0
    for r in records:
        while not fits(materialize(r), tok) and r["state"].startswith("THREAD:\n"):
            r["state"] = drop_oldest(r["state"])
            n += 1
    return n


def drop_oldest(state):
    """'THREAD:\\n> a\\n> b\\n\\nREPLY:\\nx' -> 'THREAD:\\n> b\\n\\nREPLY:\\nx'; last line gone -> 'REPLY:\\nx' (render's no-context form)."""
    head, _, reply = state.partition("\n\n")
    lines = head.split("\n")[2:]
    return ("THREAD:\n" + "\n".join(lines) + "\n\n" if lines else "") + reply


def run(job):
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import Checkpoint
    from kev.data import load_records
    from kev.model import load_tokenizer
    model, mode = job.rsplit(":", 1)
    out = RESULTS / model.replace("/", "__") / mode
    if (out / "result.json").exists():
        print(f"skip {job}: done"); return
    shutil.rmtree(out, ignore_errors=True)  # partial output from a crashed run: evaluate_records needs a fresh dir
    gl, gs = gold()
    index = [json.loads(l) for l in open(DATA / "index.jsonl")]
    records = load_records(DATA / f"{mode}.jsonl")
    assert len(records) == len(index)
    meta = Checkpoint(model).meta
    trimmed = trim(records, load_tokenizer(meta.base, revision=meta.base_revision))
    km.stage(f"{job}: dropped {trimmed} oldest context lines to fit Kev's limits")
    km.stage(f"{job}: scoring {len(records)} records on {torch.cuda.get_device_name(0)}")
    predictor = km.raw_predictor(model)
    try:
        _, rows = evaluate_records(records, predictor, out)
    finally:
        del predictor; gc.collect(); torch.cuda.empty_cache()
    labels, jig, seqs = assemble(rows, index, gs)
    for name, pl in (("pred.label", labels), ("pred.jigsaw.label", jig)):
        (out / name).write_text("\n".join(pl) + "\n")
    (out / "pred.seq.out").write_text("\n".join(" ".join(s) for s in seqs) + "\n")
    result = {"model": model, "mode": mode, "n_rows": len(rows), "context_lines_trimmed": trimmed,
              "intent_question": evaluate(gl, gs, labels, seqs), "jigsaw_mapped": evaluate(gl, gs, jig, seqs)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    km.stage(f"{job}: done UCA {result['intent_question']['UCA']:.3f} T-F1 {result['intent_question']['T-F1']:.3f}")


def check():
    from kev.checkpoint import Checkpoint
    from kev.data import load_records, materialize
    from kev.model import fits, load_tokenizer
    for base in ("jaredpalmer/kev-0.8b", "jaredpalmer/kev-4b"):
        meta = Checkpoint(base).meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        for mode in ("ctx", "noctx"):
            recs = load_records(DATA / f"{mode}.jsonl")
            trimmed = trim(recs, tok)
            over = sum(not fits(materialize(r), tok) for r in recs)
            print(f"{base} {mode}: {len(recs)} records, {trimmed} context lines trimmed, {over} still over Kev's limits")


ORDER = ["UCA"] + [f"U-F1({c})" for c in INTENTS] + ["T-F1"] + [f"T-F1({c})" for c in SLOTS if c != "O"] + ["JSA"]
REFERENCE = [  # conda-repro RESULTS.md (trained on CONDA) and local lexicon lookup; same 8,974-utterance test set
    ("always O (majority)", [0.739, 0, 0, 0, 0.850, None, None, None, None, None, None, None]),
    ("lexicon lookup, intent always O", [0.739, 0, 0, 0, 0.850, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.739]),
    ("RNN-NLU (trained on CONDA)", [0.902, 0.815, 0.705, 0.734, 0.942, 0.961, 0.934, 0.980, 0.924, 0.716, 0.992, 0.850]),
    ("Joint BERT (trained on CONDA)", [0.924, 0.876, 0.751, 0.804, 0.956, 0.983, 0.970, 0.990, 0.969, 0.895, 0.998, 0.894]),
]


def matrix():
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    lines = ["| model | context | intent via | " + " | ".join(ORDER) + " |", "|" + "---|" * (len(ORDER) + 3)]
    for name, vals in REFERENCE:
        lines.append(f"| {name} | - | - | " + " | ".join(map(fmt, vals)) + " |")
    for model in MODELS:
        for mode in ("ctx", "noctx"):
            p = RESULTS / model.replace("/", "__") / mode / "result.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            for via in ("intent_question", "jigsaw_mapped"):
                lines.append(f"| {model} | {mode} | {via} | " + " | ".join(fmt(r[via][k]) for k in ORDER) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert drop_oldest("THREAD:\n> a\n> b\n\nREPLY:\nx") == "THREAD:\n> b\n\nREPLY:\nx"
    assert drop_oldest("THREAD:\n> b\n\nREPLY:\nx") == "REPLY:\nx"
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("matrix")
    r = sub.add_parser("run"); r.add_argument("--job", action="append", required=True, help="<model>:<ctx|noctx>")
    a = ap.parse_args()
    if a.cmd == "run":
        for j in a.job: run(j)
    else:
        {"check": check, "matrix": matrix}[a.cmd]()
