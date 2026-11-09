"""CONDA_valid.csv (the repro's test set) -> Kev records for zero-shot eval, with and without conversation context.

Per utterance:
  record A: `intent` (4-way choice, paper's annotator definitions) + the 7 Jigsaw questions from to_kev.py
  record B..: one `tok_<i>` choice question per token, TOKENS_PER_RECORD per record
[SEPA] and the empty-row placeholder are labelled deterministically, never asked.
Context = earlier utterances of the same conversationId (all three CSVs; inputs only, no labels), by chatTime.

Writes <out>/{ctx,noctx}.jsonl, <out>/index.jsonl (record -> utterance, question ids), <out>/gold.{label,seq.out}.
Usage: python3 conda_to_kev.py <CONDA/data dir> <out dir>
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from to_kev import QUESTIONS as JIGSAW, render  # same questions + THREAD/REPLY format Kev was tuned on

TOKENS_PER_RECORD = 8
EMPTY = "_EMPTY_"  # same placeholder as conda-repro/scripts/prepare_conda.py

INTENT = {"type": "choice", "instructions": "What is the intent of the REPLY in this Dota 2 in-game chat?", "criteria": {
    "explicit": "Explicit toxicity: contains toxic words meant to insult or humiliate others or make them quit "
                "(cursing others, slurs, sexual or hateful words, inflammatory insults)",
    "implicit": "Implicit toxicity: no toxic words, but insulting or humiliating others is clear from the text or "
                "conversation (sarcasm, mocking, e.g. 'u are poor dude', 'ez')",
    "action": "Action: not toxic, but contains an in-game action such as report, commend, pause, stop or exit game",
    "other": "Other: none of the above; may contain curses, self-deprecation or emotion not targeted at others",
}}
INTENT_CODE = {"explicit": "E", "implicit": "I", "action": "A", "other": "O"}

SLOT_CODE = {"toxic": "T", "character": "C", "dota": "D", "slang": "S", "pronoun": "P", "other": "O"}
SLOT_CRITERIA = {  # examples = most frequent lexicon words in CONDA_train (never the test file)
    "toxic": "Toxic word: profanity, insult, slur or offensive term (e.g. {T})",
    "character": "Dota 2 hero or character name or nickname (e.g. {C})",
    "dota": "Dota-specific term: item, ability, building, map or mechanic (e.g. {D})",
    "slang": "Game or chat slang (e.g. {S})",
    "pronoun": "Pronoun, including chat spellings (e.g. {P})",
    "other": "Any other word",
}


def load(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        pairs = re.findall(r"(.*?) \((\w+)\), ", r.get("slotTokens") or "")  # CONDA_test.csv: unlabelled, context only
        r["toks"] = [t for t, _ in pairs] or [EMPTY]
        r["tags"] = [l for _, l in pairs] or ["O"]
        rows.append(r)
    return rows


def slot_question(tok, occurrence, criteria):
    which = f' (occurrence {occurrence})' if occurrence > 1 else ""
    return {"type": "choice", "instructions": f'In the REPLY, what kind of word is "{tok}"{which}?', "criteria": criteria}


def records_for(u, context, criteria):
    """-> [(record, question ids)] for one utterance. Labels go on every question (Kev's scorer wants them)."""
    state = render([{"text": c} for c in context], clean(u["utterance"]))
    first = {"intent": {**INTENT, "label": next(k for k, v in INTENT_CODE.items() if v == u["intentClass"])}}
    # Jigsaw questions have no CONDA gold; label is a placeholder, only their predictions are used
    first.update({k: {**v, "label": False if v["type"] == "noul" else 0} for k, v in JIGSAW.items()})
    out = [({"state": state, "questions": first}, list(first))]
    seen, toks = collections.Counter(), []
    for i, (t, tag) in enumerate(zip(u["toks"], u["tags"])):
        seen[t.lower()] += 1
        if tag != "SEPA" and t != EMPTY:
            q = slot_question(t, seen[t.lower()], criteria)
            q["label"] = next(k for k, v in SLOT_CODE.items() if v == tag)
            toks.append((f"tok_{i}", q))
    for j in range(0, len(toks), TOKENS_PER_RECORD):
        chunk = dict(toks[j:j + TOKENS_PER_RECORD])
        out.append(({"state": state, "questions": chunk}, list(chunk)))
    return out


def clean(text):
    return text.replace(" [SEPA] ", " / ")


def examples(train, n=8):
    c = collections.defaultdict(collections.Counter)
    for r in train:
        for t, tag in zip(r["toks"], r["tags"]):
            c[tag][t.lower()] += 1
    return {k: ", ".join(w for w, _ in c[k].most_common(n)) for k in "TCDSP"}


def main(src, out):
    files = {s: load(os.path.join(src, f"CONDA_{s}.csv")) for s in ("train", "valid", "test")}
    criteria = {k: v.format(**examples(files["train"])) for k, v in SLOT_CRITERIA.items()}
    conv = collections.defaultdict(list)
    for rows in files.values():
        for r in rows:
            conv[r["conversationId"]].append(r)
    for v in conv.values():
        v.sort(key=lambda r: (int(r["chatTime"]), int(r["Id"])))

    os.makedirs(out, exist_ok=True)
    test = files["valid"]
    with open(os.path.join(out, "gold.label"), "w") as fl, open(os.path.join(out, "gold.seq.out"), "w") as fs:
        for u in test:
            fl.write(u["intentClass"] + "\n"); fs.write(" ".join(u["tags"]) + "\n")
    index = []
    for mode in ("ctx", "noctx"):
        with open(os.path.join(out, f"{mode}.jsonl"), "w", encoding="utf-8") as f:
            for ui, u in enumerate(test):
                history = conv[u["conversationId"]]
                ctx = [clean(r["utterance"]) for r in history[:history.index(u)]] if mode == "ctx" else []
                for rec, qids in records_for(u, ctx, criteria):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if mode == "ctx":
                        index.append({"utt": ui, "qids": qids})
    with open(os.path.join(out, "index.jsonl"), "w") as f:  # same layout for both modes
        f.writelines(json.dumps(x) + "\n" for x in index)
    n_ctx = sum(1 for u in test if conv[u["conversationId"]].index(u) > 0)
    print(f"{len(test)} utterances ({n_ctx} with context) -> {len(index)} records per mode, "
          f"{sum(len(x['qids']) for x in index)} questions; token criteria: {json.dumps(criteria)}")


if __name__ == "__main__":
    u = {"utterance": "gg [SEPA] u noob", "intentClass": "E", "toks": ["gg", "[SEPA]", "u", "noob", "u"], "tags": ["S", "SEPA", "P", "T", "P"]}
    recs = records_for(u, ["prev"], {k: "x" for k in SLOT_CRITERIA})
    assert recs[0][1][0] == "intent" and recs[0][0]["questions"]["intent"]["label"] == "explicit"
    assert recs[1][1] == ["tok_0", "tok_2", "tok_3", "tok_4"]  # SEPA skipped, indices kept
    assert "(occurrence 2)" in recs[1][0]["questions"]["tok_4"]["instructions"]
    assert recs[0][0]["state"] == "THREAD:\n> prev\n\nREPLY:\ngg / u noob"
    main(*sys.argv[1:3])
