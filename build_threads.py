"""filtered.csv -> threads.jsonl: each comment with its reply chain (root -> parent) as chat context.

Thread structure: parent_id points at the id of the comment being replied to (always same article_id,
always created earlier). Walking parent_id up to a comment with no parent gives the chat history.
"""
import csv
import json
import sys

csv.field_size_limit(sys.maxsize)

MAX_CONTEXT = 5  # nearest ancestors kept; Kev state is ~1400 chars so deeper history gets cut anyway
LABELS = ["toxicity", "severe_toxicity", "obscene", "sexual_explicit", "identity_attack", "insult", "threat"]


def ancestors(i, rows, k=MAX_CONTEXT):
    """rows: id -> (parent_id, text). Return (chain root->parent, capped to k nearest; parent_missing; full depth)."""
    chain, missing = [], False
    p = rows[i][0]
    while p:
        if p not in rows:
            missing = True  # reply to a comment not in the dump
            break
        chain.append(p)
        p = rows[p][0]
    return chain[:k][::-1], missing, len(chain)


def main(src="filtered.csv", dst="threads.jsonl"):
    # pass 1: only parent + text in memory (full row dicts swap on an 8GB machine)
    with open(src, newline="", encoding="utf-8") as f:
        rows = {r["id"]: (r["parent_id"], r["comment_text"]) for r in csv.DictReader(f)}
    # pass 2: stream rows again for labels
    with open(src, newline="", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for r in csv.DictReader(f):
            i = r["id"]
            chain, missing, depth = ancestors(i, rows)
            out.write(json.dumps({
                "id": i,
                "split": r["split"],
                "article_id": r["article_id"],
                "depth": depth,
                "parent_missing": missing,
                "context": [{"id": a, "text": rows[a][1]} for a in chain],
                "comment": r["comment_text"],
                **{k: float(r[k]) if r[k] else None for k in LABELS},
                "toxicity_annotator_count": int(r["toxicity_annotator_count"]),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records -> {dst}")


def _selftest():
    rows = {"1": ("", "a"), "2": ("1", "b"), "3": ("2", "c"), "4": ("99", "d")}
    assert ancestors("3", rows) == (["1", "2"], False, 2)
    assert ancestors("3", rows, k=1) == (["2"], False, 2)  # keeps nearest
    assert ancestors("1", rows) == ([], False, 0)
    assert ancestors("4", rows) == ([], True, 0)


if __name__ == "__main__":
    _selftest()
    main(*sys.argv[1:])
