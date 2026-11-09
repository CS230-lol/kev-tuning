"""threads.jsonl -> training.jsonl + test.jsonl, balanced sample, split by article so no thread leaks across.

Strata: toxic (toxicity >= 0.5) x has_context, equal count each. Records with a broken chain
(parent_missing) are skipped: a "no context" record must really be top-level.
Inside toxic strata, RARE_QUOTA of the file is reserved per rare subtype (score >= 0.5) so each
subtype's "strong" level clears Kev's 5% minimum.
Usage: python3 split_threads.py [n_train] [n_test]
"""
import json
import random
import sys
import zlib

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TEST_ARTICLE_PCT = 20  # % of articles routed to test
SEED = 0
RARE = ["threat", "sexual_explicit", "obscene", "identity_attack"]  # rarest first
RARE_QUOTA = 0.06  # share of the whole file per rare subtype


def side(article_id):
    # stable across runs/machines (unlike hash())
    return "test" if zlib.crc32(article_id.encode()) % 100 < TEST_ARTICLE_PCT else "train"


def stratum(d):
    return (d["toxicity"] >= 0.5, bool(d["context"]))


def tag(d):
    """First rare subtype the record is strong in, else 'any'. One tag per record -> no duplicates."""
    if d["toxicity"] >= 0.5:
        for k in RARE:
            if (d[k] or 0) >= 0.5:
                return k
    return "any"


def quotas(n):
    """Per (toxic, ctx) stratum: how many records to draw from each tag pool."""
    per = n // 4
    rare = round(RARE_QUOTA * n / 2)  # toxic half carries the quota, split over 2 ctx strata
    return {True: {**{k: rare for k in RARE}, "any": per - rare * len(RARE)}, False: {"any": per}}


def main():
    rng = random.Random(SEED)
    want = {"train": quotas(N_TRAIN), "test": quotas(N_TEST)}
    pools, seen = {}, {}  # reservoir per (side, stratum, tag)
    with open("threads.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["parent_missing"]:
                continue
            s, st, t = side(d["article_id"]), stratum(d), tag(d)
            key = (s, st, t)
            k = want[s][st[0]][t]
            pool = pools.setdefault(key, [])
            seen[key] = n = seen.get(key, 0) + 1
            if len(pool) < k:
                pool.append(line)
            elif (j := rng.randrange(n)) < k:
                pool[j] = line
    for key in sorted(pools, key=str):
        print(key, f"sampled {len(pools[key])} of {seen[key]}")
    for s, path in (("train", "training.jsonl"), ("test", "test.jsonl")):
        lines = [l for key, p in pools.items() if key[0] == s for l in p]
        rng.shuffle(lines)
        with open(path, "w", encoding="utf-8") as out:
            out.writelines(lines)
        print(f"{path}: {len(lines)} records")


if __name__ == "__main__":
    assert side("123") == side("123") and stratum({"toxicity": 0.5, "context": [1]}) == (True, True)
    assert tag({"toxicity": 0.9, "threat": 0.6, "obscene": 0.7, "sexual_explicit": 0, "identity_attack": 0}) == "threat"
    assert sum(quotas(3000)[True].values()) == sum(quotas(3000)[False].values()) == 750
    main()
