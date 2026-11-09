"""CONDA metrics from the paper (Table 5): UCA, U-F1 per intent, T-F1 (+ per slot), JSA.

T-F1 = token-level micro F1 over slot classes excluding O (and the SEPA separator).

Usage: python conda_eval.py <gold.label> <gold.seq.out> <pred.label> <pred.seq.out>
Run with no args for a self-check.
"""
import sys

INTENTS = ["E", "I", "A", "O"]
SLOTS = ["T", "S", "C", "D", "P", "O"]


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def evaluate(gl, gs, pl, ps):
    assert len(gl) == len(gs) == len(pl) == len(ps), "line counts differ"
    res = {"UCA": sum(g == p for g, p in zip(gl, pl)) / len(gl)}
    for c in INTENTS:
        tp = sum(g == c and p == c for g, p in zip(gl, pl))
        res["U-F1(%s)" % c] = f1(tp, sum(p == c for p in pl) - tp, sum(g == c for g in gl) - tp)

    short = sum(len(p) < len(g) for g, p in zip(gs, ps))
    if short:
        print("warning: %d predictions shorter than gold (truncated?); missing tokens count as wrong" % short)
    ps = [p[:len(g)] + ["<none>"] * (len(g) - len(p)) for g, p in zip(gs, ps)]
    gt = [t for s in gs for t in s]
    pt = [t for s in ps for t in s]
    micro = [0, 0, 0]
    for c in SLOTS:
        tp = sum(g == c and p == c for g, p in zip(gt, pt))
        fp = sum(p == c for p in pt) - tp
        fn = sum(g == c for g in gt) - tp
        res["T-F1(%s)" % c] = f1(tp, fp, fn)
        if c != "O":
            micro = [micro[0] + tp, micro[1] + fp, micro[2] + fn]
    res["T-F1"] = f1(*micro)
    res["JSA"] = sum(a == b and g == p for a, b, g, p in zip(gl, pl, gs, ps)) / len(gl)
    return res


def read(path, split):
    return [l.split() if split else l.strip() for l in open(path, encoding="utf-8")]


def _selftest():
    gl, gs = ["E", "O"], [["T", "P"], ["O"]]
    r = evaluate(gl, gs, gl, gs)
    assert r["UCA"] == r["JSA"] == r["T-F1"] == 1.0
    r = evaluate(gl, gs, ["E", "E"], [["T", "O"], ["O"]])
    assert r["UCA"] == 0.5 and r["JSA"] == 0.0 and abs(r["T-F1"] - 2 / 3) < 1e-9
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        _selftest()
    else:
        a = sys.argv[1:5]
        r = evaluate(read(a[0], 0), read(a[1], 1), read(a[2], 0), read(a[3], 1))
        order = ["UCA"] + ["U-F1(%s)" % c for c in INTENTS] + ["T-F1"] + ["T-F1(%s)" % c for c in SLOTS] + ["JSA"]
        print("\t".join(order))
        print("\t".join("%.3f" % r[k] for k in order))
