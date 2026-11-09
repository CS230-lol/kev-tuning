"""Train / calibrate / score / publish a Kev delta on a plain GPU box (EC2), same recipe as kev_modal.py::run_train.

    python kev_ec2.py train   --data data --name jigsaw-0.8b --init-from jaredpalmer/kev-0.8b
    python kev_ec2.py train   --data data --name jigsaw-4b   --init-from jaredpalmer/kev-4b
    HF_TOKEN=... python kev_ec2.py publish --name jigsaw-4b --repo <you>/kev-4b-jigsaw-toxicity

Reuses kev_modal.py's helpers (scoring, temperature fit, report, card) so numbers match the Modal path; Modal volumes
are swapped for no-ops and outputs land in ./runs/<name>. Needs the kev repo cloned at /kev (setup_ec2.sh).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import kev_modal as km

# ponytail: Modal volume calls become no-ops; nothing else in the helpers touches Modal at run time
km.runs = km.hf_cache = types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
RUNS = Path("runs")


def train(a):
    import torch
    from kev.checkpoint import Checkpoint, read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import metrics, paired_bootstrap
    from kev.model import load_tokenizer
    from kev.suite import write_json

    out = RUNS / a.name
    if out.exists():
        sys.exit(f"{out} exists; pick a new --name")
    shutil.copytree(a.data, out / "data")
    started = time.time()
    init = Checkpoint(a.init_from)
    meta, args = init.meta, init.meta.extra["args"]
    cfg = {"lr": a.lr or min(args["lr"], km.MAX_DELTA_LR), "batch": a.batch or args["batch"], "accum": a.accum or args["accum"],
           "checkpointing": args["checkpointing"], "epochs": a.epochs, "seed": a.seed, "replay": a.replay, "p_none_pair": 0.25}
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    checks = km.check_partitions(out / "data", tok)
    write_json(out / "config.json", {"name": a.name, "init_from": a.init_from, "base": meta.base, "config": cfg, "data": checks,
                                     "kev_ref": km.KEV_REF, "gpu": torch.cuda.get_device_name(0)})
    km.stage(f"{meta.base} from {a.init_from}; config {json.dumps(cfg)}; data {json.dumps({k: (v['records'], v['over_limit']) for k, v in checks.items()})}")

    run = str(out / "checkpoint")
    cmd = [sys.executable, "-m", "kev.train", "--data", out.resolve() / "data/train.jsonl", "--init_from", a.init_from, "--out", Path(run).resolve(),
           "--device", "cuda", "--dtype", "bf16", "--base", meta.base, "--lora", meta.lora, "--head_dim", meta.head_dim,
           "--lora_targets", args["lora_targets"], "--option_isolation", int(meta.option_isolation),
           "--special_embeddings", int(meta.special_embeddings), "--weights_dtype", meta.weights_dtype,
           "--epochs", cfg["epochs"], "--lr", cfg["lr"], "--batch", cfg["batch"], "--accum", cfg["accum"],
           "--checkpointing", cfg["checkpointing"], "--seed", cfg["seed"], "--p_none_pair", cfg["p_none_pair"]]
    if meta.base_revision: cmd += ["--base_revision", meta.base_revision]
    if cfg["replay"]: cmd += ["--suite", km.SUITE, "--replay", cfg["replay"]]
    km.train_subprocess([str(c) for c in cmd], out / "train.log")

    calibration = load_records(out / "data/calibration.jsonl")
    development = load_records(out / "data/development.jsonl")
    km.stage(f"scoring on {len(calibration)} calibration + {len(development)} development records")
    report, rows, T = km.score_checkpoint(run, development, out, calibration)
    m = read_meta(run); m.temperature = T
    m.extra["temperature_fit"] = {"rows": "calibration.jsonl", "n": len(calibration), "method": "min NLL, micro", "value": T}
    write_meta(run, m)  # checkpoint serves calibrated probabilities by default
    result = km.write_result(out, development, report, rows, T, name=a.name, init_from=a.init_from, base=meta.base, config=cfg, data=checks)

    km.stage(f"scoring baseline {a.init_from} on the same records")
    b_report, b_rows, b_T = km.score_checkpoint(a.init_from, development, out / "baseline", calibration)
    result["baseline"] = {"run": a.init_from, "development": km.summary_block(b_report, b_rows, b_T)}
    result["bootstrap"] = {k: paired_bootstrap(km.tempered(rows, T), km.tempered(b_rows, b_T), metric=k, aggregation="micro") for k in ("acc", "brier", "ece")}
    if a.regression:  # forgetting check on Kev's public recipe
        sample = km.regression_sample(a.regression, cfg["seed"])
        _, r_rows, _ = km.score_checkpoint(run, sample, out / "regression/finetuned")
        _, rb_rows, _ = km.score_checkpoint(a.init_from, sample, out / "regression/baseline")
        result["regression"] = {"n": len(sample), "suite": "evals/v7/decision-v7",
                                "finetuned": metrics([r for r in r_rows if r["variant"] == "clean"]),
                                "baseline": metrics([r for r in rb_rows if r["variant"] == "clean"])}
    result["wall_seconds"] = round(time.time() - started); result["gpu"] = torch.cuda.get_device_name(0)
    write_json(out / "result.json", result)
    km.print_result(result)


def publish(a):
    from kev.suite import read_json
    if not os.environ.get("HF_TOKEN"):
        sys.exit("set HF_TOKEN (a write token from huggingface.co/settings/tokens)")
    out = RUNS / a.name
    r = read_json(out / "result.json")
    dev, b = r["development"], r["baseline"]["development"]["calibrated"]
    card = km.CARD.format(base=r["base"], repo=a.repo, init_from=r["init_from"], n_train=r["data"]["train"]["records"], temperature=r["temperature"],
                          n_cal=r["data"]["calibration"]["records"], n_dev=dev["n_questions"], acc=dev["calibrated"]["acc"], brier=dev["calibrated"]["brier"],
                          ece=dev["calibrated"]["ece"], cov=dev["calibrated"]["coverage_at_5pct_error"], b_acc=b["acc"], b_brier=b["brier"],
                          b_ece=b["ece"], b_cov=b["coverage_at_5pct_error"], kev_ref=km.KEV_REF[:12])
    (out / "model-card.md").write_text(card, encoding="utf-8")
    msg = f"Upload {a.name} (delta from {r['init_from']}; development acc {dev['calibrated']['acc']:.3f}, T {r['temperature']:.2f})"
    # public on purpose (no --private): user asked for a public repo
    subprocess.run([sys.executable, "-m", "kev.publish", "--run", str((out / "checkpoint").resolve()), "--repo", a.repo,
                    "--card", str((out / "model-card.md").resolve()), "--message", msg], check=True, cwd=km.KEV_ROOT)
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="dir with train/calibration/development.jsonl")
    t.add_argument("--name", required=True)
    t.add_argument("--init-from", default="jaredpalmer/kev-4b")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--replay", type=int, default=2000, help="public-recipe records mixed in against forgetting (0 = off)")
    t.add_argument("--regression", type=int, default=300, help="public dev groups scored for forgetting (0 = skip)")
    t.add_argument("--seed", type=int, default=0)
    # 0 = checkpoint's own value. Recipe was tuned on an 80GB H100: on OOM halve --batch and double --accum
    t.add_argument("--lr", type=float, default=0.0)
    t.add_argument("--batch", type=int, default=0)
    t.add_argument("--accum", type=int, default=0)
    p = sub.add_parser("publish")
    p.add_argument("--name", required=True)
    p.add_argument("--repo", required=True, help="<hf-user>/<model-name>")
    a = ap.parse_args()
    {"train": train, "publish": publish}[a.cmd](a)
