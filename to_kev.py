"""training.jsonl / test.jsonl (thread records) -> Kev labelled records (*.kev.jsonl).

State = reply chain (oldest -> newest) + the reply to classify, capped at MAX_STATE chars:
the reply is kept first, then ancestors nearest-first until the budget runs out.
Usage: python3 to_kev.py training.jsonl test.jsonl
"""
import json
import sys

MAX_STATE = 1400  # Kev state limit (~384 tokens)

LEVELS = ["Not present", "Borderline or mild", "Clearly present"]
SUBTYPES = {  # id -> what the subtype means (Jigsaw annotation definitions)
    "obscene": "obscene or vulgar language (swearing, cursing)",
    "sexual_explicit": "sexually explicit content",
    "identity_attack": "a negative or hateful attack on someone because of their identity (race, religion, gender, sexuality, disability...)",
    "insult": "an insult, or inflammatory or negative language toward a person or group",
    "threat": "a threat or wish of harm or violence against a person or group",
}
QUESTIONS = {
    "toxic": {"type": "noul",
              "instructions": "Is the REPLY toxic: rude, disrespectful or unreasonable enough to make someone leave the discussion?"},
    "severe_toxicity": {"type": "score",
                        "instructions": "Is the REPLY severely toxic: very hateful, aggressive or disrespectful?",
                        "criteria": ["Not severely toxic", "Possibly severely toxic"]},
    **{k: {"type": "score", "instructions": f"Does the REPLY contain {d}?", "criteria": LEVELS} for k, d in SUBTYPES.items()},
}


def level(v, cuts=(0.1, 0.5)):
    """Annotator fraction -> level index. ponytail: fixed cuts 0.1/0.5; tune if the 'mild' band proves noisy."""
    v = v or 0.0
    return sum(v >= c for c in cuts)


def clip(text, n):
    return text if len(text) <= n else text[: max(n - 1, 0)] + "…"


def render(context, comment):
    reply = "REPLY:\n" + clip(comment, MAX_STATE - 8)
    budget = MAX_STATE - len(reply) - len("THREAD:\n\n")
    if not context or budget < 80:
        return reply
    lines = []
    for c in reversed(context):  # nearest ancestor first
        line = "> " + c["text"].replace("\n", " ")
        if len(line) + 1 > budget:
            if not lines:  # direct parent alone too long: keep its head
                lines.append(clip(line, budget - 1))
            break
        lines.append(line)
        budget -= len(line) + 1
    return "THREAD:\n" + "\n".join(reversed(lines)) + "\n\n" + reply


def convert(d):
    q = {k: dict(v) for k, v in QUESTIONS.items()}
    q["toxic"]["label"] = d["toxicity"] >= 0.5
    q["severe_toxicity"]["label"] = level(d["severe_toxicity"], cuts=(0.1,))
    for k in SUBTYPES:
        q[k]["label"] = level(d[k])
    return {"state": render(d["context"], d["comment"]), "questions": q}


if __name__ == "__main__":
    assert [level(x) for x in (None, 0.05, 0.1, 0.49, 0.5, 1.0)] == [0, 0, 1, 1, 2, 2]
    s = render([{"text": "a" * 2000}, {"text": "parent"}], "hi")
    assert s.endswith("REPLY:\nhi") and "> parent" in s and len(s) <= MAX_STATE, s[:80]
    assert len(render([], "x" * 5000)) <= MAX_STATE
    for path in sys.argv[1:]:
        dst = path.replace(".jsonl", ".kev.jsonl")
        with open(path, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
            n = sum(out.write(json.dumps(convert(json.loads(l)), ensure_ascii=False) + "\n") > 0 for l in f)
        print(f"{dst}: {n} records")
