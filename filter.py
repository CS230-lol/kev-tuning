"""Stream all_data.csv -> filtered.csv keeping only the columns needed for toxicity + thread reconstruction."""
import csv
import sys

csv.field_size_limit(sys.maxsize)

KEEP = [
    "id", "comment_text", "split", "created_date", "parent_id", "article_id",
    "toxicity", "severe_toxicity", "obscene", "sexual_explicit",
    "identity_attack", "insult", "threat", "toxicity_annotator_count",
]
# ponytail: identity columns (male, muslim, ...) dropped; ~90% empty. Add back if you run the bias metric.

src = sys.argv[1] if len(sys.argv) > 1 else "all_data.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "filtered.csv"

with open(src, newline="", encoding="utf-8") as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=KEEP, extrasaction="ignore")
    w.writeheader()
    n = 0
    for row in r:
        # parent_id comes as "123.0" in some dumps; normalise to int string
        if row["parent_id"]:
            row["parent_id"] = str(int(float(row["parent_id"])))
        w.writerow(row)
        n += 1
print(f"wrote {n} rows -> {dst}")
