"""How many training and validation photos are at or above the CLIP cut-off the student chose.

    uv run python scripts/clip_compare.py

The cut-off is the score of rank 8000 among the training photos (clip8000).
Validation photos are only counted here, never selected.
"""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["seoul", "busan", "tokyo", "osaka", "taipei"]


def read(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as f:
        return [(r["path"].split("/")[0], float(r["drop_score"])) for r in csv.DictReader(f)]


train, val = read("clip_scores.csv"), read("clip_scores_val.csv")
cut = sorted((s for _, s in train), reverse=True)[7999]
print(f"cut-off: drop score {cut:.4f} (rank 8000 of {len(train)} training photos)")
print(f"{'city':8} {'train':>18} {'val':>18}")
for city in CITIES + ["total"]:
    row = []
    for photos in (train, val):
        mine = [s for c, s in photos if city in (c, "total")]
        over = sum(s >= cut for s in mine)
        row.append(f"{over:5d} / {len(mine):5d} {100 * over / len(mine):4.1f}%")
    print(f"{city:8} {row[0]:>18} {row[1]:>18}")
best = Counter()
with open(ROOT / "results" / "clip_scores_val.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if float(r["drop_score"]) >= cut:
            best[r["best"]] += 1
print("validation photos at or above the cut-off, by the closest sentence:", dict(best))
