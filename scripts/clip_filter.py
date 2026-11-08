"""Apply the student's rule for Track A photos with CLIP, then let the student set the cut-off.

The rule is the student's (LOG.md): drop road photos where no writing or
buildings can be seen, and photos with only an object or only people in them.
CLIP only scores how well each training photo matches the sentences below;
the student looks at the sheets and decides from which rank on photos go.

    uv run --with transformers python scripts/clip_filter.py score            all selected training photos
    uv run --with transformers python scripts/clip_filter.py score --limit 300   a quick timing check
    uv run python scripts/clip_filter.py sheets                               sheets at several ranks
    uv run --with transformers python scripts/clip_filter.py food --top 8000  food among the 8000 highest
    uv run python scripts/clip_filter.py food-sheets                          local and plain food, on sheets
    uv run python scripts/clip_filter.py drop --top 1500                      drop the 1500 highest,
                                                                              keeping local food

CLIP (openai/clip-vit-base-patch32) is used only here, to make
track_a/selection.csv; official.py and recipe.py never see it.
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import curate  # noqa: E402
import rules  # noqa: E402

SCORES = ROOT / "results" / "clip_scores.csv"
VAL_SCORES = ROOT / "results" / "clip_scores_val.csv"  # only to compare, never to select
VAL = ROOT / "data" / "starter" / "val"
REASON = "CLIP: 글씨나 건물이 보이지 않는 도로 사진, 또는 사물이나 인물만 있는 사진"

# The student's rule, as sentences for CLIP. A photo's score is how much of
# CLIP's choice among all the sentences falls on the DROP ones.
DROP = {
    "empty_road": "a photo of a plain road with no signs and no buildings",
    "object": "a close-up photo of a single object",
    "people": "a photo of only people",
}
KEEP = {
    "street_signs": "a photo of a city street with shop signs and writing",
    "buildings": "a photo of buildings in a city",
}
PROMPTS = {**DROP, **KEEP}

# The student's second rule: among the food photos, keep local and well-known
# dishes and drop plain food that could be anywhere, such as cafe food.
FOOD_SCORES = ROOT / "results" / "clip_food.csv"
IS_FOOD = {
    "food": "a photo of food or a drink",
    "object": "a photo of an object that is not food",
    "people": "a photo of people",
    "road": "a photo of a road",
    "buildings": "a photo of buildings",
    "nature": "a photo of plants, animals or landscape",
}
PLAIN_FOOD = {
    "coffee": "a photo of a cup of coffee",
    "cafe_dessert": "a photo of a cafe dessert or a cake",
    "bread": "a photo of a sandwich or bread",
    "fast_food": "a photo of fast food",
}
LOCAL_FOOD = {
    "korean": "a photo of traditional Korean food",
    "busan": "a photo of Busan pork soup or milmyeon cold noodles",
    "japanese": "a photo of Japanese ramen or sushi",
    "osaka": "a photo of Osaka okonomiyaki or takoyaki",
    "taiwanese": "a photo of Taiwanese xiaolongbao or night market food",
}
FOOD_KIND = {**PLAIN_FOOD, **LOCAL_FOOD}
# Dish names in the titles of Wikimedia Commons photos
DISHES = ["tteokbokki", "bibimbap", "bulgogi", "kimchi", "gukbap", "milmyeon", "naengmyeon",
          "samgyeopsal", "galbi", "japchae", "hotteok", "eomuk", "jjigae", "kimbap", "gimbap",
          "ramen", "sushi", "sashimi", "okonomiyaki", "takoyaki", "kushikatsu", "udon", "soba",
          "tempura", "yakitori", "onigiri", "donburi", "xiaolongbao", "beef noodle",
          "stinky tofu", "bubble tea", "gua bao", "lu rou fan", "oyster omelette", "떡볶이",
          "비빔밥", "국밥", "밀면", "냉면", "김치", "갈비", "寿司", "ラーメン", "お好み焼き",
          "たこ焼き", "小籠包", "牛肉麵", "臭豆腐", "珍珠奶茶"]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def prepare(path):
    """224 x 224 centre crop, normalised as CLIP expects."""
    im = rules.load_photo(path).convert("RGB")
    s = 224 / min(im.size)
    im = im.resize((max(224, round(im.width * s)), max(224, round(im.height * s))))
    left, top = (im.width - 224) // 2, (im.height - 224) // 2
    x = np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def features(out):
    """Newer transformers return an output object with the projection in pooler_output."""
    return getattr(out, "pooler_output", out)


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    tok = CLIPTokenizer.from_pretrained(name)

    def text(sentences):
        with torch.no_grad():
            t = features(model.get_text_features(**tok(list(sentences), padding=True, return_tensors="pt")))
        return t / t.norm(dim=-1, keepdim=True)

    def images(paths, pool):
        x = torch.from_numpy(np.stack(list(pool.map(lambda p: prepare(curate.TRAIN / p), paths))))
        with torch.no_grad():
            v = features(model.get_image_features(pixel_values=x))
        return v / v.norm(dim=-1, keepdim=True)

    return text, images


def read_scores(scores=SCORES):
    if not scores.exists():
        return []
    with open(scores, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_score(args):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    tok = CLIPTokenizer.from_pretrained(name)
    with torch.no_grad():
        t = features(model.get_text_features(**tok(list(PROMPTS.values()), padding=True, return_tensors="pt")))
        t = t / t.norm(dim=-1, keepdim=True)

    scores, folder = (SCORES, curate.TRAIN) if args.split == "train" else (VAL_SCORES, VAL)
    done = {r["path"] for r in read_scores(scores)}
    if args.split == "train":
        todo = [p for p in curate.read_selection() if p not in done]
    else:
        todo = [f"{c}/{p.name}" for c in rules.CITIES for p in sorted((VAL / c).glob("*.jpg"))
                if f"{c}/{p.name}" not in done]
    if args.limit:
        todo = todo[:args.limit]
    scores.parent.mkdir(parents=True, exist_ok=True)
    new = not scores.exists()
    start = time.time()
    with open(scores, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(8) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "drop_score", "best", *PROMPTS])
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            x = torch.from_numpy(np.stack(list(pool.map(lambda p: prepare(folder / p), chunk))))
            with torch.no_grad():
                v = features(model.get_image_features(pixel_values=x))
                v = v / v.norm(dim=-1, keepdim=True)
                prob = (100 * v @ t.T).softmax(dim=-1).numpy()
            for p, row in zip(chunk, prob):
                w.writerow([p, f"{row[:len(DROP)].sum():.4f}", list(PROMPTS)[row.argmax()],
                            *(f"{q:.4f}" for q in row)])
            f.flush()
            n = i + len(chunk)
            rate = n / (time.time() - start)
            print(f"\r  {n}/{len(todo)} photos, {rate:.1f}/s, about {(len(todo) - n) / rate / 60:.0f} min left",
                  end="", flush=True)
    print(f"\nscores in {scores}")


def titles():
    with open(ROOT / "data" / "starter" / "ATTRIBUTION.csv", newline="", encoding="utf-8") as f:
        return {r["file"].removeprefix("train/"): r["title"] for r in csv.DictReader(f)}


def cmd_food(args):
    text, images = load_clip()
    t_food, t_kind = text(IS_FOOD.values()), text(FOOD_KIND.values())
    title = titles()
    todo = [r["path"] for r in ranked()[:args.top]]
    start = time.time()
    with open(FOOD_SCORES, "w", newline="", encoding="utf-8") as f, ThreadPoolExecutor(8) as pool:
        w = csv.writer(f)
        w.writerow(["path", "food", "local", "best_kind", "dish_in_title", "title"])
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            v = images(chunk, pool)
            food = (100 * v @ t_food.T).softmax(dim=-1).numpy()[:, 0]
            kind = (100 * v @ t_kind.T).softmax(dim=-1).numpy()
            for p, fo, k in zip(chunk, food, kind):
                t = title.get(p, "")
                dish = next((d for d in DISHES if d in t.lower()), "")
                w.writerow([p, f"{fo:.4f}", f"{k[len(PLAIN_FOOD):].sum():.4f}",
                            list(FOOD_KIND)[k.argmax()], dish, t])
            n = i + len(chunk)
            print(f"\r  {n}/{len(todo)} photos, {n / (time.time() - start):.1f}/s", end="", flush=True)
    print(f"\nscores in {FOOD_SCORES}")


def local_food(min_food=0.5, min_local=0.5):
    """Photos the second rule keeps: food that CLIP finds local, or with a dish in its title."""
    if not FOOD_SCORES.exists():
        return {}
    with open(FOOD_SCORES, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if float(r["food"]) >= min_food]
    return {r["path"]: r for r in rows if float(r["local"]) >= min_local or r["dish_in_title"]}


def cmd_food_sheets(args):
    with open(FOOD_SCORES, newline="", encoding="utf-8") as f:
        food = [r for r in csv.DictReader(f) if float(r["food"]) >= args.min_food]
    keep = local_food(args.min_food, args.min_local)
    local = sorted((r for r in food if r["path"] in keep), key=lambda r: -float(r["local"]))
    plain = sorted((r for r in food if r["path"] not in keep), key=lambda r: float(r["local"]))
    print(f"{len(food)} food photos: {len(local)} local (kept), {len(plain)} plain (dropped)")
    for name, rows in (("food_local", local), ("food_plain", plain)):
        items = [(curate.TRAIN / r["path"],
                  f"{r['local']} {r['dish_in_title'] or r['best_kind']}") for r in rows]
        names = curate.draw_sheets(name, items)
        print(f"{name}: {len(rows)} photos on {len(names)} sheets")


def ranked():
    """Scored photos still in the selection, highest drop score first."""
    selected = set(curate.read_selection())
    rows = [r for r in read_scores() if r["path"] in selected]
    return sorted(rows, key=lambda r: -float(r["drop_score"]))


def cmd_sheets(args):
    rows = ranked()
    print(f"{len(rows)} scored photos in the selection")
    rng = np.random.default_rng(0)
    for lo, hi in zip(args.at[:-1], args.at[1:]):
        part = rows[lo:hi]
        if not part:
            break
        pick = sorted(rng.choice(len(part), min(48, len(part)), replace=False))
        items = [(curate.TRAIN / part[k]["path"],
                  f"#{lo + k + 1} {part[k]['drop_score']} {part[k]['best']}") for k in pick]
        names = curate.draw_sheets(f"clip_rank_{lo + 1}-{hi}", items)
        print(f"ranks {lo + 1} to {hi} (score {part[0]['drop_score']} to {part[-1]['drop_score']}): {names[0]}")


def cmd_drop(args):
    keep = local_food(min_local=args.min_local) if args.keep_local_food else {}
    rows = [r for r in ranked()[:args.top] if r["path"] not in keep]
    print(f"kept {args.top - len(rows)} local food photos among the {args.top} highest")
    paths = {r["path"] for r in rows}
    kept = [p for p in curate.read_selection() if p not in paths]
    curate.write_selection(kept)
    new = not curate.DROPPED.exists()
    when = datetime.now().isoformat(timespec="seconds")
    with open(curate.DROPPED, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "reason", "when"])
        for r in rows:
            w.writerow([r["path"], f"{REASON} (CLIP rank {args.top} and above; local food at {args.min_local} kept)", when])
    print(f"dropped {len(rows)} photos down to score {rows[-1]['drop_score']}; "
          f"the selection now has {len(kept)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--split", choices=["train", "val"], default="train",
                   help="val: score the validation photos, only to compare them with the training photos")
    s = sub.add_parser("sheets")
    s.add_argument("--at", type=int, nargs="+", default=[0, 250, 500, 1000, 2000, 4000, 8000])
    s = sub.add_parser("drop")
    s.add_argument("--top", type=int, required=True)
    s.add_argument("--keep-local-food", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--min-local", type=float, default=0.75, help="the student's choice: local food at or above this is kept")
    s = sub.add_parser("food")
    s.add_argument("--top", type=int, default=8000)
    s.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("food-sheets")
    s.add_argument("--min-food", type=float, default=0.5)
    s.add_argument("--min-local", type=float, default=0.5)
    args = ap.parse_args()
    {"score": cmd_score, "sheets": cmd_sheets, "drop": cmd_drop, "food": cmd_food,
     "food-sheets": cmd_food_sheets}[args.cmd](args)


if __name__ == "__main__":
    main()
