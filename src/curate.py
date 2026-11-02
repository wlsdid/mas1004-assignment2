"""Look at photos, find the ones the model gets wrong, and decide which training photos to use.

This file is complete. It draws numbered contact sheets into results/curate/;
you look at them and decide. For Track A, your decisions change
track_a/selection.csv, the list of training photos official.py uses.

    python src/curate.py look busan                   every selected Busan photo, 48 per sheet
    python src/curate.py look busan --every 20        every 20th one: a few sheets for a first look
    python src/curate.py mistakes results/baseline_seed0/model.onnx
                                                      validation photos the model got wrong most confidently
    python src/curate.py suspects results/baseline_seed0/model.onnx
                                                      selected training photos the model doubts most
    python src/curate.py drop look_busan_03 5 17 40 --reason "inside a shop"
    python src/curate.py drop busan/commons_104708282.jpg --reason "map, not a photo"
    python src/curate.py undrop busan/commons_104708282.jpg
    python src/curate.py count

Every sheet comes with a .csv of the same name listing the number and path of
each photo on it, so that `drop` can take the sheet name and numbers.

`drop` does not delete anything. It takes the photos out of
track_a/selection.csv (made from all training photos the first time) and
writes the photo, the reason and the time into track_a/dropped.csv. `undrop`
puts a photo back. Write your rule down before you start, then drop by it.

`mistakes` and `suspects` run an ONNX model on the photos the way the grader
prepares them. On a laptop without a GPU this takes a few minutes for the
validation set, and longer for all training photos; --every 4 looks at every
fourth photo only.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from rules import CITIES  # noqa: E402

TRAIN = ROOT / "data" / "starter" / "train"
SELECTION = ROOT / "track_a" / "selection.csv"
DROPPED = ROOT / "track_a" / "dropped.csv"
OUT = ROOT / "results" / "curate"
TILE, COLS, ROWS = 200, 8, 6


def read_selection():
    """Paths relative to the training folder, in order, as selection.csv lists them."""
    if not SELECTION.exists():
        return [f"{c}/{p.name}" for p, c in rules.list_photos(TRAIN)]
    with open(SELECTION, newline="", encoding="utf-8-sig") as f:
        return [r["path"] for r in csv.DictReader(f)]


def write_selection(paths):
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION, "w", newline="", encoding="utf-8") as f:
        f.write("path\n" + "".join(p + "\n" for p in paths))


def draw_sheets(name, items):
    """items: (path, caption) pairs. Writes results/curate/<name>_NN.png and .csv, 48 per sheet."""
    OUT.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    names = []
    for start in range(0, len(items), per):
        sheet_name = f"{name}_{start // per:02d}"
        chunk = items[start:start + per]
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (path, caption) in enumerate(chunk):
            x, y = (k % COLS) * TILE, (k // COLS) * (TILE + 22)
            try:
                im = rules.load_photo(path)
                im.thumbnail((TILE - 4, TILE - 4))
                sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
            except OSError:
                draw.text((x + 10, y + 90), "unreadable", fill="red")
            draw.rectangle([x + 2, y + 2, x + 30, y + 18], fill="black")
            draw.text((x + 5, y + 4), str(k + 1), fill="white")
            draw.text((x + 4, y + TILE + 4), caption[:34], fill="black")
        sheet.save(OUT / f"{sheet_name}.png")
        with open(OUT / f"{sheet_name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["number", "path", "caption"])
            for k, (path, caption) in enumerate(chunk):
                w.writerow([k + 1, path_name(path), caption])
        names.append(sheet_name)
    return names


def path_name(path):
    """A path as selection.csv writes it if it is a training photo, else as it is."""
    path = Path(path)
    try:
        return path.relative_to(TRAIN).as_posix()
    except ValueError:
        return str(path)


def predict(onnx_path, paths, size=rules.TRACK_A_IMAGE_SIZE, batch=32):
    """City probabilities (N, 5) for photos, prepared as the grader prepares them."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = []
    with ThreadPoolExecutor(os.cpu_count() or 2) as pool:
        for i in range(0, len(paths), batch):
            x = np.stack(list(pool.map(lambda p: rules.grader_input(rules.load_photo(p), size),
                                       paths[i:i + batch])))
            logits = session.run(["logits"], {"image": x})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            out.append(e / e.sum(1, keepdims=True))
            print(f"\r  {min(i + batch, len(paths))}/{len(paths)} photos", end="", flush=True)
    print()
    return np.concatenate(out)


def cmd_look(args):
    paths = [p for p in read_selection() if p.split("/")[0] == args.city][::args.every]
    names = draw_sheets(f"look_{args.city}", [(TRAIN / p, p.split("/")[1][:34]) for p in paths])
    print(f"{len(paths)} selected {args.city} photos on {len(names)} sheets in {OUT}: "
          f"{names[0]} to {names[-1]}" if names else f"no selected {args.city} photos")


def run_name(onnx):
    """results/baseline_seed0/model.onnx -> baseline_seed0, for the names of the sheets."""
    path = Path(onnx)
    return path.parent.name if path.name == "model.onnx" else path.stem


def cmd_mistakes(args):
    rows = rules.list_photos(args.photos)[::args.every]
    probs = predict(args.onnx, [p for p, _ in rows], args.size)
    for i, city in enumerate(CITIES):
        wrong = [(probs[k].max(), k) for k, (_, c) in enumerate(rows) if c == city and probs[k].argmax() != i]
        total = sum(1 for _, c in rows if c == city)
        wrong.sort(reverse=True)
        items = [(rows[k][0], f"-> {CITIES[probs[k].argmax()]} {p:.2f}") for p, k in wrong[:args.n]]
        names = draw_sheets(f"mistakes_{run_name(args.onnx)}_{city}", items)
        print(f"{city}: {len(wrong)} of {total} wrong; the {len(items)} most confident on {', '.join(names)}")


def cmd_suspects(args):
    paths = read_selection()[::args.every]
    probs = predict(args.onnx, [TRAIN / p for p in paths], args.size)
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / f"suspects_{run_name(args.onnx)}.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "city", "own_city_probability", "answer"])
        for p, pr in zip(paths, probs):
            w.writerow([p, p.split("/")[0], round(float(pr[CITIES.index(p.split('/')[0])]), 4), CITIES[pr.argmax()]])
    for i, city in enumerate(CITIES):
        mine = sorted((probs[k][i], k) for k, p in enumerate(paths) if p.split("/")[0] == city)
        items = [(TRAIN / paths[k], f"{city} {pr:.2f}, says {CITIES[probs[k].argmax()]}") for pr, k in mine[:args.n]]
        names = draw_sheets(f"suspects_{run_name(args.onnx)}_{city}", items)
        print(f"{city}: the {len(items)} photos with the lowest {city} probability on {', '.join(names)}")
    print(f"all probabilities in {table}")


def resolve(targets):
    """Sheet name + numbers, or paths, to training paths."""
    if targets and (OUT / f"{targets[0]}.csv").exists():
        with open(OUT / f"{targets[0]}.csv", newline="", encoding="utf-8") as f:
            by_number = {r["number"]: r["path"] for r in csv.DictReader(f)}
        missing = [n for n in targets[1:] if n not in by_number]
        if missing:
            raise SystemExit(f"{targets[0]} has no photo numbered {', '.join(missing)}")
        return [by_number[n] for n in targets[1:]]
    return [t.replace("\\", "/") for t in targets]


def cmd_drop(args):
    paths = resolve(args.targets)
    selection = read_selection()
    chosen = set(paths)
    unknown = [p for p in paths if p not in set(selection)]
    if unknown:
        raise SystemExit(f"not in the selection: {', '.join(unknown[:5])}")
    kept = [p for p in selection if p not in chosen]
    write_selection(kept)
    new = not DROPPED.exists()
    with open(DROPPED, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "reason", "when"])
        for p in paths:
            w.writerow([p, args.reason, datetime.now().isoformat(timespec="seconds")])
    print(f"dropped {len(paths)} photo{'s' * (len(paths) != 1)} ({args.reason}); the selection now has {len(kept)}")


def cmd_undrop(args):
    paths = resolve(args.targets)
    selection = read_selection()
    missing = [p for p in paths if not (TRAIN / p).is_file()]
    if missing:
        raise SystemExit(f"not training photos: {', '.join(missing[:5])}")
    write_selection(selection + [p for p in paths if p not in set(selection)])
    if DROPPED.exists():
        with open(DROPPED, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["path"] not in set(paths)]
        with open(DROPPED, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "reason", "when"])
            w.writeheader()
            w.writerows(rows)
    print(f"put back {len(paths)} photo{'s' * (len(paths) != 1)}")


def cmd_count(args):
    selection = read_selection()
    everything = {f"{c}/{p.name}" for p, c in rules.list_photos(TRAIN)}
    print(f"{'city':8s} {'selected':>9s} {'of':>6s}   by source")
    for city in CITIES:
        mine = [p for p in selection if p.split("/")[0] == city]
        total = sum(1 for p in everything if p.split("/")[0] == city)
        sources = {}
        for p in mine:
            src = p.split("/")[1].split("_")[0]
            sources[src] = sources.get(src, 0) + 1
        print(f"{city:8s} {len(mine):9d} {total:6d}   " + ", ".join(f"{s} {n}" for s, n in sorted(sources.items())))
    repeats = len(selection) - len(set(selection))
    print(f"total {len(selection)} rows" + (f", of which {repeats} repeat a photo" if repeats else ""))
    if DROPPED.exists():
        with open(DROPPED, newline="", encoding="utf-8") as f:
            reasons = {}
            for r in csv.DictReader(f):
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        if reasons:
            print("dropped: " + "; ".join(f"{n} {why}" for why, n in sorted(reasons.items(), key=lambda x: -x[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("look")
    p.add_argument("city", choices=CITIES)
    p.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    for name in ("mistakes", "suspects"):
        p = sub.add_parser(name)
        p.add_argument("onnx")
        p.add_argument("--n", type=int, default=48, help="photos per city")
        p.add_argument("--every", type=int, default=1, help="use every n-th photo only")
        p.add_argument("--size", type=int, default=rules.TRACK_A_IMAGE_SIZE, help="the model's image size")
        if name == "mistakes":
            p.add_argument("--photos", default=str(ROOT / "data" / "starter" / "val"))
    p = sub.add_parser("drop")
    p.add_argument("targets", nargs="+", help="a sheet name and numbers on it, or training paths")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("undrop")
    p.add_argument("targets", nargs="+")
    sub.add_parser("count")
    args = ap.parse_args()
    {"look": cmd_look, "mistakes": cmd_mistakes, "suspects": cmd_suspects, "drop": cmd_drop,
     "undrop": cmd_undrop, "count": cmd_count}[args.cmd](args)


if __name__ == "__main__":
    main()
