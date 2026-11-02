"""Check a submission the way the grader will, and score it on photos you have.

This file is complete. Do not change it.

    python src/check_submission.py                        Track B: submission/
    python src/check_submission.py --onnx results/baseline_seed0/model.onnx --size 224
    python src/check_submission.py --photos data/mine     score on any folder of <city>/ photos
    python src/check_submission.py --no-val               only the checks (train_b.py printed the score)

It reads the ONNX file only, never your training code, so it measures exactly
what you hand in. It checks the format and the size limits in rules.py, then
runs the model on data/starter/val, preparing every photo with
rules.grader_input as the grader does.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from rules import CITIES  # noqa: E402

SUBMISSION_KEYS = {"image_size", "starting_weights", "data", "summary"}


def read_submission(folder):
    """submission.json, with its problems listed."""
    problems = []
    path = Path(folder) / "submission.json"
    if not path.exists():
        return None, [f"{path} is missing"]
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, [f"submission.json is not valid JSON: {e}"]
    missing = SUBMISSION_KEYS - set(info)
    if missing:
        problems.append(f"submission.json is missing {', '.join(sorted(missing))}")
    size = info.get("image_size")
    lo, hi = rules.TRACK_B_IMAGE_SIZES
    if not isinstance(size, int) or not lo <= size <= hi:
        problems.append(f"image_size must be a whole number from {lo} to {hi}, it is {size!r}")
    return info, problems


def check_model(onnx_path, size, track):
    """Format and size checks on the ONNX file. Returns (measurements, problems)."""
    m = rules.measure_onnx(onnx_path, size)
    problems = list(m["problems"])
    if track == "B":
        if m["parameters"] > rules.TRACK_B_MAX_PARAMETERS:
            problems.append(f"{m['parameters']:,} parameters, more than {rules.TRACK_B_MAX_PARAMETERS:,}")
        if m["gmacs"] > rules.TRACK_B_MAX_GMACS:
            problems.append(f"{m['gmacs']:.3f} billion multiply-adds per photo, more than {rules.TRACK_B_MAX_GMACS}")
        if m["file_mb"] > rules.TRACK_B_MAX_FILE_MB:
            problems.append(f"the file is {m['file_mb']:.0f} MB, more than {rules.TRACK_B_MAX_FILE_MB} MB")
    return m, problems


def score_folder(onnx_path, folder, size, threads=os.cpu_count() or 2, batch=32):
    """Balanced accuracy of the ONNX model on folder/<city>/ photos."""
    import onnxruntime
    rows = rules.list_photos(folder)
    if not rows:
        return None
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def prepare(row):
        return rules.grader_input(rules.load_photo(row[0]), size)

    true, pred = [], []
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = np.stack(list(pool.map(prepare, chunk)))
            logits = session.run(["logits"], {"image": x})[0]
            if logits.shape != (len(chunk), len(CITIES)):
                raise SystemExit(f"The model gave logits of shape {logits.shape}, not ({len(chunk)}, {len(CITIES)}).")
            pred += logits.argmax(1).tolist()
            true += [CITIES.index(city) for _, city in chunk]
    score, per_city, confusion = rules.balanced_accuracy(true, pred)
    return {"n": len(rows), "score": score, "per_city": per_city, "confusion": confusion.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(ROOT / "submission"),
                    help="Track B folder with model.onnx and submission.json")
    ap.add_argument("--onnx", help="check this ONNX file instead (Track A runs write one)")
    ap.add_argument("--size", type=int, help="the image size for --onnx")
    ap.add_argument("--photos", nargs="*", help="folders of <city>/ photos to score on")
    ap.add_argument("--no-val", action="store_true",
                    help="do not score data/starter/val; train_b.py has printed that score already")
    args = ap.parse_args()

    problems = []
    if args.onnx:
        onnx_path, size, track = Path(args.onnx), args.size or rules.TRACK_A_IMAGE_SIZE, "A"
    else:
        info, problems = read_submission(args.submission)
        onnx_path, size, track = Path(args.submission) / "model.onnx", (info or {}).get("image_size", 224), "B"
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} is missing.")

    if isinstance(size, int):
        m, more = check_model(onnx_path, size, track)
        problems += more
        print(f"{onnx_path}: {m['file_mb']:.1f} MB, {m['parameters'] / 1e6:.2f} million parameters, "
              f"{m['gmacs']:.2f} billion multiply-adds per {size} x {size} photo")
    if track == "B":
        print(f"  limits: {rules.TRACK_B_MAX_PARAMETERS / 1e6:.0f} million parameters, "
              f"{rules.TRACK_B_MAX_GMACS} billion multiply-adds")
    if problems:
        print("\nNOT ACCEPTED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("  format and limits: OK")

    folders = args.photos
    if folders is None:
        val = ROOT / "data" / "starter" / "val"
        folders = [val] if val.exists() and not args.no_val else []
    for folder in folders:
        t = time.time()
        r = score_folder(onnx_path, folder, size)
        if r is None:
            print(f"\n{folder}: no photos in <city>/ folders")
            continue
        print(f"\n{folder}: score {r['score']:.3f} on {r['n']} photos ({time.time() - t:.0f}s)")
        print("  " + ", ".join(f"{c} {v:.3f}" for c, v in r["per_city"].items()))
        print("  confusion (rows: true city, columns: answer):")
        for c, row in zip(CITIES, r["confusion"]):
            print(f"    {c:7s} " + " ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main()
