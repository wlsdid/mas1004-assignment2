"""Train on Google Colab without writing anything in Colab.

This file is complete. Your agent queues the runs on your computer, you press
Run all in the notebook, and the results come back to your computer as one
zip file.

On your computer:

    python src/colab.py link     write your repository into assignment2_colab.ipynb
                                 and print the link that opens it in Colab
    python src/colab.py take     bring in the newest from_colab.zip in Downloads
    python src/colab.py take ~/Desktop/from_colab.zip

The notebook on Colab runs:

    python src/colab.py run      every run in colab_runs.txt that has not run yet
    python src/colab.py pack     results/ and submission/ into from_colab.zip

colab_runs.txt has one command per line, each starting with "python src/",
for example

    python src/official.py --tag balanced
    python src/curate.py mistakes results/balanced_seed0/model.onnx --every 2

Lines starting with # are skipped. A line that already ran with the same
recipe, selection and script is not run again, so after a broken session
Run all carries on where it stopped. A run that fails stops the rest; its
output is in the zip.
"""

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "assignment2_colab.ipynb"
RUNS = ROOT / "colab_runs.txt"
RESULTS = ROOT / "results"
DONE = RESULTS / "colab_done.txt"
LOGS = RESULTS / "colab_logs"
COLAB_EXPERIMENTS = RESULTS / "colab_experiments.csv"
EXPERIMENTS = ROOT / "experiments.csv"
ZIP_NAME = "from_colab.zip"
REPO_LINE = re.compile(r'^REPO = ".*"$', re.M)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def github_address():
    """https://github.com/<owner>/<repository> of the remote origin."""
    url = git("config", "--get", "remote.origin.url")
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(\.git)?/?$", url)
    if not m:
        sys.exit(f"The remote origin is not a GitHub repository: {url!r}")
    return f"https://github.com/{m.group(1)}/{m.group(2)}"


def cmd_link(args):
    address = github_address()
    if address.lower().endswith("/jdasam/mas1004-assignment2"):
        sys.exit("This is the template, not the student's own repository.")
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "main"
    owner_repo = address.removeprefix("https://github.com/")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            new = REPO_LINE.sub(f'REPO = "{address}"', source)
        else:  # the "Open in Colab" badge
            new = re.sub(r"colab\.research\.google\.com/github/[^/]+/[^/]+/blob/[^/]+/",
                         f"colab.research.google.com/github/{owner_repo}/blob/{branch}/", source)
        changed |= new != source
        cell["source"] = new.splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"assignment2_colab.ipynb {'now reads from' if changed else 'already reads from'} {address}")
    if changed:
        print("Commit and push assignment2_colab.ipynb before the student opens the link.")
    print(f"Colab link: https://colab.research.google.com/github/{owner_repo}/blob/{branch}/assignment2_colab.ipynb")


def queued():
    """The commands in colab_runs.txt, in order."""
    if not RUNS.exists():
        return []
    lines = [line.strip() for line in RUNS.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def run_key(line, words):
    """Changes when the line, the recipe, the selection or the script changes,
    and for check_submission.py also when the submission changes."""
    h = hashlib.sha256(line.encode())
    paths = ["track_a/recipe.py", "track_a/selection.csv", words[1]]
    if Path(words[1]).name == "check_submission.py":
        paths += ["submission/model.onnx", "submission/submission.json"]
    for path in paths:
        p = ROOT / path
        h.update(p.read_bytes() if p.is_file() else b"-")
    return h.hexdigest()[:16]


def log_name(words):
    tag = words[words.index("--tag") + 1] if "--tag" in words[:-1] else ""
    seed = f"_seed{words[words.index('--seed') + 1]}" if "--seed" in words[:-1] else ""
    stem = Path(words[1]).stem
    if stem == "curate" and len(words) > 2:
        stem += f"_{words[2]}"
    return f"{datetime.now():%m%d_%H%M%S}_{stem}" + (f"_{tag}{seed}" if tag else "") + ".txt"


def cmd_run(args):
    lines = queued()
    if not lines:
        print("colab_runs.txt is empty or missing: there is nothing to run. Ask your agent to queue "
              "the runs, push, and then run the notebook again.")
        return
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MAS1004_EXPERIMENTS": str(COLAB_EXPERIMENTS)}
    for n, line in enumerate(lines, start=1):
        words = shlex.split(line)
        if len(words) < 2 or words[0] != "python" or not words[1].startswith("src/"):
            print(f"[{n}/{len(lines)}] skipped, not a 'python src/...' command: {line}")
            continue
        key = run_key(line, words)
        if key in done:
            print(f"[{n}/{len(lines)}] already ran with this code: {line}")
            continue
        print(f"\n[{n}/{len(lines)}] {line}", flush=True)
        log = LOGS / log_name(words)
        with open(log, "w", encoding="utf-8") as out:
            out.write(f"$ {line}\n# commit {git('rev-parse', '--short', 'HEAD')}\n")
            proc = subprocess.Popen([sys.executable, "-u", *words[1:]], cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            for text in proc.stdout:
                print(text, end="", flush=True)
                out.write(text)
            proc.wait()
        if proc.returncode != 0:
            print(f"\nThis run failed (exit code {proc.returncode}). The runs after it were not started. "
                  f"The next cell still packs the output; give from_colab.zip to your agent.")
            return
        with open(DONE, "a") as f:
            f.write(key + "\n")
        done.add(key)
    print("\nAll queued runs are done.")


def cmd_pack(args):
    """results/ and submission/ into from_colab.zip, without the models in results/."""
    out = ROOT / ZIP_NAME
    out.unlink(missing_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in (RESULTS, ROOT / "submission"):
            for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
                # The grader trains Track A recipes itself, and a Track B model to hand in
                # is in submission/, so the models in results/ stay here.
                if path.is_file() and not (folder == RESULTS and path.suffix == ".onnx"):
                    z.write(path, path.relative_to(ROOT).as_posix())
                    count += 1
    print(f"packed {count} files into {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def find_zip(given):
    if given:
        path = Path(given).expanduser()
        if not path.is_file():
            sys.exit(f"There is no file {path}.")
        return path
    downloads = Path.home() / "Downloads"
    found = sorted(downloads.glob("from_colab*.zip"), key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"No from_colab*.zip in {downloads}. Ask the student where the browser saved it.")
    return found[-1]


def read_rows(text):
    return list(csv.DictReader(text.splitlines()))


def cmd_take(args):
    path = find_zip(args.zip)
    print(f"taking {path} ({datetime.fromtimestamp(path.stat().st_mtime):%d %b %H:%M})")
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts
               or not (n.startswith("results/") or n.startswith("submission/"))]
        if bad:
            sys.exit(f"{path.name} has files that do not belong to it: {bad[:3]}")
        copied, changed_submission = 0, False
        for name in names:
            if name == "results/colab_experiments.csv":
                continue
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if not dest.exists() or dest.read_bytes() != data:
                dest.write_bytes(data)
                copied += 1
                changed_submission |= name.startswith("submission/")
        print(f"{copied} new or changed files in results/ and submission/")
        if "submission/model.onnx" in names and changed_submission:
            info = json.loads(z.read("submission/submission.json")) if "submission/submission.json" in names else {}
            print(f"submission/ holds the Track B model from Colab: {info.get('summary', '')}")
        new_rows = read_rows(z.read("results/colab_experiments.csv").decode()) \
            if "results/colab_experiments.csv" in names else []
    old_rows = read_rows(EXPERIMENTS.read_text(encoding="utf-8")) if EXPERIMENTS.exists() else []
    seen = {(r.get("finished"), r.get("tag"), r.get("seed")) for r in old_rows}
    rows = [r for r in new_rows if (r.get("finished"), r.get("tag"), r.get("seed")) not in seen]
    if rows:
        new_file = not EXPERIMENTS.exists()
        fields = list(rows[0].keys()) if new_file else \
            next(csv.reader(EXPERIMENTS.read_text(encoding="utf-8").splitlines()))
        with open(EXPERIMENTS, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} new rows in experiments.csv" + (":" if rows else ""))
    for r in rows:
        print(f"  track {r['track']}  {r['tag']:<24} seed {r['seed'] or '-':<3} validation {r['val_score']}")
    logs = sorted(n for n in names if n.startswith("results/colab_logs/"))
    if logs:
        print(f"the output of every run is in results/colab_logs/, the newest {Path(logs[-1]).name}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("link").set_defaults(fn=cmd_link)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("pack").set_defaults(fn=cmd_pack)
    take = sub.add_parser("take")
    take.add_argument("zip", nargs="?", help="the zip file; the newest from_colab*.zip in Downloads if left out")
    take.set_defaults(fn=cmd_take)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
