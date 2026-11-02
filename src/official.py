"""Track A: train the fixed model on your selection with your recipe, as the grader will.

This file is complete. Do not change it. The grader runs this same file on its
own computer with your track_a/recipe.py and track_a/selection.csv, once for
each seed in rules.TRACK_A_SEEDS, and scores the models on the leaderboard
photos and, after the deadline, on the final photos. Your score is the mean of
the three.

    python src/official.py --tag baseline            one full run, seed 0
    python src/official.py --tag baseline --seed 1   the same with seed 1
    python src/official.py --quick                   20 steps, to see that the recipe runs

What is fixed here: the model (ResNet18 with the ImageNet weights and a new
last layer for the five cities), the number of photos that go through it in
training (rules.TRACK_A_BUDGET), the image size, mixed precision, and how the
model is measured and saved. What your recipe decides is in track_a/recipe.py.

Every run writes results/<tag>_seed<seed>/ (model.onnx, result.json) and,
except with --quick, adds one row to experiments.csv, with the score on the
public validation photos in data/starter/val, measured the way the grader
measures the test photos.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RECIPE_DIR = Path(os.environ.get("MAS1004_RECIPE_DIR", ROOT / "track_a"))
sys.path.insert(0, str(RECIPE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

import rules  # noqa: E402
from export import export_onnx  # noqa: E402
from rules import CITIES, MEAN, STD  # noqa: E402

HOOKS = ["BATCH_SIZE", "train_transform", "sample_weights", "make_optimizer", "make_scheduler",
         "mix", "loss_fn", "on_step", "final_model"]

# A recipe works only with the photos official.py gives it. It may not read
# files, use the network, start programs, or bring in another model, so it
# may import only these, and may not use the names below.
ALLOWED_IMPORTS = ("torch", "torchvision.transforms", "numpy", "math", "random", "PIL", "functools",
                   "itertools", "collections", "copy", "typing", "dataclasses")
DENIED_IMPORTS = ("torch.hub", "torch.utils", "torch.jit", "torch.package", "torch.serialization",
                  "torch.distributed", "torch.multiprocessing", "torch.cuda.memory", "torch.onnx")
DENIED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "load", "hub", "from_file", "fromfile", "models",
                "datasets", "io", "save", "download", "system", "popen", "urlopen", "__subclasses__", "__globals__",
                "__builtins__", "__code__", "__dict__", "__loader__", "__spec__"}
MAX_RECIPE_BYTES = 50_000


def recipe_problems(path):
    """The reasons a recipe file breaks the rules above, or an empty list."""
    import ast
    source = Path(path).read_text(encoding="utf-8")
    if len(source.encode()) > MAX_RECIPE_BYTES:
        return [f"recipe.py is longer than {MAX_RECIPE_BYTES // 1000} kB"]
    problems = []
    for node in ast.walk(ast.parse(source)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            modules += [f"{node.module}.{a.name}" for a in node.names]
        for m in modules:
            denied = any(m == d or m.startswith(d + ".") for d in DENIED_IMPORTS)
            allowed = m == "torchvision" or any(m == a or m.startswith(a + ".") for a in ALLOWED_IMPORTS)
            if denied or not allowed:
                problems.append(f"line {node.lineno}: import of {m}")
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in DENIED_NAMES:
            problems.append(f"line {node.lineno}: {name}")
    return sorted(set(problems), key=lambda p: int(p.split(":")[0].split()[1]))


def install_guard(forbidden):
    """From now on, no network, no other programs, and no reading of the forbidden folders."""
    forbidden = {q for f in forbidden if f for q in (Path(os.path.abspath(f)), Path(f).resolve())}

    def hook(event, args):
        if event in ("socket.connect", "socket.sendto") and isinstance(args[1], tuple):
            raise PermissionError("recipes may not use the network")
        if event == "socket.getaddrinfo":
            raise PermissionError("recipes may not use the network")
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"):
            raise PermissionError("recipes may not start other programs")
        if event == "open" and forbidden and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            for p in (Path(os.path.abspath(name)), Path(name).resolve()):
                if any(p == f or f in p.parents for f in forbidden):
                    raise PermissionError(f"recipes may not read {p}")

    sys.addaudithook(hook)


def build_model(seed):
    """ResNet18 with the ImageNet weights and a new last layer of 5. Not up to the recipe."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch.manual_seed(seed)
    net.fc = nn.Linear(net.fc.in_features, len(CITIES))
    return net


def read_selection(data_dir, recipe_dir):
    """(path, city) for every row of selection.csv, or for every training photo if there is none."""
    train_dir = Path(data_dir) / "train"
    selection = Path(recipe_dir) / "selection.csv"
    if not selection.exists():
        return [(p, c) for p, c in rules.list_photos(train_dir)], "all training photos"
    rows = []
    with open(selection, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            rel = (row.get("path") or "").strip().replace("\\", "/")
            city = rel.split("/")[0]
            path = train_dir / rel
            inside = train_dir.resolve() in path.resolve().parents
            if city not in CITIES or not inside or not path.is_file():
                raise SystemExit(f"selection.csv line {i}: {rel!r} is not a photo in data/starter/train/<city>/.")
            rows.append((path, city))
    if not rows:
        raise SystemExit("selection.csv has no rows. It needs a column called 'path'.")
    return rows, f"selection.csv, {len(rows)} rows"


class TrainPhotos(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, city = self.rows[i]
        out = self.transform(rules.load_photo(path))
        if not torch.is_tensor(out):
            out = TF.to_tensor(out)
        size = rules.TRACK_A_IMAGE_SIZE
        if tuple(out.shape) != (3, size, size):
            raise ValueError(f"train_transform gave shape {tuple(out.shape)}, not (3, {size}, {size}).")
        return (out.float() - self.mean) / self.std, CITIES.index(city)


class GraderPhotos(Dataset):
    """Photos prepared exactly as the grader prepares the test photos, then normalised."""

    def __init__(self, rows, size):
        self.rows, self.size = rows, size
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, city = self.rows[i]
        x = torch.from_numpy(rules.grader_input(rules.load_photo(path), self.size))
        return (x - self.mean) / self.std, CITIES.index(city)


def draw_order(n_photos, n_draws, weights, seed):
    """Which photo goes into each place of training, decided before training starts."""
    g = torch.Generator().manual_seed(seed)
    if weights is None:
        rounds = -(-n_draws // n_photos)
        order = torch.cat([torch.randperm(n_photos, generator=g) for _ in range(rounds)])
    else:
        w = torch.as_tensor(weights, dtype=torch.float64)
        if w.shape != (n_photos,) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("sample_weights must give one non-negative number for every row of the selection.")
        order = torch.multinomial(w, n_draws, replacement=True, generator=g)
    return order[:n_draws].tolist()


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def measure(model, rows, size, device, workers):
    """Balanced accuracy of a PyTorch model on rows, with the grader's preparation."""
    model.eval()
    loader = DataLoader(GraderPhotos(rows, size), batch_size=128, num_workers=workers)
    true, pred = [], []
    for x, y in loader:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred.append(model(x).argmax(1).cpu())
        true.append(y)
    return rules.balanced_accuracy(torch.cat(true).numpy(), torch.cat(pred).numpy())


def same_architecture(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa)


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()
    except OSError:
        return ""


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="run", help="a name for this run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="only 20 steps and 200 validation photos")
    ap.add_argument("--data", default=str(ROOT / "data" / "starter"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--budget", type=int, default=rules.TRACK_A_BUDGET,
                    help="for trying things out only; the grader always uses rules.TRACK_A_BUDGET")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds; the grader stops a run that takes longer")
    ap.add_argument("--no-log", action="store_true", help="do not add a row to experiments.csv")
    ap.add_argument("--skip-val", action="store_true", help="do not measure on the validation photos")
    args = ap.parse_args()

    problems = recipe_problems(RECIPE_DIR / "recipe.py")
    if problems:
        raise SystemExit("recipe.py breaks the Track A rules (see the top of official.py):\n  "
                         + "\n  ".join(problems))
    reference = build_model(args.seed)  # downloads the ImageNet weights the first time
    commit = git_commit()  # the guard below stops git too
    install_guard(os.environ.get("MAS1004_FORBIDDEN", "").split(os.pathsep))
    import recipe
    missing = [h for h in HOOKS if not hasattr(recipe, h)]
    if missing:
        raise SystemExit(f"recipe.py is missing {', '.join(missing)}. The starter recipe has all of them.")
    batch = int(recipe.BATCH_SIZE)
    lo, hi = rules.TRACK_A_BATCH_SIZES
    if not lo <= batch <= hi:
        raise SystemExit(f"BATCH_SIZE must be between {lo} and {hi}, it is {batch}.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    torch.backends.cudnn.benchmark = True

    rows, where = read_selection(args.data, RECIPE_DIR)
    budget = 20 * batch if args.quick else args.budget
    steps = budget // batch
    weights = recipe.sample_weights([c for _, c in rows],
                                    [p.relative_to(Path(args.data) / "train").as_posix() for p, _ in rows])
    order = draw_order(len(rows), steps * batch, weights, args.seed)
    loader = DataLoader(
        TrainPhotos(rows, recipe.train_transform()), batch_size=batch, sampler=order, drop_last=True,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda", persistent_workers=False)

    model = build_model(args.seed).to(device).to(memory_format=torch.channels_last)
    optimizer = recipe.make_optimizer(model)
    scheduler = recipe.make_scheduler(optimizer, steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    counts = {c: sum(1 for _, city in rows if city == c) for c in CITIES}
    print(f"Track A, seed {args.seed}, on {device}: {where} ({', '.join(f'{c} {n}' for c, n in counts.items())})")
    print(f"{steps} steps of {batch} photos = {steps * batch:,} photos through the network", flush=True)

    start = time.time()
    seen = 0
    running = []
    model.train()
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        x, target = recipe.mix(x, y)
        if tuple(x.shape) != (batch, 3, rules.TRACK_A_IMAGE_SIZE, rules.TRACK_A_IMAGE_SIZE):
            raise SystemExit(f"mix must return images of shape {(batch, 3, 224, 224)}, it gave {tuple(x.shape)}.")
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = recipe.loss_fn(logits.float(), target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        recipe.on_step(step, steps, model, optimizer)
        seen += len(y)
        running.append(loss.item())
        if (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            print(f"  step {step + 1:5d}/{steps}  loss {np.mean(running):.3f}  ({time.time() - start:.0f}s)", flush=True)
            running = []
        if args.time_limit and time.time() - start > args.time_limit:
            raise SystemExit(f"Stopped: training took longer than {args.time_limit:.0f} seconds.")
    train_seconds = time.time() - start
    if seen != steps * batch:
        raise SystemExit(f"Only {seen} photos were drawn, not {steps * batch}.")

    final = recipe.final_model(model)
    if not same_architecture(final, reference):
        raise SystemExit(f"final_model must return the same ResNet18 with {len(CITIES)} outputs.")
    final = final.to(device).to(memory_format=torch.channels_last)

    val_rows = [] if args.skip_val else rules.list_photos(Path(args.data) / "val")
    all_val = len(val_rows)
    if args.quick:
        val_rows = val_rows[::max(1, len(val_rows) // 200)]
    val_start = time.time()
    score, per_city, confusion = (measure(final, val_rows, rules.TRACK_A_IMAGE_SIZE, device, args.workers)
                                  if val_rows else (float("nan"), {}, np.zeros((len(CITIES), len(CITIES)), dtype=int)))
    val_seconds = time.time() - val_start

    out_dir = Path(args.out) / f"{args.tag}_seed{args.seed}"
    gap = export_onnx(final, out_dir / "model.onnx", rules.TRACK_A_IMAGE_SIZE)
    official = budget == rules.TRACK_A_BUDGET and not args.quick
    result = {
        "track": "A", "tag": args.tag, "seed": args.seed, "official_budget": official,
        "budget": steps * batch, "batch_size": batch, "steps": steps, "selection": where,
        "selection_counts": counts, "device": str(device), "train_seconds": round(train_seconds, 1),
        "val_score": round(score, 4), "val_per_city": {c: round(v, 4) for c, v in per_city.items()},
        "val_confusion": confusion.tolist(), "onnx_gap": gap,
        "recipe_sha": file_hash(RECIPE_DIR / "recipe.py"), "selection_sha": file_hash(RECIPE_DIR / "selection.csv"),
        "commit": commit, "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"\nvalidation score {score:.3f}  (" + ", ".join(f"{c} {v:.3f}" for c, v in per_city.items()) + ")")
    print(f"training took {train_seconds / 60:.1f} min; wrote {out_dir / 'model.onnx'}")
    if not official:
        print("This was not an official run (--quick or a different --budget). The grader's numbers will differ.")
    if args.quick:
        # Starting up counts as if it were training, so this errs on the long side.
        train_min = train_seconds / seen * rules.TRACK_A_BUDGET / 60
        val_min = val_seconds / max(1, len(val_rows)) * all_val / 60
        print(f"A full run on this computer ({device}) would take about {train_min + val_min:.0f} minutes: "
              f"{train_min:.0f} for training and {val_min:.0f} for the validation photos.")
    if not args.no_log and not args.quick:
        log = Path(os.environ.get("MAS1004_EXPERIMENTS", ROOT / "experiments.csv"))
        fields = ["finished", "track", "tag", "seed", "official_budget", "budget", "val_score", *CITIES,
                  "train_seconds", "device", "selection", "recipe_sha", "selection_sha", "commit"]
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({**result, **result["val_per_city"]})


if __name__ == "__main__":
    main()
