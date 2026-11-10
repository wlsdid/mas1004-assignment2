"""Track B: a place to start. Train any small model on any photos and write a submission.

Change anything in this file, or replace it. In Track B only the submission
counts: submission/model.onnx and submission/submission.json, checked by
src/check_submission.py. What must stay true is in README.md: at most 12
million parameters and 2.0 billion multiply-adds per photo.

    python src/train_b.py --tag b1                                  ResNet18 on the starter photos
    python src/train_b.py --tag b2 --folders data/starter/train data/mine
    python src/train_b.py --tag b3 --model mobilenet_v3_large --size 320 --epochs 8 --balance
    python src/train_b.py --tag b3 --submit                         also write submission/

Every folder holds <city>/ subfolders of photos. Photos in data/starter/val are
used to measure, so do not train on them if you want that number to mean
anything. Every run adds a row to experiments.csv.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from torchvision.io import ImageReadMode, decode_jpeg, read_file  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
import random  # noqa: E402

import rules  # noqa: E402
from export import export_onnx  # noqa: E402
from rules import CITIES, MEAN, STD  # noqa: E402

# ImageNet-1k weights from torchvision. Other starting weights work too; say
# which in submission.json.
MODELS = {
    "resnet18": ("resnet18", "IMAGENET1K_V1"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "IMAGENET1K_V2"),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1"),
    "efficientnet_b1": ("efficientnet_b1", "IMAGENET1K_V2"),
    "efficientnet_b3": ("efficientnet_b3", "IMAGENET1K_V1"),
    "regnet_y_800mf": ("regnet_y_800mf", "IMAGENET1K_V2"),
    "regnet_y_1_6gf": ("regnet_y_1_6gf", "IMAGENET1K_V2"),
    "shufflenet_v2_x2_0": ("shufflenet_v2_x2_0", "IMAGENET1K_V1"),
}


def build(name):
    fn, weights = MODELS[name]
    net = getattr(models, fn)(weights=weights)
    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, len(CITIES))
    else:
        last = net.classifier[-1]
        net.classifier[-1] = nn.Linear(last.in_features, len(CITIES))
    return net, f"torchvision {fn} {weights}"


class Photos(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, city = self.rows[i]
        return self.transform(rules.load_photo(path)), CITIES.index(city)


class RawPhotos(Dataset):
    """Photos as JPEG bytes, to be decoded on the GPU (see gpu_batch).

    Colab has two CPU cores, and decoding photos there takes most of the time
    of a run. A photo that is not a plain JPEG, or has to be turned upright,
    is opened here the way rules.load_photo opens it instead.
    """

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, city = self.rows[i]
        with Image.open(path) as im:
            plain = im.format == "JPEG" and im.mode in ("RGB", "L") and im.getexif().get(0x0112, 1) == 1
        # numpy arrays, not tensors: a worker hands tensors over through one
        # file descriptor each, and a batch of 32 separate tensors runs out of
        # them on Colab ("OSError: Bad file descriptor").
        if plain:
            return np.fromfile(path, dtype=np.uint8), CITIES.index(city), True
        return np.asarray(rules.load_photo(path).convert("RGB")), CITIES.index(city), False


def collate_raw(batch):
    return [b[0] for b in batch], torch.tensor([b[1] for b in batch]), [b[2] for b in batch]


def gpu_batch(data, is_jpeg, size, device, flip):
    """Decode a batch on the GPU and apply the same changes as the CPU transform."""
    jpegs = [torch.from_numpy(d) for d, j in zip(data, is_jpeg) if j]
    try:
        decoded = decode_jpeg(jpegs, mode=ImageReadMode.RGB, device=device) if jpegs else []
    except RuntimeError:  # a JPEG the GPU decoder cannot read: decode this batch on the CPU
        decoded = [d.to(device) for d in decode_jpeg(jpegs, mode=ImageReadMode.RGB)]
    decoded = iter(decoded)
    jitter = T.ColorJitter(0.2, 0.2, 0.2)
    out = []
    for d, j in zip(data, is_jpeg):
        img = next(decoded) if j else torch.from_numpy(d).permute(2, 0, 1).to(device)
        top, left, h, w = T.RandomResizedCrop.get_params(img, (0.35, 1.0), (3 / 4, 4 / 3))
        img = TF.resized_crop(img, top, left, h, w, [size, size], antialias=True)
        img = jitter(img.float() / 255)
        if flip and random.random() < 0.5:
            img = img.flip(-1)
        out.append(img)
    return TF.normalize(torch.stack(out), MEAN, STD)


@torch.no_grad()
def measure(net, rows, size, device, workers):
    net.eval()
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    true, pred = [], []
    for x, y in DataLoader(Photos(rows, tf), batch_size=128, num_workers=workers):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred.append(net(x.to(device)).argmax(1).cpu())
        true.append(y)
    return rules.balanced_accuracy(torch.cat(true).numpy(), torch.cat(pred).numpy())


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()
    except OSError:
        return ""


def shown(folder):
    """A folder as written in submission.json: relative to the repository when it is inside it."""
    path = Path(folder).absolute()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="resnet18")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--folders", nargs="+", default=[str(ROOT / "data" / "starter" / "train")])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--balance", action="store_true", help="draw every city equally often")
    ap.add_argument("--no-flip", action="store_true",
                    help="no left-right flips: they mirror writing and the side of the road cars drive on")
    ap.add_argument("--gpu-decode", action="store_true",
                    help="decode and change the photos on the GPU: much quicker on Colab's two CPU cores")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--submit", action="store_true", help="also copy the model into submission/")
    ap.add_argument("--val-every", type=int, default=1,
                    help="measure on every n-th validation photo only (quicker without a GPU)")
    ap.add_argument("--stop-now", action="store_true",
                    help="train no further: hand in the model of the last finished epoch in the checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = [r for folder in args.folders for r in rules.list_photos(folder)]
    for folder in map(Path, args.folders):
        if not folder.is_dir():
            print(f"{folder} does not exist; nothing from it is used")
            continue
        for d in sorted(folder.iterdir()):
            if d.is_dir() and d.name.lower() not in CITIES:
                print(f"{d} is not one of the five cities ({', '.join(CITIES)}); its photos are not used")
    counts = {c: sum(1 for _, x in rows if x == c) for c in CITIES}
    print(f"{len(rows)} photos ({', '.join(f'{c} {n}' for c, n in counts.items())}) on {device}")

    flip = [] if args.no_flip else [T.RandomHorizontalFlip()]
    tf = T.Compose([T.RandomResizedCrop(args.size, scale=(0.35, 1.0)), *flip,
                    T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    gpu_decode = args.gpu_decode and device.type == "cuda"
    data = RawPhotos(rows) if gpu_decode else Photos(rows, tf)
    extra = {"collate_fn": collate_raw} if gpu_decode else {}
    if args.balance:
        weights = [1.0 / counts[c] for _, c in rows]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(rows), replacement=True)
        loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, drop_last=True, **extra)
    else:
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, **extra)

    net, weights_name = build(args.model)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # After every epoch the training state goes to results/<tag>/checkpoint.pt
    # (in Google Drive on Colab). A run that broke off carries on from the last
    # finished epoch when it is started again with the same settings.
    out = ROOT / "results" / args.tag
    ckpt = out / "checkpoint.pt"
    settings = {k: v for k, v in vars(args).items() if k not in ("workers", "submit", "val_every", "stop_now")}
    first, seconds_before = 0, 0.0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if state["settings"] == settings:
            net.load_state_dict(state["net"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            first, seconds_before = state["epoch"], state["seconds"]
            print(f"carrying on from {ckpt}, after epoch {first}", flush=True)
        else:
            print(f"{ckpt} is from other settings; starting from the beginning", flush=True)
    if args.stop_now and first == 0:
        sys.exit(f"--stop-now needs a checkpoint of these settings in {ckpt}")
    epochs_done = first if args.stop_now else args.epochs
    if args.stop_now:
        print(f"--stop-now: handing in the model after epoch {first} of {args.epochs}", flush=True)
    start = time.time()
    for epoch in range(first, epochs_done):
        net.train()
        losses = []
        for batch in loader:
            if gpu_decode:
                x = gpu_batch(batch[0], batch[2], args.size, device, not args.no_flip)
                y = batch[1].to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(net(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        elapsed = seconds_before + time.time() - start
        print(f"epoch {epoch + 1}/{args.epochs}: loss {np.mean(losses):.3f} ({elapsed:.0f}s)", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"settings": settings, "epoch": epoch + 1, "seconds": elapsed, "net": net.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict()}, tmp)
        tmp.replace(ckpt)
    train_seconds = seconds_before + time.time() - start

    val = rules.list_photos(ROOT / "data" / "starter" / "val")[::args.val_every]
    score, per_city, _ = measure(net, val, args.size, device, args.workers)
    print(f"validation score {score:.3f} (" + ", ".join(f"{c} {v:.3f}" for c, v in per_city.items()) + ")")

    export_onnx(net, out / "model.onnx", args.size)
    info = {"image_size": args.size, "starting_weights": weights_name,
            "data": " + ".join(shown(f) for f in args.folders), "summary": f"{args.model} at {args.size}, "
            + (f"{epochs_done} of {args.epochs} epochs" if epochs_done < args.epochs else f"{args.epochs} epochs")
            + ", cities balanced" * args.balance + ", no flips" * args.no_flip}
    (out / "submission.json").write_text(json.dumps(info, indent=2))
    m = rules.measure_onnx(out / "model.onnx", args.size)
    print(f"wrote {out / 'model.onnx'}: {m['parameters'] / 1e6:.2f} million parameters, {m['gmacs']:.2f} billion multiply-adds")
    if args.submit:
        (ROOT / "submission").mkdir(exist_ok=True)
        for name in ("model.onnx", "submission.json"):
            shutil.copy(out / name, ROOT / "submission" / name)
        print("copied to submission/. Check it with: python src/check_submission.py")

    log = Path(os.environ.get("MAS1004_EXPERIMENTS", ROOT / "experiments.csv"))
    fields = ["finished", "track", "tag", "seed", "official_budget", "budget", "val_score", *CITIES,
              "train_seconds", "device", "selection", "recipe_sha", "selection_sha", "commit"]
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"finished": datetime.now().isoformat(timespec="seconds"), "track": "B", "tag": args.tag,
                    "budget": epochs_done * len(loader) * args.batch_size, "val_score": round(score, 4),
                    **{c: round(per_city.get(c, 0), 4) for c in CITIES}, "train_seconds": round(train_seconds, 1),
                    "device": str(device), "selection": f"{args.model} {args.size}px; " + " + ".join(shown(f) for f in args.folders),
                    "commit": git_commit()})
    if epochs_done == args.epochs:
        ckpt.unlink(missing_ok=True)  # finished: the model is in model.onnx


if __name__ == "__main__":
    main()
