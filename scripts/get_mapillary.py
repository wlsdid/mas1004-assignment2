"""Track B: street photos from Mapillary, sorted into data/mine/<city>/ by their GPS.

Mapillary photos are street-level photos people take with phones and car
cameras, under CC BY-SA 4.0. The starter's OSV-5M photos come from Mapillary
too, so photos already in the starter data are left out.

    uv run python scripts/get_mapillary.py --per-city 600

Needs a Mapillary access token (mapillary.com, Dashboard, Developers, register
an app, copy the client token) in data/mapillary_token.txt, which is not
pushed. The photos go to data/mine/<city>/mapillary_<id>.jpg, 512 pixels on
the shorter side like the starter photos, with data/mine/ATTRIBUTION.csv.

For each city it lists the photos in every box of a grid over the city, then
spreads its choice over the boxes, photographers and sequences, so that one
busy place (an airport) or one person does not fill a city.
"""

import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gps_sort  # noqa: E402
from rules import CITIES  # noqa: E402

OUT = ROOT / "data" / "mine"
TOKEN = ROOT / "data" / "mapillary_token.txt"
API = "https://graph.mapillary.com/images"
BOX = 0.02           # degrees on a side of each grid box
PER_SEQUENCE = 2     # the frames of one sequence look alike, so take few of each
THREADS = 16


def get_json(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError:
            time.sleep(2 + 3 * attempt)
    return {"data": []}


def starter_ids():
    """Mapillary ids of the OSV-5M photos in the starter data (osv5m_<id>.jpg)."""
    return {p.stem.removeprefix("osv5m_") for part in ("train", "val")
            for p in (ROOT / "data" / "starter" / part).glob("*/osv5m_*.jpg")}


def save(url, dest):
    with urllib.request.urlopen(url, timeout=60) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    s = 512 / min(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dest, quality=90)   # PIL writes no EXIF here, so no location is kept


def grid(city, boundaries):
    points = [p for outer, *_ in boundaries[city] for p in outer]
    lons, lats = [x for x, _ in points], [y for _, y in points]
    boxes = []
    lon = min(lons)
    while lon < max(lons):
        lat = min(lats)
        while lat < max(lats):
            if gps_sort.city_of(lat + BOX / 2, lon + BOX / 2, boundaries) == city:
                boxes.append((lon, lat))
            lat += BOX
        lon += BOX
    return boxes


def candidates(city, token, boundaries, skip):
    """Every usable photo Mapillary lists in the boxes of the city, with the box it is in."""
    fields = "id,thumb_1024_url,computed_geometry,geometry,sequence,is_pano,creator,captured_at"

    def one(box):
        lon, lat = box
        found = get_json({"access_token": token, "bbox": f"{lon},{lat},{lon + BOX},{lat + BOX}",
                          "fields": fields, "limit": 200}).get("data", [])
        out = []
        for img in found:
            if img["id"] in skip or img.get("is_pano") or not img.get("thumb_1024_url"):
                continue
            x, y = (img.get("computed_geometry") or img["geometry"])["coordinates"]
            if gps_sort.city_of(y, x, boundaries) == city:
                out.append({**img, "box": box})
        return out

    boxes = grid(city, boundaries)
    with ThreadPoolExecutor(THREADS) as pool:
        rows = [img for part in pool.map(one, boxes) for img in part]
    print(f"  {city}: {len(rows)} photos listed in {len(boxes)} boxes", flush=True)
    return rows


def choose(rows, n, rng):
    """Round-robin over the boxes, at most PER_SEQUENCE per sequence and n/20 per photographer."""
    by_box = defaultdict(list)
    for img in rows:
        by_box[img["box"]].append(img)
    for imgs in by_box.values():
        rng.shuffle(imgs)
    per_seq, per_creator = Counter(), Counter()
    max_creator = max(10, n // 20)
    chosen, boxes = [], list(by_box)
    rng.shuffle(boxes)
    while len(chosen) < n and boxes:
        for box in list(boxes):
            imgs = by_box[box]
            while imgs:
                img = imgs.pop()
                creator = (img.get("creator") or {}).get("username", "")
                if per_seq[img.get("sequence")] < PER_SEQUENCE and per_creator[creator] < max_creator:
                    per_seq[img.get("sequence")] += 1
                    per_creator[creator] += 1
                    chosen.append(img)
                    break
            if not imgs:
                boxes.remove(box)
            if len(chosen) >= n:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=600)
    args = ap.parse_args()
    if not TOKEN.exists():
        sys.exit(f"Put your Mapillary client token in {TOKEN} first.")
    token = TOKEN.read_text().strip()
    boundaries = gps_sort.load_boundaries()
    skip = starter_ids()
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    attribution = OUT / "ATTRIBUTION.csv"
    new = not attribution.exists()
    with open(attribution, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "city", "source", "mapillary_id", "sequence", "creator", "captured_at",
                        "license", "source_page"])
        for city in CITIES:
            folder = OUT / city
            folder.mkdir(exist_ok=True)
            have = {p.stem.removeprefix("mapillary_") for p in folder.glob("mapillary_*.jpg")}
            rows = [r for r in candidates(city, token, boundaries, skip) if r["id"] not in have]
            chosen = choose(rows, args.per_city - len(have), rng)

            def fetch(img):
                dest = folder / f"mapillary_{img['id']}.jpg"
                try:
                    save(img["thumb_1024_url"], dest)
                    return img, dest
                except OSError:
                    return img, None

            done = 0
            with ThreadPoolExecutor(THREADS) as pool:
                for img, dest in pool.map(fetch, chosen):
                    if dest is None:
                        continue
                    done += 1
                    w.writerow([f"{city}/{dest.name}", city, "mapillary", img["id"], img.get("sequence", ""),
                                (img.get("creator") or {}).get("username", ""), img.get("captured_at", ""),
                                "CC BY-SA 4.0", f"https://www.mapillary.com/app/?pKey={img['id']}"])
            f.flush()
            print(f"  {city}: {len(have) + done} photos in {folder} "
                  f"({len({r['box'] for r in chosen})} boxes, "
                  f"{len({(r.get('creator') or {}).get('username') for r in chosen})} photographers)",
                  flush=True)
    print(f"credits in {attribution}")


if __name__ == "__main__":
    main()
