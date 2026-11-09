"""Track B: street photos from Mapillary, sorted into data/mine/<city>/ by their GPS.

Mapillary photos are street-level photos people take with phones and car
cameras, under CC BY-SA 4.0. The starter's OSV-5M photos come from Mapillary
too, so photos already in the starter data are left out.

    uv run python scripts/get_mapillary.py --per-city 800

Needs a Mapillary access token (mapillary.com, Dashboard, Developers, register
an app, copy the client token) in data/mapillary_token.txt, which is not
pushed. The photos go to data/mine/<city>/mapillary_<id>.jpg, 512 pixels on
the shorter side like the starter photos, with data/mine/ATTRIBUTION.csv.
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
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gps_sort  # noqa: E402
from rules import CITIES  # noqa: E402

OUT = ROOT / "data" / "mine"
TOKEN = ROOT / "data" / "mapillary_token.txt"
API = "https://graph.mapillary.com/images"
BOX = 0.01           # degrees on a side of each random box searched
PER_SEQUENCE = 2     # the frames of one sequence look alike, so take few of each


def get_json(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError as e:
            if attempt == 2:
                print(f"  request failed: {e}")
                return {"data": []}
            time.sleep(3)


def starter_ids():
    """Mapillary ids of the OSV-5M photos in the starter data (osv5m_<id>.jpg)."""
    ids = set()
    for part in ("train", "val"):
        for p in (ROOT / "data" / "starter" / part).glob("*/osv5m_*.jpg"):
            ids.add(p.stem.removeprefix("osv5m_"))
    return ids


def save(url, dest):
    with urllib.request.urlopen(url, timeout=60) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    s = 512 / min(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dest, quality=90)   # PIL writes no EXIF here, so no location is kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=800)
    ap.add_argument("--max-boxes", type=int, default=600, help="random boxes to search per city at most")
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
            per_seq = Counter()
            points = [p for outer, *_ in boundaries[city] for p in outer]
            lons, lats = [x for x, _ in points], [y for _, y in points]
            boxes = 0
            while len(have) < args.per_city and boxes < args.max_boxes:
                boxes += 1
                lon, lat = rng.uniform(min(lons), max(lons) - BOX), rng.uniform(min(lats), max(lats) - BOX)
                if gps_sort.city_of(lat + BOX / 2, lon + BOX / 2, boundaries) != city:
                    continue
                found = get_json({"access_token": token, "bbox": f"{lon},{lat},{lon + BOX},{lat + BOX}",
                                  "fields": "id,thumb_1024_url,computed_geometry,geometry,sequence,"
                                            "is_pano,creator,captured_at", "limit": 50}).get("data", [])
                rng.shuffle(found)
                for img in found:
                    if len(have) >= args.per_city:
                        break
                    i, seq = img["id"], img.get("sequence", "")
                    if i in have or i in skip or img.get("is_pano") or not img.get("thumb_1024_url"):
                        continue
                    if per_seq[seq] >= PER_SEQUENCE:
                        continue
                    lon_i, lat_i = (img.get("computed_geometry") or img["geometry"])["coordinates"]
                    if gps_sort.city_of(lat_i, lon_i, boundaries) != city:
                        continue
                    dest = folder / f"mapillary_{i}.jpg"
                    try:
                        save(img["thumb_1024_url"], dest)
                    except OSError as e:
                        print(f"  {i}: {e}")
                        continue
                    have.add(i)
                    per_seq[seq] += 1
                    w.writerow([f"{city}/{dest.name}", city, "mapillary", i, seq,
                                (img.get("creator") or {}).get("username", ""), img.get("captured_at", ""),
                                "CC BY-SA 4.0", f"https://www.mapillary.com/app/?pKey={i}"])
                print(f"\r  {city}: {len(have)}/{args.per_city} photos, {boxes} boxes searched", end="", flush=True)
                f.flush()
            print()
    print(f"photos in {OUT}; credits in {attribution}")


if __name__ == "__main__":
    main()
