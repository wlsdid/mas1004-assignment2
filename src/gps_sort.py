"""Sort your own photos into the five cities by where they were taken.

This file is complete.

    python src/gps_sort.py ~/Pictures/busan_trip
    python src/gps_sort.py ~/Downloads/no_gps --city taipei

Most phone photos carry the place they were taken (GPS) in their EXIF data.
For every photo in the folder this reads that place, finds which city's
boundary it falls in (src/cities.geojson, the same boundaries the starter
data used), and saves a copy in data/mine/<city>/. The copy is turned upright
and has no EXIF data left in it: no place, no time, no phone model. Your
originals are not changed.

A copy is named after the folder and the original file, with a short code
for the original's place on your computer, so that two photos never get the
same name and running this again replaces the same copies. Photos without GPS,
or taken outside the five cities, are listed at the end and not copied. If
you know where photos without GPS were taken, run again with --city.

Photos must be JPEG or PNG. iPhones save HEIC unless Settings, Camera,
Formats is set to Most Compatible; ask your agent to convert HEIC first.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["seoul", "busan", "tokyo", "osaka", "taipei"]
GPS_IFD = 0x8825


def load_boundaries():
    data = json.loads((Path(__file__).with_name("cities.geojson")).read_text())
    out = {}
    for f in data["features"]:
        g = f["geometry"]
        polygons = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out[f["properties"]["city"]] = polygons
    return out


def inside_ring(lon, lat, ring):
    """Ray casting: does a horizontal ray from the point cross the ring an odd number of times?"""
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat):
            x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if x > lon:
                hit = not hit
    return hit


def city_of(lat, lon, boundaries):
    for city, polygons in boundaries.items():
        for outer, *holes in polygons:
            if inside_ring(lon, lat, outer) and not any(inside_ring(lon, lat, h) for h in holes):
                return city
    return None


def gps_of(image):
    """(latitude, longitude) from the EXIF data, or None."""
    gps = image.getexif().get_ifd(GPS_IFD)
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def degrees(values, ref):
        d, m, s = (float(v) for v in values)
        value = d + m / 60 + s / 3600
        return -value if ref in ("S", "W") else value

    try:
        return degrees(gps[2], gps.get(1, "N")), degrees(gps[4], gps.get(3, "E"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "photo"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--city", choices=CITIES, help="the city of every photo in the folder, for photos without GPS")
    ap.add_argument("--out", default=str(ROOT / "data" / "mine"))
    args = ap.parse_args()

    boundaries = load_boundaries()
    folder = Path(args.folder).expanduser()
    out = Path(args.out)
    copied, no_gps, outside, unreadable = {c: 0 for c in CITIES}, [], [], []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            if path.suffix.lower() in (".heic", ".heif"):
                unreadable.append(path.name)
            continue
        try:
            with Image.open(path) as image:
                place = gps_of(image)
                upright = ImageOps.exif_transpose(image).convert("RGB")
        except OSError:
            unreadable.append(path.name)
            continue
        if args.city:
            city = args.city
        elif place is None:
            no_gps.append(path.name)
            continue
        else:
            city = city_of(*place, boundaries)
            if city is None:
                outside.append(path.name)
                continue
        code = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]
        dest = out / city / f"{slug(folder.name)}_{slug(path.stem)}_{code}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        upright.save(dest, "JPEG", quality=92)  # no exif= argument: the copy has no EXIF data
        copied[city] += 1

    print("copied: " + ", ".join(f"{c} {n}" for c, n in copied.items()))
    for label, names in (("no GPS (run again with --city if you know where)", no_gps),
                         ("outside the five cities", outside),
                         ("could not read (HEIC or broken)", unreadable)):
        if names:
            print(f"{label}: {len(names)}, e.g. {', '.join(names[:5])}")


if __name__ == "__main__":
    main()
