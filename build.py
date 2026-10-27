#!/usr/bin/env python3
"""build.py - run the whole pothole pipeline for ONE uploaded photo.

Order (same as running the scripts by hand):
    solar_shadow_update.py -> pothole_pipeline.py -> pothole_volume.py
    -> pothole_depth.py -> pothole_report.py

Everything happens inside a private job folder, so jobs never overwrite each other:
    <job_dir>/images/image1.jpg   the upload
    <job_dir>/main.xlsx           written from the user's form values
    <job_dir>/pothole information.csv, processed/, area_debug/, depth_debug/
    <job_dir>/report.pdf          pothole image + volume   (the result)
    <job_dir>/result.csv          full numbers
    <job_dir>/log.txt             output of every step

Command line:
    python build.py --image photo.jpg --latitude 23.02 --longitude 72.57 \
        --direction 90 --date 15-03-2026 --time 11:30 --height 1.5 \
        --weights weights/best.pt --fov-deg 65 --job-dir jobs/test1

Server settings (also used by app.py) come from arguments here, and from environment
variables in app.py: WEIGHTS, FOV_DEG, SHADOW_FRACTION.
"""
import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
IMAGE_NAME = "image1"                      # "image no" used in main.xlsx
STEP_TIMEOUT_S = 900


class JobError(Exception):
    """A problem the user can be told about."""


class NoPotholeFound(JobError):
    pass


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
def validate_metadata(m):
    """Return cleaned values or raise ValueError with a readable message."""
    def number(key, label, lo=None, hi=None):
        try:
            v = float(str(m[key]).strip())
        except (KeyError, ValueError):
            raise ValueError(f"{label} must be a number")
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must be a number")
        if lo is not None and not lo <= v <= hi:
            raise ValueError(f"{label} must be between {lo:g} and {hi:g}")
        return v

    lat = number("latitude", "Latitude", -90, 90)
    lon = number("longitude", "Longitude", -180, 180)
    direction = number("direction", "Direction", -360, 360)
    height = number("height", "Capture height (m)")
    if not 0 < height <= 200:
        raise ValueError("Capture height must be greater than 0 and at most 200 m")

    raw_date = str(m.get("date", "")).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            date = dt.datetime.strptime(raw_date, fmt)
            break
        except ValueError:
            date = None
    if date is None:
        raise ValueError("Date must look like DD-MM-YYYY")

    raw_time = str(m.get("time", "")).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t = dt.datetime.strptime(raw_time, fmt)
            break
        except ValueError:
            t = None
    if t is None:
        raise ValueError("Time must look like HH:MM (24-hour)")

    return {"latitude": lat, "longitude": lon, "direction": direction, "height": height,
            "date": date.strftime("%d-%m-%Y"), "time": t.strftime("%H:%M")}


def write_metadata(job_dir, m):
    pd.DataFrame([{
        "image no": IMAGE_NAME,
        "latitude(N)": m["latitude"],
        "longitude(E)": m["longitude"],
        "gps Direction(True North)": m["direction"],
        "time(hr:min)": m["time"],
        "date": m["date"],
        "height of photo capture(Meters)": m["height"],
    }]).to_excel(job_dir / "main.xlsx", index=False)


# --------------------------------------------------------------------------
# Running the scripts
# --------------------------------------------------------------------------
def run_step(title, script, args, job_dir):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    try:
        p = subprocess.run(cmd, cwd=job_dir, capture_output=True, text=True,
                           timeout=STEP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with open(job_dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== {title}: TIMED OUT ===\n")
        raise JobError(f"{title} took too long and was stopped")
    with open(job_dir / "log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== {title} (exit {p.returncode}) ===\n{p.stdout}\n{p.stderr}\n")
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        raise JobError(f"{title} failed: {tail[-1] if tail else 'no details'}")
    return p.stdout


def _table(job_dir):
    return pd.read_csv(job_dir / "pothole information.csv", encoding="utf-8-sig")


def run_job(job_dir, image_path, metadata, weights, fov_deg=65.0, shadow_fraction=0.6,
            progress=None):
    """Run everything for one photo. Returns a dict describing the result.

    progress: optional callable(text) told which step is starting.
    Raises ValueError (bad inputs) or JobError / NoPotholeFound (processing problems).
    """
    say = progress or (lambda _t: None)
    job_dir = Path(job_dir).resolve()
    meta = validate_metadata(metadata)
    if weights is None or not Path(weights).is_file():
        raise JobError(f"Model weights not found: {weights}")
    weights = str(Path(weights).resolve())

    job_dir.mkdir(parents=True, exist_ok=True)
    images = job_dir / "images"
    images.mkdir(exist_ok=True)
    image_path = Path(image_path)
    for old in images.iterdir():
        old.unlink()
    shutil.copy2(image_path, images / (IMAGE_NAME + image_path.suffix.lower()))
    write_metadata(job_dir, meta)

    say("Calculating sun position")
    run_step("Sun position", "solar_shadow_update.py", ["main.xlsx"], job_dir)

    say("Finding the pothole")
    run_step("Pothole detection", "pothole_pipeline.py",
             ["--images-dir", "images", "--excel", "main.xlsx", "--weights", weights], job_dir)
    t = _table(job_dir)
    if "Crop File" not in t.columns or t["Crop File"].fillna("").astype(str).str.strip().eq("").all():
        raise NoPotholeFound("No pothole was detected in this photo")

    say("Outlining the pothole")
    run_step("Pothole outline", "pothole_volume.py",
             ["--weights", weights, "--area-mode", "auto"], job_dir)
    t = _table(job_dir)
    if "Mask File" not in t.columns or t["Mask File"].fillna("").astype(str).str.strip().eq("").all():
        raise JobError("The pothole outline could not be found")

    say("Measuring shadow and depth")
    run_step("Depth and volume", "pothole_depth.py",
             ["--weights", weights, "--fov-deg", fov_deg, "--shadow-fraction", shadow_fraction],
             job_dir)
    t = _table(job_dir)
    vols = pd.to_numeric(t.get("Volume (litres)"), errors="coerce")
    if vols is None or vols.notna().sum() == 0:
        flag_set = {f for v in t.get("Flags", pd.Series(dtype=str)).dropna()
                    for f in str(v).split(";") if f}
        if any(f.startswith("no_shadow_ratio") for f in flag_set):
            raise JobError("The sun was below the horizon at the date, time and place entered, "
                           "so no shadow depth can be worked out. Check the date, time and longitude.")
        if "no_shadow_found" in flag_set or "shadow_outside_pothole_mask" in flag_set:
            raise JobError("No usable shadow was found inside the pothole, so depth could not be measured")
        if "no_scale" in flag_set:
            raise JobError("Scale could not be worked out. Check the capture height")
        raise JobError("Volume could not be worked out (" + "; ".join(sorted(flag_set)) + ")")

    say("Building the report")
    run_step("Report", "pothole_report.py", ["--out", "report.pdf"], job_dir)
    shutil.copy2(job_dir / "pothole information.csv", job_dir / "result.csv")

    return {"pdf": str(job_dir / "report.pdf"), "csv": str(job_dir / "result.csv"),
            "potholes": int(vols.notna().sum()),
            "volume_litres": [round(float(v), 2) for v in vols.dropna()]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--latitude", required=True)
    ap.add_argument("--longitude", required=True)
    ap.add_argument("--direction", required=True,
                    help="angle of True North in the photo (same meaning as gps Direction(True North) in main.xlsx)")
    ap.add_argument("--date", required=True, help="DD-MM-YYYY")
    ap.add_argument("--time", required=True, help="HH:MM, 24-hour")
    ap.add_argument("--height", required=True, help="capture height in metres")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--fov-deg", type=float, default=65.0)
    ap.add_argument("--shadow-fraction", type=float, default=0.6)
    ap.add_argument("--job-dir", default="job")
    a = ap.parse_args()
    meta = {k: getattr(a, k) for k in ("latitude", "longitude", "direction", "date", "time", "height")}
    try:
        res = run_job(a.job_dir, a.image, meta, a.weights, a.fov_deg, a.shadow_fraction,
                      progress=lambda t: print(f"[..] {t}"))
    except (ValueError, JobError) as e:
        sys.exit(f"ERROR: {e}")
    print(f"Done. Volume(s): {res['volume_litres']} L\nReport: {res['pdf']}\nTable:  {res['csv']}")


if __name__ == "__main__":
    main()