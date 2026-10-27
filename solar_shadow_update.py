#!/usr/bin/env python3
"""
Reads main.xlsx (READ-ONLY) and writes "pothole information.csv" containing every
original column exactly as it is (including any extra columns such as
"height of photo capture(Meters)") plus two new columns at the end:

    Sun Elevation Angle (deg)
    Shadow Ratio

main.xlsx itself is never modified.

Required columns (exact names; matched ignoring case/extra spaces):
    image no | latitude(N) | longitude(E) | gps Direction(True North)
    time(hr:min) | date (DD-MM-YYYY)
Any other columns are carried through untouched and not used.

Usage:
    python solar_shadow_csv.py                 # main.xlsx -> "pothole information.csv"
    python solar_shadow_csv.py --sheet Sheet1

Requires: pandas, numpy, openpyxl   (pip install pandas numpy openpyxl)
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

EXCEL_FILE = "main.xlsx"
OUTPUT_CSV = "pothole information.csv"

COL_IMG = "image no"
COL_LAT = "latitude(N)"
COL_LON = "longitude(E)"
COL_DIR = "gps Direction(True North)"
COL_TIME = "time(hr:min)"
COL_DATE = "date"

REQUIRED = [COL_IMG, COL_LAT, COL_LON, COL_DIR, COL_TIME, COL_DATE]
STD_MERIDIAN = 82.5

OUT_ELEV = "Sun Elevation Angle (deg)"
OUT_SHADOW = "Shadow Ratio"


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def parse_date(value):
    """Timestamp from DD-MM-YYYY text, a datetime/date, or an Excel serial number."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    s = str(value).strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%m-%y"):
        try:
            return pd.Timestamp(dt.datetime.strptime(s, fmt))
        except ValueError:
            continue
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def parse_time_to_hours(value):
    """Decimal hours from 'HH:MM' text, time/datetime objects or Excel day fractions."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (dt.datetime, pd.Timestamp, dt.time)):
        return value.hour + value.minute / 60 + value.second / 3600
    if isinstance(value, dt.timedelta):
        return value.total_seconds() / 3600
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v * 24 if 0 <= v < 1 else np.nan
    s = str(value).strip().upper()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%H.%M"):
        try:
            t = dt.datetime.strptime(s, fmt)
            return t.hour + t.minute / 60 + t.second / 3600
        except ValueError:
            continue
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return parsed.hour + parsed.minute / 60 + parsed.second / 3600


def standardize_columns(df):
    """Rename headers to the exact expected names, ignoring case, stray/non-breaking
    spaces and spacing around brackets. Unrecognised columns keep their own names."""
    def key(s):
        s = str(s).replace("\u00a0", " ").strip().lower()
        for ch in "() ":
            s = s.replace(ch, "")
        return s

    canonical = {key(c): c for c in REQUIRED}
    rename = {c: canonical[key(c)] for c in df.columns if key(c) in canonical}
    return df.rename(columns=rename)


# --------------------------------------------------------------------------
# Calculation
# --------------------------------------------------------------------------
def compute_elevation_and_shadow(df, meridian=STD_MERIDIAN):
    """Return (elevation_deg, shadow_ratio) Series aligned to df's index."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}\nFound: {list(df.columns)}")

    lat = pd.to_numeric(df[COL_LAT], errors="coerce")
    lon = pd.to_numeric(df[COL_LON], errors="coerce")
    dates = df[COL_DATE].apply(parse_date)
    hours = df[COL_TIME].apply(parse_time_to_hours).astype(float)

    # Day of year 1-365 (day 366 of a leap year is capped at 365)
    d = dates.dt.dayofyear.clip(upper=365).astype(float)

    decl = -23.45 * np.cos(np.radians((360.0 / 365.0) * (d + 10)))
    omega = (hours - 12.0) * 15.0 + (lon - meridian)

    lat_r, decl_r, omega_r = np.radians(lat), np.radians(decl), np.radians(omega)
    sin_alpha = (np.sin(decl_r) * np.sin(lat_r)
                 + np.cos(decl_r) * np.cos(lat_r) * np.cos(omega_r))
    alpha = np.degrees(np.arcsin(np.clip(sin_alpha, -1.0, 1.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 1.0 / np.tan(np.radians(alpha))
    ratio = pd.Series(ratio, index=df.index).where(alpha > 0)   # blank if sun not above horizon

    return alpha, ratio


# --------------------------------------------------------------------------
# CSV formatting: keep date/time looking the way they do in Excel
# --------------------------------------------------------------------------
def _fmt_date(v):
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(v).strftime("%d-%m-%Y")
    return v


def _fmt_time(v):
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.time)):
        return v.strftime("%H:%M")
    return v


def build_output(path=EXCEL_FILE, sheet=0, meridian=STD_MERIDIAN):
    """Read the Excel file (read-only) and return original columns + 2 new columns."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet)
    df = standardize_columns(df)

    alpha, ratio = compute_elevation_and_shadow(df, meridian)

    out = df.copy()
    out[COL_DATE] = out[COL_DATE].apply(_fmt_date)
    out[COL_TIME] = out[COL_TIME].apply(_fmt_time)
    out[OUT_ELEV] = alpha.round(4)
    out[OUT_SHADOW] = ratio.round(4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=EXCEL_FILE, help=f"Excel file (default: {EXCEL_FILE})")
    ap.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    ap.add_argument("--meridian", type=float, default=STD_MERIDIAN,
                    help="Standard meridian in deg E (default 82.5)")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    try:
        out = build_output(args.input, sheet, args.meridian)
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), OUTPUT_CSV)
    try:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")   # utf-8-sig opens cleanly in Excel
    except PermissionError:
        sys.exit(f"Cannot write '{out_path}'. Close it in Excel and run again.")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()