#!/usr/bin/env python3
"""Pothole report PDF: every area-debug image with its volume written beside it.

Reads "pothole information.csv" (after pothole_volume.py AND pothole_depth.py have run)
and the *_area_overlay.png images in area_debug/. Each pothole gets one row:
the area overlay on the left and its volume on the right.
Several potholes per page, one PDF out.

Usage:
    py pothole_report.py
    py pothole_report.py --csv "pothole information.csv" --area-dir area_debug --out pothole_report.pdf
    py pothole_report.py --per-page 2 --sort volume

Requires: pandas, reportlab, pillow   (pip install pandas reportlab pillow)
"""
import argparse
import os
import sys

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def num(row, col):
    v = pd.to_numeric(row.get(col), errors="coerce")
    return None if pd.isna(v) else float(v)


def fmt(v, unit, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f} {unit}"


def wrap(text, font, size, width, pdf):
    words, lines, cur = str(text).replace(";", "; ").split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdf.stringWidth(trial, font, size) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="pothole information.csv")
    ap.add_argument("--area-dir", default="area_debug", help="Folder with <crop>_area_overlay.png")
    ap.add_argument("--out", default="pothole_report.pdf")
    ap.add_argument("--per-page", type=int, default=3, help="Potholes per page (default 3)")
    ap.add_argument("--sort", choices=["csv", "volume"], default="csv",
                    help="csv = CSV order (default), volume = largest first")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    name_col = next((c for c in df.columns if str(c).strip().lower() == "image no"), None)
    if name_col is None:
        sys.exit("CSV has no 'image no' column")
    if "Volume (litres)" not in df.columns:
        print("WARNING: no 'Volume (litres)' column; run pothole_volume.py then pothole_depth.py first.")

    items, missing = [], []
    for _, r in df.iterrows():
        if "Crop File" in df.columns and (pd.isna(r["Crop File"]) or not str(r["Crop File"]).strip()):
            continue                                  # photo with no pothole found
        name = str(r[name_col]).strip()
        img = os.path.join(args.area_dir, name + "_area_overlay.png")
        if not os.path.isfile(img):
            missing.append(name)
            continue
        items.append((name, img, r))
    if not items:
        sys.exit(f"No area overlay images found in '{args.area_dir}'. Run pothole_volume.py first.")
    if args.sort == "volume":
        items.sort(key=lambda t: -(num(t[2], "Volume (litres)") or -1))

    W, H = A4
    margin = 36
    head = 54
    per = max(1, args.per_page)
    slot_h = (H - 2 * margin - head) / per
    img_w = (W - 2 * margin) * 0.52
    text_x_off = img_w + 18
    text_w = (W - 2 * margin) - text_x_off

    pdf = canvas.Canvas(args.out, pagesize=A4)
    pdf.setTitle("Pothole report")
    pages = (len(items) + per - 1) // per
    total_l = sum(num(r, "Volume (litres)") or 0 for _, _, r in items)

    for i, (name, path, r) in enumerate(items):
        slot = i % per
        if slot == 0:
            if i:
                pdf.showPage()
            pdf.setFont("Helvetica-Bold", 15)
            pdf.setFillColor(colors.black)
            pdf.drawString(margin, H - margin - 6, "Pothole report")
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(colors.grey)
            pdf.drawRightString(W - margin, H - margin - 6, f"Page {i // per + 1} of {pages}")
            pdf.setStrokeColor(colors.lightgrey)
            pdf.line(margin, H - margin - 32, W - margin, H - margin - 32)

        top = H - margin - head - slot * slot_h
        box_h = slot_h - 14

        # image, scaled to fit, left
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(img_w / iw, box_h / ih)
        dw, dh = iw * scale, ih * scale
        pdf.drawImage(ImageReader(path), margin, top - dh, dw, dh)
        pdf.setStrokeColor(colors.lightgrey)
        pdf.rect(margin, top - dh, dw, dh, stroke=1, fill=0)

        # numbers, right
        x = margin + text_x_off
        y = top - 14
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(x, y, name)
        y -= 30
        vol_l = num(r, "Volume (litres)")
        pdf.setFont("Helvetica-Bold", 24)
        pdf.setFillColor(colors.HexColor("#0b5cab") if vol_l is not None else colors.grey)
        pdf.drawString(x, y, "no volume" if vol_l is None else f"{vol_l:.2f} L")

    pdf.showPage()
    pdf.save()
    print(f"Saved {args.out}: {len(items)} pothole(s) on {pages} page(s).")
    if missing:
        print(f"No area overlay for {len(missing)} row(s): {', '.join(missing[:10])}"
              f"{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()