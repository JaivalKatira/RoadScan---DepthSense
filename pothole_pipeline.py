#!/usr/bin/env python3
"""
Pothole pipeline: rotate -> detect -> square-crop -> update metadata CSV.

For every image listed in the Excel metadata file (read-only) the script:
  1. Rotates the image so True North sits at the target angle (default 90 deg).
  2. Detects potholes (YOLO weights if given, otherwise a crude OpenCV fallback).
     By default detection runs on the ORIGINAL colour photo (what the model was trained on,
     no rotation padding, no filtering) and the boxes are then mapped into the rotated frame.
  3. Saves a square, zoomed crop around each pothole.
  4. Updates the target CSV (default pothole information.csv):
       - the True North angle column is updated to reflect the rotation
       - the capture-height column is recalculated for the zoom factor

Output inside --out-dir:
    imageN_pothole1.jpeg, imageN_pothole2.jpeg, ...   (one per pothole, rotated AND cropped, colour)
    imageN_outlined.jpeg                              (original image, detected potholes outlined)
Optional detection preprocessing (denoise, illumination flattening, CLAHE, sharpening) is OFF by
default because the YOLO model is trained on plain colour images; enable with --enhance only if
you also train on enhanced images. Detections that overlap or nearly touch are merged into
one bigger pothole (disable: --no-merge); merged potholes show their parts as thin yellow boxes
inside the red outline.
    (pass --save-rotated to also keep full rotated frames in rotated_full/)

Conventions / assumptions (see README-style notes at the bottom of this file):
  * "gps Direction(True North)" is treated as the current angle of True North in
    the image, measured counter-clockwise from the +x (right) axis, like a math
    angle (up = 90 deg). The image is rotated by (target - angle) so North ends
    up at `target` (90 deg = straight up). If your images come out turned the
    wrong way, re-run with --invert-rotation.
  * Zoom factor  = min(original height, original width) / crop side (px, in the
    rotated image).  New height = old height / zoom factor.

Usage:
    python pothole_pipeline.py --images-dir ./images --weights pothole.pt
    python pothole_pipeline.py --images-dir ./images            # OpenCV fallback

Requires: pandas numpy openpyxl opencv-python   (+ ultralytics for YOLO)
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column handling
# --------------------------------------------------------------------------
COL_IMG = "image no"
COL_DIR = "gps Direction(True North)"
COL_HEIGHT = "height of photo capture(Meters)"

EXTRA_COLS = [
    "Source Image",
    "Pothole Index",
    "Crop File",
    "Rotation Applied (deg)",
    "Zoom Factor",
    "Detection Confidence",
]

# Columns added by pothole_volume.py (run first) / pothole_depth.py (run second). They belong to the
# previous set of crops, so they are dropped if this script is re-run (re-run those two afterwards).
DOWNSTREAM_COLS = [
    "Mask Length (px)", "Mask Width (px)", "Mask File", "Area Flags",
    "GSD (m/px)", "Shadow Direction (deg)", "Shadow Source", "Shadow Detection Confidence",
    "Shadow Width Unclipped (px)", "Shadow Width Clipped Full (px)", "Shadow Fraction Used", "Shadow Clip Mask",
    "Shadow Width (px)", "Shadow Width (m)", "Depth (m)", "Depth (cm)", "Pothole Mask Area (px)",
    "Crop Detection Confidence", "Box Source", "Flags",
    "Length (m)", "Width (m)", "Surface Area (m2)", "Volume (m3)", "Volume (litres)",
    "Area Mode", "Shape Factor",
]

IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _key(s):
    s = str(s).replace("\u00a0", " ").strip().lower()
    for ch in "() ":
        s = s.replace(ch, "")
    return s


def find_col(df, canonical, *needles):
    """Find a column by exact (normalised) name, else by substring(s)."""
    k = _key(canonical)
    for c in df.columns:
        if _key(c) == k:
            return c
    for c in df.columns:
        if all(n in _key(c) for n in needles):
            return c
    return None


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def norm180(angle):
    """Normalise to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def rotate_image(img, angle_ccw, expand=True):
    """Rotate by angle_ccw degrees (counter-clockwise). Returns (rotated, valid_mask)."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_ccw, 1.0)
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        M[0, 2] += nw / 2.0 - cx
        M[1, 2] += nh / 2.0 - cy
    else:
        nw, nh = w, h
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (nw, nh),
                          flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, mask, M


def square_crop_box(x1, y1, x2, y2, W, H, padding):
    """Square box around the detection, clamped to stay inside the image."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding)
    side = int(round(min(max(side, 8), W, H)))
    x0 = int(round(min(max(cx - side / 2.0, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2.0, 0), H - side)))
    return x0, y0, side


def enhance_for_detection(img, valid_mask=None, clip_limit=3.0, grid=8):
    """Detection-only preprocessing (saved crops keep the original colours):
       1. grayscale + edge-preserving denoise (bilateral)
       2. illumination flattening: divide by a heavily blurred 'background' so
          uneven light / long shadow gradients no longer hide the pothole shape
          (the blur ignores the black padding added by the rotation)
       3. CLAHE local contrast boost
       4. mild unsharp mask so pothole rims/edges are crisper
    Returned as 3-channel BGR because YOLO expects 3 channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 40, 9)
    g = gray.astype(np.float32)
    h, w = g.shape
    m = (valid_mask > 0).astype(np.float32) if valid_mask is not None else np.ones_like(g)

    # background estimate on a small copy (fast), mask-aware (normalised convolution)
    sw, sh = max(w // 8, 8), max(h // 8, 8)
    g_s = cv2.resize(g * m, (sw, sh), interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(sw, sh) / 6.0
    bg_s = cv2.GaussianBlur(g_s, (0, 0), sigma) / (cv2.GaussianBlur(m_s, (0, 0), sigma) + 1e-6)
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)

    mean_bg = float(bg[m > 0].mean()) if (m > 0).any() else 128.0
    flat = np.clip(g / (bg + 1.0) * mean_bg, 0, 255) * m
    flat = flat.astype(np.uint8)

    flat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(flat)
    blur = cv2.GaussianBlur(flat, (0, 0), 3)
    sharp = cv2.addWeighted(flat, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def box_to_rotated(b, M):
    """Map a box (x1,y1,x2,y2,conf) from the original image into the rotated frame:
    transform its 4 corners with M and take the enclosing axis-aligned box."""
    c = np.array([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]], dtype=np.float32)
    p = cv2.transform(c, M)[0]
    return (float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max()), b[4])


def merge_detections(dets, gap_factor=1.0):
    """Merge boxes that overlap/touch or sit close together into one bigger pothole.
    Two boxes are joined when the gap between them is <= gap_factor x the larger
    side of the SMALLER box (so small neighbouring fragments are treated as parts of
    one pothole). Joining is transitive (A-B and B-C -> one box).
    Returns (merged_boxes, groups); groups[i] lists the raw boxes behind merged_boxes[i]."""
    n = len(dets)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def gap(a, b):
        dx = max(b[0] - a[2], a[0] - b[2], 0.0)
        dy = max(b[1] - a[3], a[1] - b[3], 0.0)
        return (dx * dx + dy * dy) ** 0.5

    def size(a):
        return max(a[2] - a[0], a[3] - a[1])

    for i in range(n):
        for j in range(i + 1, n):
            if gap(dets[i], dets[j]) <= gap_factor * min(size(dets[i]), size(dets[j])):
                parent[find(i)] = find(j)

    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(dets[i])
    merged, groups = [], []
    for members in buckets.values():
        merged.append((min(m[0] for m in members), min(m[1] for m in members),
                       max(m[2] for m in members), max(m[3] for m in members),
                       max(m[4] for m in members)))
        groups.append(members)
    return merged, groups


def draw_outlines(orig_img, dets, M, groups=None):
    """Draw each detection on the ORIGINAL (unrotated) image.
    Boxes were found in the rotated image, so their corners are mapped back with
    the inverse of the rotation; they appear as tilted rectangles."""
    out = orig_img.copy()
    Minv = cv2.invertAffineTransform(M)
    thick = max(2, int(round(min(out.shape[:2]) / 250)))
    scale = max(0.6, min(out.shape[:2]) / 1200)
    for i, (x1, y1, x2, y2, conf) in enumerate(dets, start=1):
        def to_pts(b):
            c = np.array([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]], dtype=np.float32)
            return cv2.transform(c, Minv)[0].round().astype(np.int32)

        # thin yellow boxes = the small detections that were merged into this pothole
        if groups is not None and len(groups[i - 1]) > 1:
            for mb in groups[i - 1]:
                cv2.polylines(out, [to_pts(mb)], True, (0, 255, 255), max(1, thick // 2), cv2.LINE_AA)
        pts = to_pts((x1, y1, x2, y2))
        cv2.polylines(out, [pts], True, (0, 0, 255), thick, cv2.LINE_AA)
        label = f"pothole {i}" + (f" {conf:.2f}" if conf else "")
        org = (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 8, 15))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
class YoloDetector:
    def __init__(self, weights, conf, max_det=20, imgsz=None, tta=False):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics is not installed:  pip install ultralytics")
        self.model = YOLO(weights)
        if imgsz is None:        # default: the size the weights were trained at
            try:
                imgsz = int(self.model.ckpt["train_args"]["imgsz"])
            except Exception:
                imgsz = 1024
        self.conf = conf
        self.max_det = max_det          # raw cap before merging (final cap is applied after merge)
        self.imgsz = imgsz
        self.tta = tta

    def __call__(self, img, valid_mask):
        res = self.model.predict(img, conf=self.conf, max_det=self.max_det,
                                 imgsz=self.imgsz, augment=self.tta, verbose=False)[0]
        out = []
        if res.boxes is not None:
            # Class 0 = Pothole. Class 1 = Shadow; NEVER crop shadows as potholes.
            for b, c, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                 res.boxes.conf.cpu().numpy(),
                                 res.boxes.cls.cpu().numpy()):
                if int(cls) != 0:
                    continue
                out.append((*map(float, b), float(c)))
        out.sort(key=lambda d: -d[4])              # most confident first
        return out[: self.max_det]                 # hard cap on detections per image


class OpenCVDetector:
    """Very rough fallback: large, dark, compact blobs on the road surface.
    Expect false positives/negatives; a trained YOLO model is far better."""

    def __init__(self, max_det=5):
        self.max_det = max_det

    def __call__(self, img, valid_mask):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        valid = cv2.erode(valid_mask, np.ones((15, 15), np.uint8)) > 0   # ignore rotation padding
        if valid.sum() < 100:
            return []
        thr = np.percentile(gray[valid], 10)
        dark = ((gray <= thr) & valid).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_valid = float(valid.sum())
        dets = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (0.003 * area_valid <= a <= 0.35 * area_valid):
                continue
            hull_a = cv2.contourArea(cv2.convexHull(c))
            if hull_a <= 0 or a / hull_a < 0.6:          # skip ragged shapes
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append((float(x), float(y), float(x + w), float(y + h), 0.0, a))
        dets.sort(key=lambda d: -d[5])
        return [d[:5] for d in dets[: self.max_det]]


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def index_images(images_dir):
    """Map lower-case stem -> full path."""
    found = {}
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS:
            found.setdefault(stem.lower(), os.path.join(images_dir, f))
    return found


def read_metadata(excel_path, sheet):
    df = pd.read_excel(excel_path, sheet_name=sheet)          # read-only
    img_c = find_col(df, COL_IMG, "image")
    dir_c = find_col(df, COL_DIR, "direction")
    hei_c = find_col(df, COL_HEIGHT, "height")
    missing = [n for n, c in (("image no", img_c), ("gps Direction", dir_c),
                              ("height", hei_c)) if c is None]
    if missing:
        sys.exit(f"Excel is missing column(s): {missing}. Found: {list(df.columns)}")
    df[dir_c] = pd.to_numeric(df[dir_c], errors="coerce")
    df[hei_c] = pd.to_numeric(df[hei_c], errors="coerce")
    df[img_c] = df[img_c].astype(str).str.strip()
    return df, img_c, dir_c, hei_c


def load_base_csv(csv_path, excel_df, e_img, e_dir, e_hei):
    """Existing CSV rows (one per source image), or the Excel table if no CSV yet."""
    if os.path.isfile(csv_path):
        base = pd.read_csv(csv_path, encoding="utf-8-sig")
        base = base.drop(columns=[c for c in DOWNSTREAM_COLS if c in base.columns])
    else:
        base = excel_df.copy()
    b_img = find_col(base, COL_IMG, "image")
    if b_img is None:
        sys.exit(f"'{csv_path}' has no 'image no' column. Found: {list(base.columns)}")
    src_col = "Source Image" if "Source Image" in base.columns else b_img
    base["_src"] = base[src_col].astype(str).str.strip()
    base = base.drop_duplicates("_src", keep="first").set_index("_src")
    return base, b_img


# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default="images", help="Folder with image1.jpeg, image2.jpeg, ... (default: images)")
    ap.add_argument("--excel", default="main.xlsx", help="Metadata Excel (read-only)")
    ap.add_argument("--sheet", default=0, help="Excel sheet name/index (default first)")
    ap.add_argument("--csv", default="pothole information.csv", help='Target CSV to update (default "pothole information.csv")')
    ap.add_argument("--output-csv", help="Write here instead of updating --csv in place")
    ap.add_argument("--out-dir", default="processed", help="Folder for rotated images and crops")
    ap.add_argument("--weights", help="YOLO pothole weights (.pt). Omit to use OpenCV fallback")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="YOLO confidence threshold (default 0.15; model is still low-recall)")
    ap.add_argument("--max-det", type=int, default=2,
                    help="Maximum potholes detected/cropped per image (default 2)")
    ap.add_argument("--enhance", action="store_true",
                    help="Apply grayscale + contrast boost before detection (off: the model is trained on plain colour)")
    ap.add_argument("--detect-on", choices=["original", "rotated"], default="original",
                    help="Run detection on the original photo and map boxes to the rotated frame "
                         "(default), or on the rotated image (black padding, unseen in training)")
    ap.add_argument("--clip-limit", type=float, default=3.0,
                    help="CLAHE contrast strength for detection (default 3.0; higher = stronger)")
    ap.add_argument("--raw-det", type=int, default=20,
                    help="Max raw detections per image BEFORE merging (default 20)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="YOLO inference size (default: the size the weights were trained at)")
    ap.add_argument("--tta", action="store_true", help="YOLO test-time augmentation (slower, often more complete boxes)")
    ap.add_argument("--merge-gap", type=float, default=0.3,
                    help="Merge detections whose gap <= this x the larger side of the smaller box (default 0.3)")
    ap.add_argument("--no-merge", action="store_true", help="Disable merging of nearby detections")
    ap.add_argument("--padding", type=float, default=0.15, help="Extra margin around each box (0.15 = 15%%)")
    ap.add_argument("--target-angle", type=float, default=90.0, help="Where True North should end up (deg)")
    ap.add_argument("--invert-rotation", action="store_true", help="Rotate the opposite way")
    ap.add_argument("--save-rotated", action="store_true",
                    help="Also save the full rotated image (off by default; only cropped+rotated potholes are saved)")
    ap.add_argument("--no-expand", action="store_true", help="Keep original canvas size when rotating (corners are cut)")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    meta, e_img, e_dir, e_hei = read_metadata(args.excel, sheet)
    base, b_img = load_base_csv(args.csv, meta, e_img, e_dir, e_hei)
    b_dir = find_col(base, COL_DIR, "direction") or e_dir
    b_hei = find_col(base, COL_HEIGHT, "height") or e_hei

    images = index_images(args.images_dir)
    if not images:
        sys.exit(f"No images found in '{os.path.abspath(args.images_dir)}'. "
                 "Use --images-dir to point at the folder containing image1.jpeg, ...")

    # One output per pothole: the rotated AND cropped image, saved straight into out-dir
    crop_dir = args.out_dir
    os.makedirs(crop_dir, exist_ok=True)
    rot_dir = os.path.join(args.out_dir, "rotated_full")
    if args.save_rotated:
        os.makedirs(rot_dir, exist_ok=True)

    if args.weights:
        detector = YoloDetector(args.weights, args.conf, args.raw_det, args.imgsz, args.tta)
        print(f"Detector: YOLO ({args.weights}), imgsz {detector.imgsz}, detect on {args.detect_on}, TTA {'on' if args.tta else 'off'}; "
              f"close detections merged, max {args.max_det} pothole(s) per image after merging")
    else:
        detector = OpenCVDetector(args.raw_det)
        print("Detector: OpenCV fallback (rough). Pass --weights for a trained YOLO pothole model.")

    out_rows = []
    for _, row in meta.iterrows():
        name = row[e_img]
        theta, height = row[e_dir], row[e_hei]
        path = images.get(name.lower())
        if path is None:
            print(f"[skip] {name}: no image file found")
            continue
        if pd.isna(theta) or pd.isna(height):
            print(f"[skip] {name}: missing angle or height in Excel")
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] {name}: could not read {path}")
            continue
        ext = os.path.splitext(path)[1]
        oh, ow = img.shape[:2]
        ref = float(min(oh, ow))

        # 1. rotate so True North sits at the target angle
        delta = norm180(args.target_angle - float(theta))
        if args.invert_rotation:
            delta = -delta
        rotated, mask, M = rotate_image(img, delta, expand=not args.no_expand)
        if args.save_rotated:                      # optional debug copy, off by default
            cv2.imwrite(os.path.join(rot_dir, name + ext), rotated)
        new_angle = round((float(theta) + (-delta if args.invert_rotation else delta)) % 360.0, 4)

        # 2. detect (on a grayscale, contrast-boosted copy; crops below use the colour image)
        H, W = rotated.shape[:2]
        if args.detect_on == "original":
            full = np.full((oh, ow), 255, np.uint8)
            det_img = enhance_for_detection(img, full, args.clip_limit) if args.enhance else img
            raw = [box_to_rotated(b, M) for b in detector(det_img, full)]
        else:
            det_img = enhance_for_detection(rotated, mask, args.clip_limit) if args.enhance else rotated
            raw = detector(det_img, mask)

        # small neighbouring detections are treated as parts of one bigger pothole
        if args.no_merge:
            dets, groups = list(raw), [[d] for d in raw]
        else:
            dets, groups = merge_detections(raw, args.merge_gap)
        # keep the best --max-det potholes (highest confidence, then largest)
        order = sorted(range(len(dets)),
                       key=lambda i: (-dets[i][4], -(dets[i][2] - dets[i][0]) * (dets[i][3] - dets[i][1])))
        order = order[: args.max_det]
        dets = [dets[i] for i in order]
        groups = [groups[i] for i in order]

        # original image with the detected potholes outlined (boxes mapped back to original frame)
        outlined = draw_outlines(img, dets, M, groups)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_outlined{ext}"), outlined)

        base_row = base.loc[name].to_dict() if name in base.index else {b_img: name}

        def make_row(img_name, idx, crop_file, zoom, conf, new_h):
            r = dict(base_row)
            r[b_img] = img_name
            r[b_dir] = new_angle
            r[b_hei] = round(new_h, 4)
            r["Source Image"] = name
            r["Pothole Index"] = idx
            r["Crop File"] = crop_file
            r["Rotation Applied (deg)"] = round(delta, 4)
            r["Zoom Factor"] = round(zoom, 4)
            r["Detection Confidence"] = conf
            return r

        # 3. square crops + 4. metadata
        if not dets:
            print(f"[ok]   {name}: rotated {delta:+.2f} deg, no pothole found")
            out_rows.append(make_row(name, "", "", 1.0, "", float(height)))
            continue

        for i, (x1, y1, x2, y2, conf) in enumerate(dets, start=1):
            x0, y0, side = square_crop_box(x1, y1, x2, y2, W, H, args.padding)
            crop = rotated[y0:y0 + side, x0:x0 + side]
            crop_name = f"{name}_pothole{i}"
            cv2.imwrite(os.path.join(crop_dir, crop_name + ext), crop)
            zoom = ref / side
            new_h = float(height) / zoom
            out_rows.append(make_row(crop_name, i, crop_name + ext, zoom,
                                     round(conf, 4) if conf else "", new_h))
        print(f"[ok]   {name}: rotated {delta:+.2f} deg, {len(dets)} pothole crop(s)")

    if not out_rows:
        sys.exit("Nothing processed.")

    out_df = pd.DataFrame(out_rows)
    first = [c for c in base.columns if c in out_df.columns and c != "_src"]
    cols = first + [c for c in out_df.columns if c not in first]
    cols = [c for c in cols if c != "_src"]
    out_df = out_df[cols]

    target = args.output_csv or args.csv
    if os.path.isfile(target) and not args.output_csv:
        bak = target + ".bak"
        if not os.path.exists(bak):                      # keep the very first original safe
            shutil.copy2(target, bak)
            print(f"Backup of original CSV: {bak}")
    try:
        out_df.to_csv(target, index=False, encoding="utf-8-sig")
    except PermissionError:
        sys.exit(f"Cannot write '{target}'. Close it in Excel and run again.")
    print(f"\nUpdated CSV: {target} ({len(out_df)} rows)")
    print(f"Rotated + cropped pothole images: {crop_dir}")


if __name__ == "__main__":
    main()