#!/usr/bin/env python3
"""
Pothole depth from shadow width.

Input : the CSV written by pothole_pipeline.py (default "pothole information.csv") and the
        square pothole crops it produced (default folder "processed").
Output: the same CSV ("pothole information.csv"), updated in place with the depth columns
        (GSD, shadow width, depth, ...) added to each pothole-crop row.

Method
------
1. YOLO (best.pt) finds the pothole inside each square crop. If no weights are given, or
   nothing is detected, the crop itself (minus the pipeline padding) is used as the region.
2. Inside that region OpenCV segments the shadow: Otsu threshold on the L (lightness)
   channel, morphological clean-up, largest dark connected component.
3. The shadow is then CLIPPED to the pixel-level pothole mask that pothole_volume.py
   saved (run it FIRST: YOLO mask, else GrabCut), so the shadow ends where it leaves the
   pothole instead of running on to the detection-box edge. Surface area, volume and
   litres are finished here too, from the mask area x GSD^2 x clipped depth.
   Manual correction: only --shadow-fraction (default 0.6) of the clipped shadow length is
   used, counted from the pothole edge towards the centre; this feeds depth and volume. Shadow length (px) is the extent of the
   clipped shadow along the shadow axis. --no-clip-shadow restores the old behaviour.
   Yellow in the depth_debug overlay = pothole boundary, red = clipped shadow.
4. Pixels -> metres with the ground sampling distance (GSD):
        GSD = 2 * H * tan(FOV/2) / crop_short_side_px
   where H is the (zoom-corrected) capture height column written by pothole_pipeline.py.
5. Depth = shadow_width_m / Shadow Ratio, with Shadow Ratio = 1 / tan(sun elevation)
   taken from the CSV (a shadow of horizontal length L is cast by a vertical drop
   L * tan(elevation), i.e. depth = L / ratio).

IMPORTANT: the camera field of view (--fov-deg) is NOT in your data. The default is only a
placeholder; set it to the FOV across the SHORTER side of your original photos, or give a
direct scale with --gsd (metres per pixel).

Usage:
    python pothole_depth.py --weights best.pt --fov-deg 65
    python pothole_depth.py --csv "pothole information.csv" --crops-dir processed --gsd 0.002

Requires: numpy pandas opencv-python (+ ultralytics for YOLO)
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

COL_IMG = "image no"
COL_HEIGHT = "height of photo capture(Meters)"
COL_RATIO = "Shadow Ratio"
COL_CROP = "Crop File"
IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_FOV_DEG = 65.0


# --------------------------------------------------------------------------
# Shared helpers (also imported by pothole_volume.py)
# --------------------------------------------------------------------------
def _key(s):
    s = str(s).replace("\u00a0", " ").strip().lower()
    for ch in "() ":
        s = s.replace(ch, "")
    return s


def find_col(df, canonical, *needles):
    k = _key(canonical)
    for c in df.columns:
        if _key(c) == k:
            return c
    for c in df.columns:
        if all(n in _key(c) for n in needles):
            return c
    return None


def index_images(folder):
    found = {}
    if not os.path.isdir(folder):
        return found
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS:
            found.setdefault(stem.lower(), os.path.join(folder, f))
    return found


def load_table(csv_path, crops_dir):
    """Read the pipeline CSV and return a list of dicts:
    {name, path, height, ratio} for every row that has a pothole crop on disk."""
    if not os.path.isfile(csv_path):
        sys.exit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    c_img = find_col(df, COL_IMG, "image")
    c_hei = find_col(df, COL_HEIGHT, "height")
    c_rat = find_col(df, COL_RATIO, "shadow", "ratio")
    c_crop = COL_CROP if COL_CROP in df.columns else None
    missing = [n for n, c in (("image no", c_img), ("height", c_hei),
                              ("Shadow Ratio", c_rat)) if c is None]
    if missing:
        sys.exit(f"'{csv_path}' is missing column(s): {missing}. Found: {list(df.columns)}\n"
                 "Run solar_shadow_update.py, then pothole_pipeline.py first.")

    images = index_images(crops_dir)
    rows = []
    for idx, r in df.iterrows():
        name = str(r[c_img]).strip()
        crop_file = ""
        if c_crop is not None:
            crop_file = "" if pd.isna(r[c_crop]) else str(r[c_crop]).strip()
            if not crop_file:                       # pipeline found no pothole in this photo
                continue
        path = os.path.join(crops_dir, crop_file) if crop_file else None
        if not path or not os.path.isfile(path):
            path = images.get(name.lower())
        if path is None:
            print(f"[skip] {name}: crop image not found in '{crops_dir}'")
            continue
        rows.append({
            "idx": idx,
            "name": name,
            "path": path,
            "height": pd.to_numeric(r[c_hei], errors="coerce"),
            "ratio": pd.to_numeric(r[c_rat], errors="coerce"),
        })
    return rows


class PotholeDetector:
    """YOLO wrapper returning the single best pothole box inside a crop (or None)."""

    def __init__(self, weights=None, conf=0.10, imgsz=None):
        self.model = None
        self.conf = conf
        self.imgsz = imgsz
        if weights:
            if not os.path.isfile(weights):
                sys.exit(f"Weights not found: {weights}")
            try:
                from ultralytics import YOLO
            except ImportError:
                sys.exit("ultralytics is not installed:  pip install ultralytics")
            self.model = YOLO(weights)
            if self.imgsz is None:
                try:
                    self.imgsz = int(self.model.ckpt["train_args"]["imgsz"])
                except Exception:
                    self.imgsz = 960

    def best_box(self, img):
        if self.model is None:
            return None
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz,
                                 max_det=10, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        # Only class 0 is the pothole; class 1 is the trained shadow detector.
        selected = res.boxes.cls.cpu().numpy().astype(int) == 0
        if not selected.any():
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()[selected]
        conf = res.boxes.conf.cpu().numpy()[selected]
        # Several fragments may be reported: take their union (the whole pothole), keep best conf.
        x1, y1 = xyxy[:, 0].min(), xyxy[:, 1].min()
        x2, y2 = xyxy[:, 2].max(), xyxy[:, 3].max()
        return float(x1), float(y1), float(x2), float(y2), float(conf.max())


def fallback_box(img, padding=0.15):
    """Whole crop minus the padding the pipeline added around the pothole."""
    h, w = img.shape[:2]
    side = min(h, w) / (1.0 + padding)
    cx, cy = w / 2.0, h / 2.0
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, 0.0


def gsd_from(height_m, crop_short_px, fov_deg):
    """Metres per pixel for a camera at height_m whose FOV (across the crop's
    short side) is fov_deg."""
    return 2.0 * height_m * math.tan(math.radians(fov_deg) / 2.0) / float(crop_short_px)


# --------------------------------------------------------------------------
# Shadow measurement
# --------------------------------------------------------------------------
def measure_shadow(img, box, method="dt", angle_deg=None):
    """Segment a locally darker, chromatically similar cast-shadow candidate.

    angle_deg: image direction of shadow length in degrees clockwise from right.
    When unspecified, infer the narrow dimension of the candidate; this is an
    approximation and should be checked in debug overlays.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
    if x2-x1 < 12 or y2-y1 < 12:
        return None
    roi = img[y1:y2,x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:,:,0]
    short = min(light.shape)
    # Compare illumination to a locally smoothed scene, not a global threshold:
    # uneven road colours can be dark without being shadows.
    blur = cv2.GaussianBlur(light, (0,0), max(5.,short/7.))
    residual = blur-light
    smooth = cv2.GaussianBlur(light, (5,5), 0)
    otsu_thr,_ = cv2.threshold(smooth.astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    local_thr = max(5.,float(np.percentile(residual,76)))
    # Two candidates: local photometric attenuation + absolute low-light region.
    candidates = [(residual > local_thr) & (smooth < np.percentile(smooth,72)),
                  (smooth < otsu_thr) & (residual > 2.5)]
    k = max(3, (short//65)|1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    best = None
    for candidate in candidates:
        mask = cv2.morphologyEx((candidate.astype('uint8')*255), cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)
        n,labels,stats,_ = cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,n):
            area=int(stats[i,cv2.CC_STAT_AREA])
            fraction=area/float((x2-x1)*(y2-y1))
            if area < max(18,short*short*0.002) or not (0.008 <= fraction <= 0.68):
                continue
            comp=(labels==i)
            surrounding=cv2.dilate(comp.astype('uint8'),np.ones((9,9),np.uint8)).astype(bool)&~comp
            if surrounding.sum()<10:
                continue
            contrast=float(np.median(light[surrounding])-np.median(light[comp]))
            # Cast shadows tend to preserve chromaticity relative to nearby lit road.
            chroma=float(np.linalg.norm(np.median(lab[surrounding,1:],axis=0)-np.median(lab[comp,1:],axis=0)))
            if contrast < 7 or chroma > 35:
                continue
            # Reject giant dark pothole interiors and tiny noise; prefer locally
            # contrasted, coherent patches over simply choosing the largest blob.
            score=contrast/(1+chroma/18.)*min(1.,area/max(30.,short*short*.025))
            if best is None or score>best[0]:
                best=(score,comp.astype('uint8')*255,area,contrast)
    if best is None:
        return None
    _,comp,area,contrast=best
    cnts,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt=max(cnts,key=cv2.contourArea)
    xy=np.column_stack(np.where(comp>0))[:,::-1].astype(np.float32)
    if len(xy)<10:
        return None
    if angle_deg is not None:
        axis=np.array([math.cos(math.radians(angle_deg)),math.sin(math.radians(angle_deg))])
    else:
        # Short axis of fitted ellipse/PCA approximates cast-shadow band length.
        # A known solar direction should be passed for reliable measurements.
        cov=np.cov(xy,rowvar=False)
        _,vectors=np.linalg.eigh(cov)
        axis=vectors[:,0]
    projection=xy@axis
    length=float(np.percentile(projection,95)-np.percentile(projection,5))
    if not np.isfinite(length) or length<2:
        return None
    mask_full=np.zeros((H,W),np.uint8)
    mask_full[y1:y2,x1:x2]=comp
    cnt_full=cnt+np.array([[[x1,y1]]],dtype=cnt.dtype)
    return {"width_px":length,"area_px":area,"roi_area_px":(x2-x1)*(y2-y1),
            "contrast":contrast,"mask_full":mask_full,"contour":cnt_full,
            "angle_deg":float(math.degrees(math.atan2(axis[1],axis[0])))}



def measure_trained_shadow(img, model, pothole_box, conf=0.10, imgsz=960, angle_deg=None):
    """Use class-1 YOLO shadow boxes; clip to the pothole region.

    Trained shadow boxes locate the cast-shadow region; OpenCV then refines
    its pixels and measures the *projection length*, not its bbox width.
    """
    if model is None:
        return None
    pred = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return None
    classes = pred.boxes.cls.cpu().numpy().astype(int)
    coords = pred.boxes.xyxy.cpu().numpy()
    scores = pred.boxes.conf.cpu().numpy()
    ph, pw = img.shape[:2]
    px1,py1,px2,py2=[int(round(v)) for v in pothole_box[:4]]
    px1,py1,px2,py2=max(px1,0),max(py1,0),min(px2,pw),min(py2,ph)
    best=None
    for j in np.where(classes==1)[0]:
        x1,y1,x2,y2=[int(round(v)) for v in coords[j]]
        x1,y1,x2,y2=max(x1,px1),max(y1,py1),min(x2,px2),min(y2,py2)
        if x2-x1 < 4 or y2-y1 < 4:continue
        overlap=(x2-x1)*(y2-y1)
        whole=max(1,(coords[j,2]-coords[j,0])*(coords[j,3]-coords[j,1]))
        if overlap / whole < 0.35:continue
        # Refine the learned bounding box with a local chromatic-lightness mask.
        roi=img[y1:y2,x1:x2]
        light=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)[:,:,0]
        light=cv2.GaussianBlur(light,(5,5),0)
        thresh=np.percentile(light,65)
        binary=((light<=thresh).astype(np.uint8)*255)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        n, labels, stats, _=cv2.connectedComponentsWithStats(binary,8)
        if n<2:continue
        k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        component=(labels==k).astype(np.uint8)*255
        if np.count_nonzero(component)<12:continue
        ys,xs=np.where(component>0)
        xy=np.column_stack([xs+x1,ys+y1]).astype(float)
        if angle_deg is not None:
            axis=np.array([math.cos(math.radians(angle_deg)),math.sin(math.radians(angle_deg))])
        else:
            # PCA minor axis = shadow band thickness, the same fallback as before.
            _,eig=np.linalg.eigh(np.cov(xy,rowvar=False))
            axis=eig[:,0]
        proj=xy@axis
        length=float(np.percentile(proj,95)-np.percentile(proj,5))
        if not np.isfinite(length) or length<=1:continue
        full=np.zeros((ph,pw),np.uint8)
        full[y1:y2,x1:x2]=component
        cnts,_=cv2.findContours(full,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:continue
        cnt=max(cnts,key=cv2.contourArea)
        score=float(scores[j])*(overlap/whole)
        item={'width_px':length,'area_px':int(np.count_nonzero(component)),
              'roi_area_px':max(1,(px2-px1)*(py2-py1)),
              'contrast':float(np.percentile(light,85)-np.percentile(light,15)),
              'mask_full':full,'contour':cnt,'angle_deg':float(math.degrees(math.atan2(axis[1],axis[0]))),
              'confidence':float(scores[j]),'source':'yolo_shadow'}
        if best is None or score>best[0]:best=(score,item)
    return best[1] if best else None


def load_saved_mask(mask_dir, name, shape):
    """Pothole mask written by pothole_volume.py (0/255 PNG), resized to the crop if needed."""
    p = os.path.join(mask_dir, name + "_area_mask.png")
    if not os.path.isfile(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    return m if m.any() else None


def clip_shadow_to_pothole(meas, pothole_mask, min_pixels=12, fraction=1.0):
    """Stop the shadow at the pothole boundary, then keep only `fraction` of it.

    1. CLIP: the shadow candidate is found inside the detection box, so it can run on past
       the pothole rim into the surrounding road. Intersecting it with the pixel-level
       pothole mask (saved by pothole_volume.py) cuts it where the ray leaves the pothole.
       Length is measured along the SAME axis as before, as the full extent (min to max
       projection) of the clipped shadow.
    2. FRACTION (manual correction, default 0.6 set by the caller): only that fraction of
       the clipped length counts, measured from the end of the shadow nearest the pothole
       edge towards the pothole centre. The edge end is the end whose pixels are closest
       to the mask boundary (distance transform of the pothole mask).

    Returns the updated measurement dict, or None if nothing of the shadow lies inside
    the pothole mask.
    """
    inside = (meas["mask_full"] > 0) & (pothole_mask > 0)
    inside = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == k).astype(np.uint8)
    if int(comp.sum()) < min_pixels:
        return None
    ys, xs = np.where(comp > 0)
    a = math.radians(meas["angle_deg"])
    axis = np.array([math.cos(a), math.sin(a)])
    perp = np.array([-axis[1], axis[0]])
    pts = np.column_stack([xs, ys]).astype(float)
    proj = pts @ axis
    lo, hi = float(proj.min()), float(proj.max())
    full_len = hi - lo + 1.0
    if not np.isfinite(full_len) or full_len < 2:
        return None

    # Which end of the shadow is at the pothole edge? (smaller distance to mask boundary)
    dist = cv2.distanceTransform((pothole_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    span = hi - lo
    d_lo = float(dist[ys[proj <= lo + 0.1 * span], xs[proj <= lo + 0.1 * span]].mean())
    d_hi = float(dist[ys[proj >= hi - 0.1 * span], xs[proj >= hi - 0.1 * span]].mean())
    edge_is_lo = d_lo <= d_hi
    keep = float(fraction) * span
    if edge_is_lo:
        r_lo, r_hi = lo, lo + keep
    else:
        r_lo, r_hi = hi - keep, hi
    sel = (proj >= r_lo - 1e-6) & (proj <= r_hi + 1e-6)
    comp_used = np.zeros_like(comp)
    comp_used[ys[sel], xs[sel]] = 1
    if int(comp_used.sum()) < 2:
        return None
    used_len = float(fraction) * full_len
    mid = float(np.median(pts[sel] @ perp))
    edge_pt = axis * (lo if edge_is_lo else hi) + perp * mid
    cut_pt = axis * (r_hi if edge_is_lo else r_lo) + perp * mid
    p1 = tuple(int(round(v)) for v in edge_pt)      # pothole edge
    p2 = tuple(int(round(v)) for v in cut_pt)       # 60% point towards the centre
    cnts, _ = cv2.findContours(comp_used, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Unclipped length on the SAME measure (full extent along the axis) so the numbers
    # are comparable; meas["width_px"] itself is a trimmed 5-95 percentile.
    uy, ux = np.where(meas["mask_full"] > 0)
    uproj = np.column_stack([ux, uy]).astype(float) @ axis
    unclipped = float(uproj.max() - uproj.min() + 1.0)
    out = dict(meas)
    out.update({
        "width_px_unclipped": unclipped,
        "width_px_clipped_full": full_len,
        "shadow_fraction": float(fraction),
        "area_px_unclipped": meas["area_px"],
        "width_px": used_len,
        "area_px": int(comp_used.sum()),
        "mask_full": comp_used * 255,
        "contour": max(cnts, key=cv2.contourArea),
        "line": (p1, p2),
        "clipped": True,
    })
    return out


def save_debug(img, box, meas, text, path, pothole_mask=None):
    out = img.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
    if pothole_mask is not None:
        pc, _ = cv2.findContours((pothole_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, pc, -1, (0, 255, 255), 2)      # yellow = pothole boundary
    if meas is not None:
        cv2.drawContours(out, [meas["contour"]], -1, (0, 0, 255), 2)
    if meas is not None:
        if meas.get("line") is not None:
            cv2.line(out, meas["line"][0], meas["line"][1], (255, 0, 255), 2)
        else:
            ys,xs=np.where(meas["mask_full"]>0)
            if len(xs):
                cx,cy=int(np.median(xs)),int(np.median(ys))
                a=math.radians(meas["angle_deg"])
                half=meas["width_px"]/2
                p1=(int(cx-half*math.cos(a)),int(cy-half*math.sin(a)))
                p2=(int(cx+half*math.cos(a)),int(cy+half*math.sin(a)))
                cv2.line(out,p1,p2,(255,0,255),2)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(out.shape[:2]) / 900), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="pothole information.csv", help="CSV from pothole_pipeline.py")
    ap.add_argument("--crops-dir", default="processed", help="Folder with imageN_potholeK crops")
    ap.add_argument("--weights", default=None, help="YOLO weights (best.pt). Omit for whole-crop fallback")
    ap.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO inference size (default: trained size)")
    ap.add_argument("--padding", type=float, default=0.15,
                    help="Padding the pipeline added around potholes; used by the fallback box")
    ap.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG,
                    help=f"Camera FOV (deg) across the SHORTER side of the original photo "
                         f"(default {DEFAULT_FOV_DEG:g} = placeholder, set yours!)")
    ap.add_argument("--gsd", type=float, default=None,
                    help="Direct scale in metres per pixel; overrides height/FOV calculation")
    ap.add_argument("--width-method", choices=["dt", "minrect"], default="dt",
                    help="Shadow width estimator (default dt = distance-transform ridge)")
    ap.add_argument("--shadow-angle-deg", type=float, default=None,
                    help="Image direction of cast shadow (clockwise degrees from right). Recommended when known; otherwise infer band short axis")
    ap.add_argument("--min-contrast", type=float, default=15.0,
                    help="Flag shadows whose lightness contrast vs the rest is below this (default 15)")
    ap.add_argument("--mask-dir", default="area_debug",
                    help="Folder holding the <crop>_area_mask.png files written by pothole_volume.py (default area_debug)")
    ap.add_argument("--no-clip-shadow", action="store_true",
                    help="Do not clip the shadow to the pothole mask (old behaviour)")
    ap.add_argument("--shadow-fraction", type=float, default=0.6,
                    help="Manual correction: only this fraction of the clipped shadow length counts, "
                         "measured from the pothole edge towards the centre (default 0.6)")
    ap.add_argument("--shape-factor", type=float, default=1.0,
                    help="Volume = surface area x depth x shape factor (default 1.0)")
    ap.add_argument("--out-csv", default="pothole information.csv",
                    help="Where to write the updated table (default: updates the same file)")
    ap.add_argument("--debug-dir", default="depth_debug", help="Folder for overlay images ('' to disable)")
    args = ap.parse_args()

    if not 0 < args.shadow_fraction <= 1:
        sys.exit("--shadow-fraction must be in (0, 1]")
    rows = load_table(args.csv, args.crops_dir)
    table = pd.read_csv(args.csv, encoding="utf-8-sig")
    if not rows:
        sys.exit("No pothole crops found to process.")
    if args.gsd is None and args.fov_deg == DEFAULT_FOV_DEG:
        print(f"WARNING: using placeholder FOV {DEFAULT_FOV_DEG:g} deg. Depths scale with it; "
              "pass --fov-deg or --gsd for real values.")
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    det = PotholeDetector(args.weights, args.conf, args.imgsz)
    print(f"Detector: {'YOLO ' + args.weights if args.weights else 'none (whole-crop fallback)'}")

    out, out_idx = [], []
    for r in rows:
        name = r["name"]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] {name}: unreadable image")
            continue
        flags = []
        box = det.best_box(img)
        src = "yolo"
        if box is None:
            box = fallback_box(img, args.padding)
            src = "fallback"
            flags.append("no_detection_used_whole_crop")

        h_m, ratio = r["height"], r["ratio"]
        short = min(img.shape[:2])
        gsd = args.gsd if args.gsd else (gsd_from(h_m, short, args.fov_deg)
                                         if pd.notna(h_m) and h_m > 0 else float("nan"))
        meas = measure_trained_shadow(img, det.model, box, args.conf, det.imgsz or args.imgsz or 960,
                                     args.shadow_angle_deg)
        if meas is None:
            flags.append('trained_shadow_not_detected')
            meas = measure_shadow(img, box, args.width_method, args.shadow_angle_deg)

        # Stop the shadow at the pothole boundary instead of the detection-box edge.
        # The mask comes from pothole_volume.py, which must have been run first.
        pothole_mask, clip_mode, w_unclipped = None, "", float("nan")
        if meas is not None and args.no_clip_shadow:
            flags.append("shadow_fraction_not_applied")
        if meas is not None and not args.no_clip_shadow:
            pothole_mask = load_saved_mask(args.mask_dir, name, img.shape)
            clip_mode = str(table.at[r["idx"], "Area Mode"]) if "Area Mode" in table.columns else ""
            if pothole_mask is None:
                flags.append("shadow_not_clipped_no_pothole_mask")
                flags.append("shadow_fraction_not_applied")
            else:
                if clip_mode == "opencv_contour":
                    flags.append("shadow_clip_mask_is_grabcut_review")
                clipped = clip_shadow_to_pothole(meas, pothole_mask, fraction=args.shadow_fraction)
                w_unclipped = clipped["width_px_unclipped"] if clipped else meas["width_px"]
                if clipped is None:
                    flags.append("shadow_outside_pothole_mask")
                    meas = None            # nothing of the shadow lies inside the pothole
                else:
                    meas = clipped
                    flags.append("shadow_clipped_to_pothole_mask")
                    if meas["width_px_clipped_full"] < 0.5 * w_unclipped:
                        flags.append("shadow_clip_removed_over_half")

        w_px = w_m = depth = float("nan")
        if meas is None:
            flags.append("no_shadow_found")
        else:
            w_px = meas["width_px"]
            if args.shadow_angle_deg is None:
                flags.append("shadow_direction_inferred")
            if meas["contrast"] < args.min_contrast:
                flags.append("low_shadow_contrast")
            if meas["area_px"] > 0.6 * meas["roi_area_px"]:
                flags.append("shadow_fills_most_of_pothole")
        if pd.isna(ratio) or ratio <= 0:
            flags.append("no_shadow_ratio(sun_below_horizon_or_blank)")
        if pd.isna(gsd):
            flags.append("no_scale")
        if meas is not None and pd.notna(gsd):
            w_m = w_px * gsd
            if pd.notna(ratio) and ratio > 0:
                depth = w_m / ratio

        # Area step results (pothole_volume.py, run first) -> metres and volume
        trow = table.loc[r["idx"]]
        area_px = pd.to_numeric(trow.get("Pothole Mask Area (px)"), errors="coerce")
        len_px = pd.to_numeric(trow.get("Mask Length (px)"), errors="coerce")
        wid_px = pd.to_numeric(trow.get("Mask Width (px)"), errors="coerce")
        area_flags = trow.get("Area Flags")
        area_m2 = volume = length_m = width_m = float("nan")
        if pd.isna(area_px):
            flags.append("no_area_mask(run_pothole_volume_first)")
        elif pd.notna(gsd) and gsd > 0:
            area_m2 = area_px * gsd * gsd
            if pd.notna(len_px):
                length_m = len_px * gsd
            if pd.notna(wid_px):
                width_m = wid_px * gsd
            if pd.notna(depth) and depth > 0:
                volume = area_m2 * depth * args.shape_factor
        if pd.isna(volume):
            flags.append("volume_not_computed")
        if isinstance(area_flags, str) and area_flags:
            flags.extend(f for f in area_flags.split(";") if f)

        if args.debug_dir:
            txt = f"{name} w={w_px:.1f}px depth={depth * 100:.1f}cm" if pd.notna(depth) else f"{name} no depth"
            save_debug(img, box, meas, txt, os.path.join(args.debug_dir, f"{name}_depth.jpg"),
                       pothole_mask)

        out_idx.append(r["idx"])
        out.append({
            "GSD (m/px)": gsd,
            "Shadow Direction (deg)": round(meas["angle_deg"], 2) if meas else "",
            "Shadow Source": meas.get("source", "opencv_fallback") if meas else "none",
            "Shadow Detection Confidence": round(meas.get("confidence", float("nan")), 4) if meas and "confidence" in meas else "",
            "Shadow Width Unclipped (px)": round(w_unclipped, 3) if pd.notna(w_unclipped) else "",
            "Shadow Clip Mask": clip_mode if meas is not None and meas.get("clipped") else "",
            "Shadow Width Clipped Full (px)": round(meas["width_px_clipped_full"], 3) if meas is not None and meas.get("clipped") else "",
            "Shadow Fraction Used": meas["shadow_fraction"] if meas is not None and meas.get("clipped") else "",
            "Shadow Width (px)": round(w_px, 3) if pd.notna(w_px) else "",
            "Shadow Width (m)": round(w_m, 5) if pd.notna(w_m) else "",
            "Depth (m)": round(depth, 5) if pd.notna(depth) else "",
            "Depth (cm)": round(depth * 100, 3) if pd.notna(depth) else "",
            "Crop Detection Confidence": round(box[4], 4) if src == "yolo" else "",
            "Box Source": src,
            "Length (m)": round(length_m, 5) if pd.notna(length_m) else "",
            "Width (m)": round(width_m, 5) if pd.notna(width_m) else "",
            "Surface Area (m2)": round(area_m2, 6) if pd.notna(area_m2) else "",
            "Volume (m3)": round(volume, 6) if pd.notna(volume) else "",
            "Volume (litres)": round(volume * 1000, 3) if pd.notna(volume) else "",
            "Shape Factor": args.shape_factor,
            "Flags": ";".join(dict.fromkeys(flags)),
        })
        d = f"{depth * 100:.1f} cm" if pd.notna(depth) else "n/a"
        v = f", volume {volume * 1000:.3f} L" if pd.notna(volume) else ""
        print(f"[ok]   {name}: shadow {w_px:.1f}px -> depth {d}{v}" if meas else f"[warn] {name}: no shadow found")

    res = pd.DataFrame(out, index=out_idx)
    table = table.drop(columns=[c for c in res.columns if c in table.columns]).join(res)
    try:
        table.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    except PermissionError:
        sys.exit(f"Cannot write '{args.out_csv}'. Close it in Excel and run again.")
    print(f"\nSaved: {args.out_csv} ({len(out)} pothole rows updated, {len(table)} rows total)")
    print("Done. Check depth_debug/ (yellow = pothole boundary, red = clipped shadow) before trusting depth/volume.")


if __name__ == "__main__":
    main()