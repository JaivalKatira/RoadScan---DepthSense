#!/usr/bin/env python3
"""Pothole AREA step (run BEFORE pothole_depth.py).

Finds the pixel-level pothole boundary in every crop, never a guessed ellipse:
    YOLO segmentation mask > OpenCV GrabCut contour (box-guided).

What it writes
--------------
* <debug-dir>/<crop>_area_mask.png      the pothole mask (0/255). ALWAYS written:
                                        pothole_depth.py reads it to stop each cast
                                        shadow at the pothole boundary.
* <debug-dir>/<crop>_area_overlay.png   mask drawn on the crop (review it).
* CSV columns: Pothole Mask Area (px), Mask Length (px), Mask Width (px), Area Mode,
               Mask File, Area Flags.

Metres, depth and volume need the scale and the clipped shadow, so they are finished
by pothole_depth.py (Surface Area, Volume (m3), Volume (litres), ...).

Usage:
    python pothole_volume.py --weights best.pt --area-mode auto
"""
import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pothole_depth import PotholeDetector, fallback_box, find_col, index_images


def clean_mask(mask, box, shape, min_fraction=0.03):
    """Keep connected pothole component near the detection centre, with smooth edges."""
    h,w=shape[:2]
    x1,y1,x2,y2=[int(round(v)) for v in box[:4]]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    support=np.zeros((h,w),np.uint8)
    support[y1:y2,x1:x2]=1
    mask=(mask.astype(bool)&support.astype(bool)).astype(np.uint8)
    k=max(3, (min(h,w)//160)|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    n, labels, stats, centroids=cv2.connectedComponentsWithStats(mask,8)
    if n<=1: return None
    cx,cy=(x1+x2)/2,(y1+y2)/2
    candidates=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]
        if area<max(25,(x2-x1)*(y2-y1)*min_fraction): continue
        dx=(centroids[i][0]-cx)/max(1,x2-x1)
        dy=(centroids[i][1]-cy)/max(1,y2-y1)
        candidates.append((area/(1+4*(dx*dx+dy*dy)),i))
    if not candidates: return None
    label=max(candidates)[1]
    return (labels==label).astype(np.uint8)


def grabcut_contour(img,box,padding=0.15,iterations=6):
    """Use detection ROI as soft foreground prior, not a rectangular area estimate.

    GrabCut can fail for asphalt that visually matches surrounding road: flags/overlays
    make such failures reviewable, without inventing a geometric shape.
    """
    h,w=img.shape[:2]
    x1,y1,x2,y2=[int(round(v)) for v in box[:4]]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    bw,bh=x2-x1,y2-y1
    if bw<12 or bh<12: return None
    # Crop contains detection with its surrounding road. Pixels outside box are
    # known background, inner box core probable foreground, remainder uncertain.
    margin=max(12,int(max(bw,bh)*0.3))
    ax,ay,bx,by=max(0,x1-margin),max(0,y1-margin),min(w,x2+margin),min(h,y2+margin)
    roi=img[ay:by,ax:bx]
    if min(roi.shape[:2])<16: return None
    mask=np.full(roi.shape[:2],cv2.GC_BGD,np.uint8)
    lx1,ly1,lx2,ly2=x1-ax,y1-ay,x2-ax,y2-ay
    mask[ly1:ly2,lx1:lx2]=cv2.GC_PR_FGD
    # Seed a small centre rather than marking the entire box as foreground.
    sx=max(2,int(bw*0.18)); sy=max(2,int(bh*0.18))
    cx,cy=(lx1+lx2)//2,(ly1+ly2)//2
    mask[max(ly1,cy-sy):min(ly2,cy+sy),max(lx1,cx-sx):min(lx2,cx+sx)]=cv2.GC_FGD
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(roi,mask,None,bg,fg,iterations,cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    binary=np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    full=np.zeros((h,w),np.uint8)
    full[ay:by,ax:bx]=binary
    return clean_mask(full,box,img.shape)


def yolo_mask(pred,box,img):
    """Only class-0 pothole segmentation pixels contribute to area."""
    if pred.masks is None or pred.boxes is None:return None
    classes=pred.boxes.cls.cpu().numpy().astype(int)
    indices=np.where(classes==0)[0]
    if not len(indices):return None
    masks=pred.masks.data.cpu().numpy()
    combined=np.zeros(img.shape[:2],np.uint8)
    for j in indices:
        resized=cv2.resize(masks[j],(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
        combined[resized>0.5]=255
    return clean_mask(combined,box,img.shape,min_fraction=0.005)


def save_overlay(img,mask,name,folder,mode):
    view=img.copy()
    tinted=np.zeros_like(img); tinted[:]=(40,195,70)
    view[mask>0]=cv2.addWeighted(view[mask>0],0.55,tinted[mask>0],0.45,0)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view,contours,-1,(0,0,255),2)
    cv2.putText(view,f"AREA MASK: {mode}",(8,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.imwrite(os.path.join(folder,name+'_area_overlay.png'),view)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv','--depth-csv',dest='csv',default='pothole information.csv')
    ap.add_argument('--crops-dir',default='processed')
    ap.add_argument('--weights',default=None)
    ap.add_argument('--conf',type=float,default=0.10)
    ap.add_argument('--imgsz',type=int,default=None)
    ap.add_argument('--padding',type=float,default=0.15)
    ap.add_argument('--area-mode',choices=['auto','mask','contour'],default='auto',help='auto: YOLO mask, else OpenCV contour. No ellipse/rectangle fallback.')
    ap.add_argument('--debug-dir',default='area_debug',help='Folder for masks (always written) and overlays; pothole_depth.py reads the masks from here')
    ap.add_argument('--out-csv',default='pothole information.csv')
    args=ap.parse_args()
    mask_dir=args.debug_dir or 'area_debug'
    os.makedirs(mask_dir,exist_ok=True)
    if not os.path.isfile(args.csv):sys.exit('CSV not found: run solar_shadow_update.py and pothole_pipeline.py first')
    df=pd.read_csv(args.csv,encoding='utf-8-sig')
    c_img=find_col(df,'image no','image')
    if c_img is None:
        sys.exit('Missing image no column in CSV')
    images=index_images(args.crops_dir)
    det=PotholeDetector(args.weights,args.conf,args.imgsz)
    rows=[]; indices=[]
    for idx,r in df.iterrows():
        if 'Crop File' in df and (pd.isna(r['Crop File']) or str(r['Crop File']).strip()==''):continue
        name=str(r[c_img]).strip()
        file=str(r['Crop File']).strip() if 'Crop File' in df else ''
        path=os.path.join(args.crops_dir,file) if file else None
        if not path or not os.path.isfile(path):path=images.get(name.lower())
        flags=[]
        area_px=length_px=width_px=float('nan'); mode='unavailable'; mask_file=''
        if not path or not os.path.isfile(path):flags.append('crop_not_found')
        else:
            img=cv2.imread(path)
            if img is None:flags.append('unreadable_image')
            else:
                prediction=None
                if det.model is not None:
                    prediction=det.model.predict(img,conf=args.conf,imgsz=args.imgsz or det.imgsz or 960,verbose=False)[0]
                if prediction is not None and prediction.boxes is not None and len(prediction.boxes)>0:
                    classes=prediction.boxes.cls.cpu().numpy().astype(int)
                    valid=np.where(classes==0)[0]
                    if len(valid):
                        boxes=prediction.boxes.xyxy.cpu().numpy()
                        confs=prediction.boxes.conf.cpu().numpy()
                        j=int(valid[np.argmax(confs[valid])])
                        box=tuple(map(float,boxes[j]))+(float(confs[j]),)
                    else:
                        box=fallback_box(img,args.padding)
                        flags.append('volume_used_whole_crop')
                else:
                    box=fallback_box(img,args.padding)
                    flags.append('volume_used_whole_crop')
                mask=None
                if args.area_mode in ('auto','mask') and prediction is not None:
                    mask=yolo_mask(prediction,box,img)
                    if mask is not None:mode='yolo_mask'
                if mask is None and args.area_mode in ('auto','contour'):
                    mask=grabcut_contour(img,box,args.padding)
                    if mask is not None:
                        mode='opencv_contour'
                        flags.append('opencv_contour_review')
                stale=os.path.join(mask_dir,name+'_area_mask.png')
                if mask is None:
                    flags.append('contour_unavailable')
                    if args.area_mode=='mask':flags.append('mask_not_available')
                    if os.path.isfile(stale):os.remove(stale)   # never leave a mask from an earlier run
                else:
                    area_px=int(np.count_nonzero(mask))
                    coords=np.column_stack(np.where(mask>0))
                    if len(coords):
                        py,px=coords[:,0],coords[:,1]
                        length_px=float(max(np.ptp(px)+1,np.ptp(py)+1))
                        width_px=float(min(np.ptp(px)+1,np.ptp(py)+1))
                    x1,y1,x2,y2=[int(round(v)) for v in box[:4]]
                    rim=mask[max(0,y1):min(img.shape[0],y2),max(0,x1):min(img.shape[1],x2)]
                    if rim.size and (np.any(rim[0]) or np.any(rim[-1]) or np.any(rim[:,0]) or np.any(rim[:,-1])):
                        flags.append('contour_touches_box')
                    cv2.imwrite(stale,mask*255)
                    mask_file=name+'_area_mask.png'
                    if args.debug_dir:save_overlay(img,mask,name,mask_dir,mode)
        indices.append(idx)
        rows.append({'Pothole Mask Area (px)':area_px if pd.notna(area_px) else '',
            'Mask Length (px)':length_px if pd.notna(length_px) else '',
            'Mask Width (px)':width_px if pd.notna(width_px) else '',
            'Area Mode':mode,'Mask File':mask_file,
            'Area Flags':';'.join(dict.fromkeys(flags))})
        print(f'{name}: mask {area_px} px ({mode})' if pd.notna(area_px) else f'{name}: no pothole mask ({mode})')
    result=pd.DataFrame(rows,index=indices)
    df=df.drop(columns=[c for c in result.columns if c in df.columns]).join(result)
    df.to_csv(args.out_csv,index=False,encoding='utf-8-sig')
    print(f'Saved {args.out_csv}; masks in {mask_dir}/. Inspect the *_area_overlay.png files, then run: python pothole_depth.py --weights best.pt')


if __name__=='__main__':main()