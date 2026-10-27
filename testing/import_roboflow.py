#!/usr/bin/env python3
"""Import a Roboflow YOLOv8 detection ZIP without leaking validation into training.

Example:
 python import_roboflow.py pothole1.v1-version-1.yolov8.zip
 python import_roboflow.py dataset.zip --dest training/dataset --val-ratio 0.2 --seed 42 --overwrite
"""
import argparse
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def validate_label(data, filename):
    num = 0
    for line_no, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        try:
            cls_float = float(parts[0]); coords = [float(x) for x in parts[1:]]
        except ValueError as exc:
            raise ValueError(f'{filename}:{line_no}: invalid number') from exc
        if cls_float not in (0, 1):
            raise ValueError(f'{filename}:{line_no}: expected class 0 (Pothole) or 1 (Shadow), got {parts[0]}')
        if len(coords) != 4:
            raise ValueError(f'{filename}:{line_no}: expected 4 YOLO bbox coordinates, got {len(coords)}')
        if not all(0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f'{filename}:{line_no}: box coordinates must be normalized with positive size')
        if coords[0]-coords[2]/2 < -0.001 or coords[0]+coords[2]/2 > 1.001 or coords[1]-coords[3]/2 < -0.001 or coords[1]+coords[3]/2 > 1.001:
            raise ValueError(f'{filename}:{line_no}: box extends outside image')
        num += 1
    return num


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip_path', type=Path)
    parser.add_argument('--dest', type=Path, default=Path('training/dataset'))
    parser.add_argument('--val-ratio', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 0.05 <= args.val_ratio <= 0.5:
        parser.error('--val-ratio must be between 0.05 and 0.5')
    if not args.zip_path.is_file() or not zipfile.is_zipfile(args.zip_path):
        parser.error(f'Missing/invalid ZIP: {args.zip_path}')
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()) and not args.overwrite:
        parser.error(f'{dest} is not empty; use --overwrite to replace this dataset')
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(args.zip_path) as archive:
            for entry in archive.infolist():
                # Prevent unsafe ZIP paths and absolute paths.
                location = (root / entry.filename).resolve()
                if not location.is_relative_to(root.resolve()):
                    raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
            archive.extractall(root)
        roots = [r for r in [root, *root.iterdir()] if r.is_dir()]
        source = next((r for r in roots if (r/'train'/'images').is_dir()), None)
        if source is None:
            raise ValueError('ZIP has no train/images; export Roboflow in YOLOv8 format')
        split_pairs = {}
        for split in ('train', 'valid', 'test'):
            idir = source / split / 'images'
            ldir = source / split / 'labels'
            pairs = []
            if idir.is_dir():
                for img in sorted(idir.iterdir()):
                    if img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    label = ldir / (img.stem + '.txt')
                    if not label.exists():
                        # An empty label is legitimate only if the export contains an empty .txt.
                        raise ValueError(f'Missing label file: {label}')
                    # Roboflow occasionally exports a polygon among detection boxes.
                    # Normalize such rows to enclosing xywh boxes for a detect model.
                    normalized=[]
                    for line_no, raw in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                        if not raw.strip():continue
                        vals=raw.split()
                        if len(vals)>5 and (len(vals)-1)%2==0:
                            pts=[float(x) for x in vals[1:]]
                            xs,ys=pts[0::2],pts[1::2]
                            x1,x2=min(xs),max(xs);y1,y2=min(ys),max(ys)
                            raw=f'{vals[0]} {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}'
                            print(f'[normalize] {label.name}:{line_no} polygon -> bbox')
                        normalized.append(raw)
                    normalized_text='\n'.join(normalized)+'\n'
                    boxes = validate_label(normalized_text, str(label))
                    pairs.append((img, label, boxes, normalized_text))
            split_pairs[split] = pairs
        if not split_pairs['train']:
            raise ValueError('No training images found')
        if not split_pairs['valid']:
            originals = split_pairs['train'][:]
            if len(originals) < 5:
                raise ValueError('At least 5 images are required to create a useful validation split')
            random.Random(args.seed).shuffle(originals)
            val_count = max(2, round(len(originals)*args.val_ratio))
            split_pairs['valid'] = originals[:val_count]
            split_pairs['train'] = originals[val_count:]
            print('No validation images in ZIP: creating seeded 80/20-style train/validation split.')
        # Check duplicates between splits based on same filename and content.
        import hashlib
        hashes = {}
        for split, pairs in split_pairs.items():
            for img, _, _, _ in pairs:
                h = hashlib.sha256(img.read_bytes()).hexdigest()
                if h in hashes and hashes[h] != split:
                    raise ValueError(f'Identical image appears in {hashes[h]} and {split}: {img.name}')
                hashes[h] = split
        stage = Path(td) / 'prepared_dataset'
        counts = Counter()
        for split, pairs in split_pairs.items():
            for img, label, boxes, normalized_text in pairs:
                out_img = stage / split / 'images' / img.name
                out_lbl = stage / split / 'labels' / label.name
                out_img.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out_img)
                out_lbl.write_text(normalized_text,encoding='utf-8')
                counts[split] += 1
                counts['boxes'] += boxes
        # Absolute data root makes the YAML usable from any working directory.
        (stage / 'data.yaml').write_text(
            f'path: "{dest.as_posix()}"\ntrain: train/images\nval: valid/images\n'
            + ('test: test/images\n' if counts['test'] else '')
            + 'nc: 2\nnames:\n  0: Pothole\n  1: Shadow\n', encoding='utf-8')
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, dest)
    print(f'Dataset ready: {dest}')
    print(f"train={counts['train']}, valid={counts['valid']}, test={counts['test']}, boxes={counts['boxes']}")
    print(f'Config: {dest / "data.yaml"}')
    print('Next: python train_pothole.py --data training/dataset/data.yaml')


if __name__ == '__main__':
    main()
