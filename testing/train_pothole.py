#!/usr/bin/env python3
"""Train and evaluate a two-class pothole + shadow YOLO detector; publish best.pt.

Example: python train_pothole.py --data training/dataset/data.yaml --device 0
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('training/dataset/data.yaml'))
    p.add_argument('--model', default='yolov8s.pt', help='Pretrained detection model, e.g. yolov8n.pt or yolov8s.pt')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--batch', type=int, default=4, help='Reduce to 2 or 1 if GPU runs out of memory')
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--device', default=None, help='0 for NVIDIA GPU, cpu for CPU; auto if omitted')
    p.add_argument('--workers', type=int, default=0, help='0 is safer on Windows')
    p.add_argument('--project', default='runs/pothole')
    p.add_argument('--name', default='yolov8s_v1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=Path('best.pt'))
    p.add_argument('--resume-from', default=None, help='Previous trained best.pt to fine-tune on a new dataset')
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f'Missing dataset YAML: {args.data}. Run import_roboflow_updated.py first.')
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        p.error(f'Missing dependency ({exc}). Install with pip install ultralytics torch')
    device = args.device if args.device is not None else ('0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | data: {args.data.resolve()}')
    model = YOLO(args.resume_from or args.model)
    results = model.train(
        data=str(args.data.resolve()), task='detect', epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        device=device, workers=args.workers, project=str(Path(args.project).resolve()),
        name=args.name, exist_ok=False, seed=args.seed, deterministic=True,
        pretrained=True, optimizer='auto', amp=True, cache=False,
        mosaic=0.5, close_mosaic=10, mixup=0.0, degrees=10.0,
        translate=0.1, scale=0.35, fliplr=0.5, flipud=0.2,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, plots=True,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise FileNotFoundError(f'Best checkpoint not found: {best}')
    print(f'\nValidating best checkpoint: {best}')
    trained = YOLO(str(best))
    metrics = trained.val(data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch,
                          device=device, split='val', plots=True)
    print(f'Validation: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
    print('Class IDs: 0=Pothole; 1=Shadow')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    print(f'\nBEST WEIGHTS READY: {args.out.resolve()}')
    print(f'Original training run: {run_dir}')
    print('Use with: python pothole_pipeline.py --images-dir images --excel main.xlsx --weights best.pt')


if __name__ == '__main__':
    main()
