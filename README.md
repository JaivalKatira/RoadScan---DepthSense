# RoadScan — DepthSense

**Pothole detection, shadow-based depth estimation, and approximate volume reporting from road photographs.**

RoadScan — DepthSense combines a custom-trained **YOLOv8 object detector** with image processing and solar-position calculations. Users can upload a pothole photograph, supply its capture metadata, and generate a **PDF report** and **CSV results**.

> **Important:** Depth and volume are estimates based on detected regions, image scale, capture metadata, and visible shadows. This is a research/prototype workflow, **not a validated surveying or engineering measurement tool**. Results depend heavily on image quality, camera geometry, metadata accuracy, and shadow visibility.

## Features

- Two-class object detection: **Pothole** and **Shadow**.
- Import and validate YOLOv8-format Roboflow datasets.
- Train, validate, and export a YOLOv8 detector.
- Estimate pothole depth using shadow geometry and solar elevation.
- Estimate affected area and approximate volume.
- Generate downloadable **PDF** and **CSV** reports.
- Use a browser-based **FastAPI** interface or run a single analysis from the command line.

## Repository structure

```text
RoadScan---DepthSense/
├── app/
│   └── app.py                 # FastAPI web app and upload/download endpoints
├── testing/
│   ├── import_roboflow.py     # Import and verify a Roboflow YOLOv8 ZIP
│   └── train_pothole.py      # YOLOv8 training, validation and export
├── build.py                  # Orchestrates the complete analysis for one photo
├── pothole_pipeline.py       # Pothole detection and crop generation
├── pothole_volume.py         # Pothole segmentation/mask and area processing
├── pothole_depth.py          # Shadow measurements, depth and volume estimates
├── pothole_report.py         # PDF report generation
└── solar_shadow_update.py    # Solar elevation / shadow geometry calculations
```

**Not included in the repository ZIP:** the training dataset, trained `.pt` model weights, a dependency lockfile, or sample output files. Supply these locally as described below.

## 1. Requirements and installation

- Python **3.10 or newer** (a modern version such as 3.11 is recommended).
- `pip` and internet access for initial package/model installation.
- NVIDIA CUDA-capable GPU **recommended for training**; CPU training is possible but much slower.
- Windows PowerShell commands are shown below; the Python commands also work on macOS/Linux after activating the environment appropriately.

Clone the repository:

```powershell
git clone https://github.com/JaivalKatira/RoadScan---DepthSense.git
cd RoadScan---DepthSense
```

Create and activate a virtual environment (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the packages used by the scripts:

```powershell
python -m pip install ultralytics torch torchvision numpy pandas openpyxl opencv-python Pillow reportlab fastapi "uvicorn[standard]" python-multipart
```

**GPU users:** install the PyTorch build matching your NVIDIA driver/CUDA environment following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The generic command above does not guarantee a CUDA-enabled installation.

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell blocks activation, you can run `.venv\Scripts\python.exe` directly instead of activating.

## 2. Prepare the training dataset

The model expects **two detection classes** with these exact IDs:

| Class ID | Class name |
|---|---|
| `0` | `Pothole` |
| `1` | `Shadow` |

1. Prepare and annotate images (including bounding boxes for potholes and their visible shadows) in [Roboflow](https://roboflow.com/) or another compatible tool.
2. Export the dataset as a **YOLOv8 object detection ZIP**, containing `train/images` and corresponding `train/labels`. Existing `valid`/`test` splits are supported.
3. Save the downloaded ZIP in the repository folder; for example, name it `dataset.zip`.
4. From the **repository root**, run:

```powershell
python testing/import_roboflow.py dataset.zip --dest training/dataset
```

This import script checks image/annotation pairs, verifies the two class IDs, converts supported polygon labels into bounding boxes, checks duplicate images across splits, and creates `training/dataset/data.yaml`. If the export lacks a validation split, it creates one from training data (at least five initial training images are required for this case).

To deliberately replace an existing imported dataset:

```powershell
python testing/import_roboflow.py dataset.zip --dest training/dataset --overwrite
```

**Do not use `--overwrite` unless you intend to replace that local dataset.** Keep training, validation and test images independent for meaningful evaluation.

## 3. Train the YOLOv8 model

Run the training script from the **repository root**:

### GPU training (NVIDIA device 0)

```powershell
python testing/train_pothole.py --data training/dataset/data.yaml --model yolov8s.pt --epochs 150 --imgsz 960 --batch 4 --device 0
```

### CPU training

```powershell
python testing/train_pothole.py --data training/dataset/data.yaml --model yolov8s.pt --epochs 150 --imgsz 960 --batch 2 --device cpu
```

Defaults in `testing/train_pothole.py` include 150 epochs, image size 960, batch size 4, early-stopping patience 30, and zero dataloader workers (a conservative Windows setting). Training also runs validation and prints **mAP@50** and **mAP@50–95**.

Useful options:

| Option | Purpose |
|---|---|
| `--model yolov8n.pt` | Train a smaller, potentially faster model |
| `--epochs 100` | Change training epochs |
| `--imgsz 640` | Reduce training image size |
| `--batch 2` | Reduce GPU memory usage |
| `--device cpu` | Train without a CUDA GPU |
| `--name experiment_2` | Give the run a new name |
| `--out weights/best.pt` | Export the best model to a specific path |
| `--resume-from path/to/best.pt` | Start a new fine-tuning run from existing weights |

By default, training saves experiment artifacts under `runs/pothole/` and copies the best validated weights to **`best.pt` in the repository root**. The `--resume-from` flag initializes a *new* training run from saved weights; it is not a full optimizer-state resume.

Before using the model, examine validation metrics and predictions on held-out images. Training successfully does **not** guarantee accurate real-world pothole depth or volume estimates.

## 4. Start the web application

The web app requires trained YOLO weights. If you just trained, `best.pt` should exist in the repository root. Otherwise, obtain a compatible trained `best.pt` file first.

From the repository root, in **PowerShell**:

```powershell
$env:WEIGHTS = (Resolve-Path .\best.pt).Path
python -m uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Open in your browser:

**http://127.0.0.1:8000**

To check whether the web service is responding, visit:

**http://127.0.0.1:8000/healthz**

The application supports `.jpg`, `.jpeg`, and `.png` uploads and defaults to a **20 MB** maximum upload size.

> **Why set `WEIGHTS` explicitly?** `app/app.py` resolves its default relative model path from the `app/` directory, while the training script writes `best.pt` to the repository root. Passing the absolute file path avoids a model-not-found error.

### How to analyze an image in the browser

1. Open the local web URL above.
2. Upload a clear photograph of the pothole.
3. Enter the required capture metadata: **latitude, longitude, true-north direction, capture date, capture time, and camera height above the road (metres)**.
4. Submit the form. The app reports progress as it processes the photo.
5. On successful completion, download the **PDF report** and **CSV results**.

**Metadata guidance:** Use the actual image location, time, direction convention and capture height. `direction` must follow the repository's `gps Direction(True North)` convention; do not automatically substitute phone compass heading without checking it. Date/time should represent the photograph's actual capture conditions. Shadows must be sufficiently visible and usable for shadow-based depth calculation.

To stop the local server, press **Ctrl+C** in the PowerShell terminal. The `127.0.0.1` address works on **the computer running the server**, not automatically on other devices or the public internet. Remote access requires a separately configured and secured deployment.

## 5. Alternative: analyze one photo from the command line

The `build.py` script runs the complete pipeline without using the website. After placing your trained weights at `best.pt`, run from the repository root:

```powershell
python build.py --image .\photo.jpg --latitude 23.02 --longitude 72.57 --direction 90 --date 15-03-2026 --time 11:30 --height 1.5 --weights .\best.pt --job-dir .\jobs\example
```

Replace the example coordinates, direction, capture date/time, height and image path with **your actual values**. The example values are placeholders, not recommended calibration settings.

The job folder will contain the generated `report.pdf`, `result.csv`, logs and intermediate output (if the analysis completes successfully).

## 6. Processing workflow

```text
Input photograph + capture metadata
                 |
                 v
      solar_shadow_update.py
       Solar elevation/ratio
                 |
                 v
        pothole_pipeline.py
        Detect/crop potholes
                 |
                 v
         pothole_volume.py
        Segment pothole area
                 |
                 v
          pothole_depth.py
     Shadow, depth, volume estimates
                 |
                 v
         pothole_report.py
          PDF + CSV results
```

`build.py` orchestrates these scripts and stores each web submission in a separate job directory. This pipeline requires usable detection, scale and shadow information; otherwise the job may fail rather than return a reliable estimate.

## 7. Web API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Browser interface |
| `/healthz` | `GET` | Service health response |
| `/api/jobs` | `POST` | Upload photo and metadata; returns a job ID |
| `/api/jobs/{job_id}` | `GET` | Poll processing status |
| `/api/jobs/{job_id}/report.pdf` | `GET` | Download the finished PDF |
| `/api/jobs/{job_id}/result.csv` | `GET` | Download the finished CSV |

The upload API accepts multipart form fields `photo`, `latitude`, `longitude`, `direction`, `date`, `time`, and `height`. Jobs are processed through the application's background worker; `POST /api/jobs` does not immediately return the final report.

Optional environment variables include `WEIGHTS`, `JOBS_DIR`, `FOV_DEG`, `SHADOW_FRACTION`, `MAX_UPLOAD_MB`, and `JOB_TTL_HOURS`. Defaults can be reviewed near the top of `app/app.py`.

## 8. Troubleshooting

| Problem | What to check |
|---|---|
| `Missing dataset YAML` | Import the dataset first; run commands from the repository root. |
| `Model weights not found` | Train/export `best.pt` and set `WEIGHTS` to its absolute path. |
| `No module named ...` | Activate the correct virtual environment and install the dependencies. |
| CUDA out-of-memory | Lower `--batch` (try 2 or 1) and/or `--imgsz`; use a smaller YOLO model. |
| CPU training very slow | Use a supported CUDA-capable GPU if available. |
| No pothole detected | Check image quality and whether the trained model detects similar roads/potholes. |
| No usable shadow / depth | Use a photo with visible shadow; verify correct time, location, north direction and capture height. |
| Browser cannot connect | Confirm the server is still running and visit `http://127.0.0.1:8000`. |
| Processing fails | Inspect the job folder's `log.txt` and job status for the failed pipeline step. |

## 9. Notes and limitations

- The provided repository code is a **prototype**; site-specific testing and calibration are necessary before relying on measurements.
- Shadow-based estimation can fail for diffuse lighting, darkness, weak or occluded shadows, and inaccurate acquisition metadata.
- Volume estimates should not be treated as ground-truth pavement repair quantities without independent measurement.
- Training data quality, bounding-box accuracy, domain coverage, and held-out validation significantly affect performance.
- The repository ZIP does not include a license file. Unless a license is added, reuse/redistribution permissions are **not automatically granted**.

## Maintainer

GitHub: [JaivalKatira](https://github.com/JaivalKatira)  
Repository: [RoadScan---DepthSense](https://github.com/JaivalKatira/RoadScan---DepthSense)
