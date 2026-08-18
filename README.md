# ADAS Object Detection & Collision Avoidance

Calibrated camera-LIDAR fusion, KITTI-fine-tuned detection, and a
time-to-collision risk engine — built as a perception module in the same
category as Forward Collision Warning / early-AEB systems and fleet
safety-analytics pipelines.

**Model on Hugging Face Hub:** [mokshhere/adas-kitti-yolo11m](https://huggingface.co/mokshhere/adas-kitti-yolo11m)
**Container:** `docker pull ghcr.io/ishaannk/adas-rebuild:latest`
**Weights (direct download):** [Release weights-v1](https://github.com/ishaannk/adas-rebuild/releases/tag/weights-v1)

This started as a Colab prototype and has been rebuilt end-to-end: real
sensor calibration instead of a shortcut, a detector fine-tuned on KITTI's
own label taxonomy instead of raw COCO classes, and a risk engine driven by
tracked closing speed instead of a static per-frame distance threshold.

## Before / after

The original prototype used a generic COCO-pretrained detector. Fine-tuning
on KITTI didn't just raise a metric — it changed *what the model considers
a real object*. KITTI's own ground truth for this frame contains exactly
one labeled object (a pedestrian, no bicycles). The COCO-pretrained model,
detecting a generic "bicycle" class, boxes every parked bike in the rack.
The fine-tuned model learned KITTI's actual annotation semantics and
correctly ignores them:

| Before — COCO-pretrained | After — fine-tuned on KITTI |
|---|---|
| ![before](docs/images/fusion_before_coco_pretrained.png) | ![after](docs/images/fusion_after_kitti_finetuned.png) |

Both images show the corrected fusion pipeline (real calibration, frustum-
constrained LIDAR association) — the difference is only the detector's
training data.

## Industry use cases

This is one perception core wrapped two different ways, because those are
the two shapes this kind of system actually ships in:

- **In-vehicle perception module (edge, real-time).** Forward Collision
  Warning / early-AEB decision support — the same category of function as
  aftermarket ADAS dashcams (Mobileye-style) and OEM/Tier-1 L2 stacks. This
  shape needs a hard latency budget and an embedded-friendly export path,
  which is why the detector is also benchmarked as ONNX.
- **Fleet safety analytics (cloud, batch).** The same pipeline as an
  offline service — ingest dashcam or fleet footage, emit near-miss and
  risk events. This is what telematics and insurance fleet-safety vendors
  run for driver scoring and usage-based insurance.
- **OEM/Tier-1 offline validation.** Testing detection and fusion
  algorithms against a public benchmark (KITTI) before they ever touch
  embedded ECU hardware — standard practice before deploying anything to
  a vehicle.

## Architecture

```
Camera frame            LIDAR sweep (.bin velodyne)
     │                         │
     ▼                         ▼
Detector (fine-tuned)   Calibrated projection
  2D boxes               (P2 · R0_rect · Tr_velo_to_cam)
     │                         │
     └───────────┬─────────────┘
                  ▼
      Frustum fusion (points inside box, median depth)
                  │
                  ▼
      Multi-object tracker (persistent IDs, relative velocity)
                  │
                  ▼
      Risk engine (time-to-collision)
                  │
                  ▼
            Alert / report
```

## What was actually fixed

The original prototype (`notebooks/adas_prototype_original.py`) used KITTI's
ground-truth 3D box location as if it were a raw LIDAR point cloud, applied
no camera calibration, and matched distance by nearest point *anywhere in
the frame* rather than inside the detected box — so a pedestrian could
silently inherit a truck's distance. Full list of what was wrong and how
each was fixed is in [`docs/PLAN.md`](docs/PLAN.md).

## Results

**Detector**: YOLO11m fine-tuned on KITTI's own taxonomy (Car, Van, Truck,
Pedestrian, Person_sitting, Cyclist, Tram, Misc), 100 epochs on 2× GPU.

| | mAP50 | mAP50-95 |
|---|---|---|
| Ultralytics validation, all classes | 0.946 | 0.781 |

KITTI-protocol-style eval (easy/moderate/hard difficulty tiers, per-class
IoU thresholds — see `src/adas/eval/kitti_eval.py`):

| | Easy | Moderate | Hard |
|---|---|---|---|
| Mean AP | 0.955 | 0.934 | 0.928 |

**Latency** (imgsz=960):

| Backend | Latency | FPS |
|---|---|---|
| PyTorch, GPU | 23.3 ms | 42.9 |
| ONNX Runtime, CPU | 22.3 ms | 44.8 |

ONNX on CPU alone matches GPU throughput — a good sign for deployment
without dedicated accelerator hardware. Full numbers, per-class breakdowns,
and the training log are in [`docs/PLAN.md`](docs/PLAN.md).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Download KITTI Object Detection Benchmark (training split only, ~20GB)
bash scripts/download_kitti.sh

# KITTI labels -> YOLO format, deterministic 85/15 train/val split
python scripts/prepare_dataset.py

# Fine-tune (both GPUs if available)
python scripts/train.py --device 0,1

# KITTI-protocol eval + latency + ONNX export
python scripts/evaluate.py --weights runs/detect/kitti_finetune/weights/best.pt
python scripts/benchmark_latency.py --weights runs/detect/kitti_finetune/weights/best.pt
python scripts/export_onnx.py --weights runs/detect/kitti_finetune/weights/best.pt

# Visualize the corrected fusion pipeline on a few frames
python scripts/run_fusion_demo.py --weights runs/detect/kitti_finetune/weights/best.pt

python -m pytest tests/
```

## Repo structure

```
src/adas/
  data/        KITTI reader, calibration parsing, YOLO conversion
  detection/   pinned ultralytics detector wrapper
  fusion/      calibrated LIDAR projection + frustum association
  tracking/    multi-object tracker (reuses ultralytics ByteTrack/BoT-SORT)
  risk/        time-to-collision risk engine
  eval/        KITTI-protocol-style evaluation
scripts/       CLI entry points for every step above
configs/       dataset.yaml
tests/         unit tests, no real KITTI data required
notebooks/     original Colab prototype, kept for provenance
docs/PLAN.md   full rebuild plan, diagnostics, and results
Dockerfile     FastAPI inference service (see Deployment below)
space/         Hugging Face Space demo (Gradio) — built and tested, not yet deployed
```

## Deployment

Two independently versioned pieces, on purpose — see `docs/PLAN.md` for why:
the model lives on Hugging Face Hub, the runtime is a Docker image that
fetches it at startup.

```bash
# Prebuilt image (published by .github/workflows/docker.yml on every version tag)
docker pull ghcr.io/ishaannk/adas-rebuild:latest
docker run -p 8000:8000 ghcr.io/ishaannk/adas-rebuild:latest

# Or build locally
docker build -t adas-perception .
docker run -p 8000:8000 adas-perception

curl http://localhost:8000/health

curl -X POST http://localhost:8000/analyze -F "image=@frame.png"

# Full calibrated fusion — image + matching KITTI calib + velodyne
curl -X POST http://localhost:8000/analyze \
  -F "image=@frame.png" -F "calib=@calib.txt" -F "velodyne=@frame.bin"
```

The image is CPU-only and ONNX-backed on purpose: ONNX Runtime on CPU alone
benchmarked at 44.8 FPS, matching GPU PyTorch throughput (see Results above),
so no CUDA base image is needed for real-time performance.

**HF Space demo**: `space/` has a Gradio demo (upload-and-detect, plus a
"real KITTI frame" tab running the full fusion pipeline on bundled sample
data) — built and validated locally end-to-end, but not yet deployed, since
Hugging Face requires a PRO subscription to host Docker/Gradio Spaces even
on free `cpu-basic` hardware. `python scripts/publish_space.py --repo-id
<your-username>/<space-name>` publishes it once that's sorted out.

## Known limitations

- **KITTI's license** restricts this dataset to non-commercial research use.
  The methodology and pipeline here transfer to a real product; these
  specific KITTI-trained weights are not licensed for commercial deployment
  as-is.
- **Ultralytics YOLO11 is AGPL-3.0.** Fine for research/portfolio use; a
  commercial deployment needs an Ultralytics Enterprise license or a
  permissively-licensed alternative detector.
- **Time-to-collision is unit-tested on synthetic sequences, not validated
  on real KITTI multi-frame data.** The KITTI Tracking Benchmark (sequences
  + ego-motion) kept for future advancements 

## Credits

Dataset: [KITTI](https://www.cvlibs.net/datasets/kitti/) (Geiger et al.).
Detector: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).
Original prototype: `notebooks/adas_prototype_original.py`.
