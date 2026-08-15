# ADAS Perception Rebuild — Plan

Full design rationale and industry framing: see the published roadmap artifact
(link kept in project notes). This file is the working reference committed
alongside the code.

## Diagnostics — bugs in the original prototype (`notebooks/adas_prototype_original.py`)

| Code | Severity | Issue |
|------|----------|-------|
| DTC-01 | Critical | `objects.location` (KITTI ground-truth 3D box position) was used as if it were a raw LIDAR point cloud. No `.bin` velodyne file was ever loaded. |
| DTC-02 | Critical | Pixel/point association divided raw 3D coordinates with no `P2`/`R0_rect`/`Tr_velo_to_cam` calibration applied — "distance" was not metrically meaningful. |
| DTC-03 | Moderate | Distance assigned by nearest point in the *whole frame*, not constrained to the detection's box — a pedestrian could inherit a truck's distance. |
| DTC-04 | Moderate | No tracking across frames — no relative velocity, so a stationary and a fast-closing object at the same distance looked identical. |
| DTC-05 | Minor | "Real-time" claimed but never implemented — static dataset frames processed one at a time. |
| DTC-06 | Minor | `torch.hub.load('ultralytics/yolov5', ...)` re-clones GitHub at runtime instead of a pinned package. |
| DTC-07 | Minor | No packaging, tests, CI, or license. |

Fixes: DTC-01/02/03 → `src/adas/fusion/`. DTC-04 → `src/adas/tracking/` +
`src/adas/risk/ttc.py`. DTC-06 → `src/adas/detection/detector.py`.

## Data

KITTI **Object Detection Benchmark**, training split only (7,481 frames:
images + labels + calib + velodyne). The benchmark's "testing" split ships
no ground-truth labels (held out for KITTI's submission server) and is never
extracted, to save disk.

**Deliberately skipped:** the KITTI Tracking Benchmark (sequences + oxts
ego-motion, ~94GB) — this workspace's disk quota can't absorb it alongside
the existing project data already on the volume. Consequence: P4's
time-to-collision logic is validated with synthetic multi-frame unit tests
(`tests/test_ttc.py`), not real KITTI sequences. Revisit if more disk frees
up — a single small raw KITTI "drive" sequence (a few hundred MB to ~2GB)
would unlock genuine end-to-end temporal validation without the full
Tracking Benchmark's footprint.

Train/val split: deterministic 85/15 over sorted frame ids, fixed seed 42
(`src/adas/data/convert_to_yolo.py`) — not the literature-standard Chen et
al. 3712/3769 split, since that id list wasn't available offline.

**License flag:** KITTI's terms restrict use to non-commercial research.
The methodology and pipeline here transfer to a real product; the
KITTI-trained weights themselves are not shippable commercially without
separate data/licensing.

## Model

`ultralytics` YOLO11m, fine-tuned on KITTI's own taxonomy (Car, Van, Truck,
Pedestrian, Person_sitting, Cyclist, Tram, Misc) rather than reported on raw
COCO classes. Trained on both available GPUs via ultralytics' DDP support.

**License flag:** Ultralytics YOLO11 ships AGPL-3.0. Fine for a demo; a
commercial deployment needs an Ultralytics Enterprise license or a
permissively-licensed alternative detector.

## Fusion, tracking, risk

- `src/adas/fusion/projection.py` — calibrated velodyne → image-plane projection.
- `src/adas/fusion/frustum.py` — box-constrained point association, median depth.
- `src/adas/tracking/tracker.py` — thin wrapper over ultralytics' ByteTrack/BoT-SORT (reused, not reimplemented).
- `src/adas/risk/ttc.py` — per-track distance history → closing speed → time-to-collision, alert tiers shaped after Euro NCAP AEB test-protocol bands (brake / warn / monitor).

## Results

**Training**: YOLO11m, 100 epochs, both GPUs, 2.506 hours, `runs/detect/kitti_finetune/`.

Ultralytics validation (COCO-style, mAP50 / mAP50-95, on the held-out val split):

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.946 | 0.781 |
| Car | 0.980 | 0.879 |
| Van | 0.976 | 0.868 |
| Truck | 0.984 | 0.899 |
| Pedestrian | 0.901 | 0.571 |
| Person_sitting | 0.845 | 0.635 |
| Cyclist | 0.940 | 0.734 |
| Tram | 0.975 | 0.855 |
| Misc | 0.964 | 0.808 |

**KITTI-protocol-inspired eval** (`scripts/evaluate.py`, easy/moderate/hard tiers, per-class IoU thresholds, same-class ignore-regions for out-of-tier GT — see `src/adas/eval/kitti_eval.py`):

| Class | Easy | Moderate | Hard |
|---|---|---|---|
| mean | 0.955 | 0.934 | 0.928 |
| Car | 0.997 | 0.993 | 0.976 |
| Van | 0.989 | 0.973 | 0.971 |
| Truck | 1.000 | 1.000 | 1.000 |
| Pedestrian | 0.973 | 0.921 | 0.871 |
| Person_sitting | 0.750 | 0.694 | 0.756 |
| Cyclist | 0.984 | 0.943 | 0.922 |
| Tram | 0.981 | 0.993 | 0.993 |
| Misc | 0.968 | 0.957 | 0.933 |

(Minor easy < moderate/hard inversions on Tram and Person_sitting are sampling
noise from tiny class counts — 72 and 44 instances respectively — not a bug;
first implementation *did* have a real bug here: same-class GT boxes that
were real objects but too small/occluded for a given tier were counted as
false positives instead of ignored, which inverted the whole easy/moderate/
hard ordering. Caught via a sanity check that the ordering should never
invert on a uniformly-performing detector; regression-tested in
`tests/test_kitti_eval.py`.)

**Latency** (`scripts/benchmark_latency.py`, `scripts/export_onnx.py`, imgsz=960):

| Backend | Latency | FPS |
|---|---|---|
| PyTorch, GPU | 23.3 ms/frame | 42.9 |
| PyTorch, CPU | 396.7 ms/frame | 2.5 |
| ONNX Runtime, CPU | 22.3 ms/frame | 44.8 |

ONNX Runtime on CPU alone matches GPU PyTorch throughput — a good sign for
edge/embedded deployment without dedicated accelerator hardware.

## Roadmap status

- [x] P1 — repo foundation, KITTI data pipeline, calibration parsing
- [x] P2 — YOLO11m fine-tuned, 100 epochs, results above; published to Hugging Face Hub as `mokshhere/adas-kitti-yolo11m` (both `.pt` and `.onnx`)
- [x] P3 — fusion correctness fix (calibrated projection + frustum association), unit-tested, visually validated on real KITTI frames
- [x] P4 — TTC risk engine + tracker wrapper implemented and unit-tested; real multi-frame validation deferred (no Tracking Benchmark data — open item)
- [x] P5 — KITTI-protocol eval, latency/FPS benchmarking, ONNX export — results above
- [~] P6 — Docker service (FastAPI, CPU/ONNX, fetches the model from HF Hub at
  startup) built and validated end-to-end against the real published model.
  Published to GHCR (`ghcr.io/ishaannk/adas-rebuild`) via
  `.github/workflows/docker.yml`, which also runs the real image and hits
  `/health` + `/analyze` as a smoke test before considering the build good.
  Weights also mirrored as a GitHub Release (`weights-v1`) for direct
  download without a Hugging Face account. HF Space demo (`space/`) built
  and validated locally, but not deployed — Hugging Face requires a PRO
  subscription to host Docker/Gradio Spaces even on free `cpu-basic`
  hardware; publishing it is a one-command follow-up
  (`scripts/publish_space.py`) once that's sorted out.
