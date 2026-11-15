# Recorded video verification before the folder reorganisation

Seven local-video runs completed with the same automation source hash shown in
the evidence below. Four are original visible-light reference clips from the
Halmstad author dataset; three are local demonstration clips. The reference
source/commit, license and per-file hashes are retained in the local source
manifest. These are processing checks, not a held-out accuracy benchmark.

19 unit checks passed: 14 automation policy/playback checks and 5 dataset checks.
All seven generated videos were fully decoded again. Frame counts, CSV records,
analysed/skipped status, class coverage and source-code hashes were checked.

| Clip | Model | Size | Analysed / decoded | Startup s | Processing s | Processing FPS | Rate changes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test01.mp4 | yolov8s.pt | 1280 | 111 / 201 | 7.57 | 10.0 | 20.09 | 1 |
| test02.mp4 | yolov8s.pt | 1280 | 412 / 511 | 8.82 | 24.78 | 20.62 | 4 |
| test04.mp4 | yolov8s.pt | 1280 | 278 / 615 | 8.69 | 71.52 | 8.6 | 4 |
| V_AIRPLANE_001.mp4 | yolov8s.pt | 640 | 327 / 327 | 8.5 | 7.93 | 41.23 | 0 |
| V_BIRD_001.mp4 | yolov8m.pt | 640 | 246 / 305 | 7.97 | 7.47 | 40.84 | 2 |
| V_DRONE_001.mp4 | yolov8x.pt | 640 | 171 / 301 | 5.06 | 6.29 | 47.88 | 3 |
| V_HELICOPTER_001.mp4 | yolov8s.pt | 640 | 301 / 301 | 6.62 | 6.17 | 48.77 | 0 |

Startup calibration is separate from processing time. These times reflect this
machine and workload only. Different runs can select different profiles.

## Observed limitations

The current standard weights support airplane and bird only. They cannot identify
drone or helicopter as those classes, and may mislabel such objects as airplane.
The bird reference clip produced no detections in the earlier reviewed runs;
the counts below record this run rather than hiding that failure. Confidence,
box count and percentage of detected frames are not accuracy. At the time of
these video checks, the downloaded MAT annotations still needed to be scored.

Model and image size are selected at startup. Only the analysis stride adapts
during a run. Automatic SAHI selection and learned model switching are not part
of this implementation. Skipped frames are marked and carry no stale boxes.

Prediction row counts by clip (not unique objects or true positives):
- `test01.mp4`: `{'airplane': 443, 'bird': 1}`
- `test02.mp4`: `{'airplane': 240}`
- `test04.mp4`: `{'airplane': 267}`
- `V_AIRPLANE_001.mp4`: `{'airplane': 319}`
- `V_BIRD_001.mp4`: `{}`
- `V_DRONE_001.mp4`: `{'airplane': 103}`
- `V_HELICOPTER_001.mp4`: `{'airplane': 224}`

Automation source SHA-256: `a3ea3acce3273389e3fb574203416b7f44ef7fb69c1e7c34001e0c8b2e35473d`

## Training log

The later pilot covered civilian airplanes and birds in still images. We
checked 1,263 previously used Halmstad frames for missing labels, invalid boxes,
duplicate images and recording overlap between splits. We also reviewed 340
Open Images V7 candidates and prepared 48 images with YOLO labels. The 22
official test images were held out of both training runs.

| Run | Starting weights | Data | Checkpoint saved locally |
| --- | --- | --- | --- |
| Halmstad pilot | `yolov8n.pt` | 20 training images and 6 validation images from the checked data | `halmstad_matched_experimental.pt` |
| Open Images V7 pilot | `yolov8n.pt` | 20 training images and 6 validation images from the reviewed sample | `open_images_checked_experimental.pt` |

Each run used eight epochs. Both checkpoints and the pretrained YOLOv8n and
YOLOv8s baselines were evaluated on the same 22 test images. The pretrained
YOLOv8n scored highest in this pilot. Checkpoints and measured results are kept
locally under `results/weekly_evaluation/`. The Open Images attribution is
stored with the prepared images. [Open Images V7 data and annotations](https://storage.googleapis.com/openimages/web/download_v7.html)

New training runs write `training_run.json` next to their weights. The record
connects the dataset file and starting model to the saved checkpoint and the
CSV for the selected validation or test split.
