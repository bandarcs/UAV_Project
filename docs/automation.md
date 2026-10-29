# Recorded video automation

`automation.py` is an offline academic experiment for local video files. Supply
one video; the program discovers local detection weights, checks their supported
classes, measures several processing profiles and selects one automatically.
It does not connect to an aircraft or accept camera/stream URLs.

```powershell
.\venv\Scripts\python.exe automation.py test01.mp4
```

No model, image size, tracker, confidence or frame-rate choice is required from
the person running the experiment. `--output-dir` can optionally name a new
results folder. It must not already exist.

## Automatic decisions

1. Read the source dimensions and frame rate and sample three time positions.
2. Measure brightness, contrast and a fixed-scale sharpness indicator.
3. Find trusted local `yolov8*.pt` files beside the program and custom detection
   weights in `weights/academic/`. No weights or data are downloaded.
4. Warm up and time the available 640/960/1280 input profiles. Ultralytics
   performs aspect-preserving letterboxing; the original video is not stretched.
5. Compare compatible class sets, prioritising coverage of airplane, helicopter,
   drone and bird. Within the measured source-rate budget, apply a documented
   size/model preference. These preferences are heuristics, not accuracy scores.
6. Select an analysis stride from measured time. Update this stride after every
   20 analysed frames. Slower analysis increases the stride promptly; two
   consistently faster windows allow the stride to recover without oscillation.
   The selected model
   and input size remain fixed within a run; there is no frame-by-frame model
   switching or learned controller.
7. Detect and track with ByteTrack, write a video and log the decisions.

The annotated output keeps all decoded frames and the original playback rate.
Frames skipped by scheduling are visibly marked **Not analysed** and have no
copied boxes. `frames.csv` distinguishes an unanalysed frame from a frame with
zero detections. CSV coordinates refer to the original input dimensions; the
output video may be reduced to a 1280-pixel long edge. Track IDs are associations,
not guaranteed physical identities, and sampling can affect tracking quality.

## Evidence files

- `calibration.json`: successful and failed timing trials.
- `decision.json`: source measurements, startup choice and supported/missing classes.
- `decisions.csv`: changes in analysis stride during processing.
- `frames.csv`: analysis status of every decoded frame.
- `detections.csv`: detections, confidence, ID and original-image coordinates.
- `annotated.mp4`: visual output.
- `summary.json`: completion status, separate startup/processing times and counts.

An early decoding stop is flagged. Exceptions save a failed/interrupted summary;
the program does not silently report them as successful detection experiments.

## Class support and dataset reference

Local standard COCO weights support airplane and bird, not drone or helicopter.
The program reports these missing classes; filename matching cannot add training.
Output video frames also display the limited supported classes, so a successful
processing run cannot be mistaken for a validated four-class recognition result.
Placing reviewed, compatible custom weights in `weights/academic/` makes them
eligible for calibration without adding a model picker to the user workflow.

The reference for four-class academic data is the
[Halmstad author dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset),
whose video labels are Airplane, Bird, Drone and Helicopter. It publishes visible
and thermal recordings and MATLAB annotations under CC0-1.0. These recordings
have a ground-camera viewpoint, so they do not establish performance from an
airborne camera. Downloading, annotation conversion, reviewed sequence splits,
training and held-out evaluation are separate tasks; this script performs none
of them. VisDrone's pedestrian/vehicle annotations are not four-class airborne
training data.

## Limits of this experiment

Measured timing depends on hardware, other running programs and thermal state.
The timing budget includes a margin but does not guarantee a sustained rate.
Initial frame-quality measurements cannot establish classification accuracy or
recover details absent from the recording. No precision, recall or mAP is
reported without reviewed annotations. More confidence or more boxes is not
treated as evidence of better accuracy.

API references: [prediction](https://docs.ultralytics.com/modes/predict/),
[tracking](https://docs.ultralytics.com/modes/track/).

After a run, the default video player opens `annotated.mp4`. Use `--no-open` for batch processing. If no player is installed, the saved file path is printed and results remain available.
