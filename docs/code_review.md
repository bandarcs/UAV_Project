# Code review and entry points

All active Python filenames describe their purpose. Inference programs that
overlapped are consolidated instead of keeping several inconsistent versions.

| Entry point | Purpose |
| --- | --- |
| `automation.py` | Automatic local-video profiling, scheduling, YOLO + ByteTrack, saved results and playback |
| `detect_airborne_targets.py` | Manual fixed-setting baseline with preview and CSV |
| `detect_video_sahi.py` | Separate sliced-inference airplane baseline; detection, not ByteTrack |
| `prepare_dataset.py` | Validate reviewed YOLO labels, sequence splits and duplicates |
| `train_detector.py` | Single-model entry point to the shared training pipeline |
| `compare_models.py` | Multi-model training/evaluation and measured comparison |
| `test_automation.py` | Profile selection, scheduling, class coverage and playback checks |
| `test_prepare_dataset.py` | Annotation integrity, negatives and split-leakage checks |

Removed overlapping scripts: `Detect airborne_targets.py`,
`detect_plane_better.py`, `predict_drone_or_plane.py`. Their previous versions
remain recoverable in Git history. `track_aircraft.py` is now
`detect_video_sahi.py` because it performs sliced detection without tracking.
`train_drone_vs_plane.py` is now `train_detector.py` and shares the comparison
pipeline instead of duplicating training and evaluation logic.

Corrections: no silent overwrite of manual output; civilian bird class in the
manual baseline; SAHI checks file paths, settings and writer creation, releases
resources on exceptions and resolves airplane labels by name rather than assuming
class ID 4 in every model. Automatic scheduling can recover from stride two to
one; incomplete model class coverage stays visible on output video. Playback
failure leaves saved results intact. Training and comparison require explicit
reviewed data; they are not evidence that training has already been performed.

The automated workflow selects a model and size at startup and adjusts stride
during the run. It does not automatically select SAHI or learn a controller.
Passing software checks is not a guarantee of detector accuracy or absence of
all possible faults. The reference-video experiment and observed limitations are
in [verification](automation_verification.md).

Runtime smoke checks: the retained manual and SAHI paths each processed a 12-frame derivative of the author airplane clip. Both output videos were decoded again and contained all 12 frames. These checks establish pipeline execution, not accuracy. Training was not run; its CLI and missing-data rejection were checked.
