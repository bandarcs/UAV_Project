# UAV Project

University research prototype for detecting and tracking airborne objects in video. This private repository shares the existing source code with the project team.

## Current code

- `detect_airborne_targets.py`: video/camera analysis with Ultralytics YOLO and ByteTrack; writes annotated video and tracking records.
- `track_aircraft.py`: a separate SAHI detection experiment. Despite its filename, it does not implement ByteTrack tracking.
- `detect_plane_better.py`: an earlier airplane detection experiment.
- `predict_drone_or_plane.py`, `train_drone_vs_plane.py`, and `compare_models.py`: earlier prediction and training experiments.
- `Detect airborne_targets.py`: an earlier copy, kept to preserve the existing work. Use `detect_airborne_targets.py` as the main script.
- `run.bat`: the existing launcher for the earlier detection experiment.

The code is an unfinished prototype. The automatic choice of model and analysis method described in the report is a proposed feature. The current `auto` image-size option is a heuristic; it is not a validated accuracy measure. Detection labels depend on the supplied model weights. Standard COCO weights do not provide every airborne class discussed in the project. Past experiments on VisDrone do not validate airborne-object recognition.

## Run the existing main script

Use a local video that you are allowed to analyse and a compatible weights file. Videos, datasets and model weights are deliberately not stored in this Git repository.

```powershell
.\venv\Scripts\python.exe detect_airborne_targets.py --help
.\venv\Scripts\python.exe detect_airborne_targets.py --source "path\to\video.mp4" --weights "path\to\model.pt" --imgsz 1280
```

The second command is an example, not a newly verified experiment. The model path must point to an existing weights file. Outputs are written under `runs/airborne` by default.

The existing development environment uses Python 3.14.8. `requirements.txt` records its main package versions. The GPU machine uses the CUDA 12.8 builds of PyTorch and torchvision. A clean installation on another computer has not yet been verified; match the PyTorch build to that computer before running experiments.

## Work as a team

After the repository owner adds your GitHub account as a collaborator, clone the repository:

```powershell
git clone https://github.com/bandarcs/UAV_Project.git
cd UAV_Project
git switch -c your-change-name
```

Commit changes on your branch, push the branch, and open a pull request for review. The repository stays private. A link alone does not give another account access.

Do not commit videos, datasets, weights, credentials, or the virtual environment. Record the source and licence of any material used in future experiments. No new training, simulation, or evaluation was performed as part of setting up this repository.
