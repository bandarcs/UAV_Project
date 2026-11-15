# Project layout

Use [`run.bat`](../run.bat) on Windows. It forwards the menu and any command-line arguments to [`uav_project.py`](../uav_project.py).

| Path | Purpose |
| --- | --- |
| `uav_project.py` | Runs the menu, video replay, automatic model and size selection, detection, tracking, and result saving. |
| `tools/prepare_dataset.py` | Checks image labels, class names, and data splits. |
| `tools/train_detector.py` and `tools/compare_models.py` | Train or compare models with a reviewed dataset. |
| `experiments/detect_airborne_targets.py` | Manual detection and tracking experiment. |
| `experiments/detect_video_sahi.py` | Separate detection experiment using image slices. |
| `tests/` | Checks for the analysis policy and dataset preparation. |

Start at `main` in `uav_project.py`, then follow `execute` to see which task runs. The `run` function reads sample frames, picks the settings automatically, and writes the results. Explanations sit beside the relevant statements as Python `#` comments. Training, experiments, and software checks have their own entry points.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The old `uav_session.py` command is now `uav_project.py`. To replace the old `automation.py <video>` command, use `run.bat <video>`. The demo, saved-clip input, replay and clip-splitting options are described in the [session guide](session_workflow.md).
