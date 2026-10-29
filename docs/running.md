# Run the existing prototype

Use a Python environment with the packages in `requirements.txt`. Model weights and input video must exist locally; they are not included in the repository.

```powershell
.\venv\Scripts\python.exe detect_airborne_targets.py --help
.\venv\Scripts\python.exe detect_airborne_targets.py --source "path\to\video.mp4" --weights "path\to\model.pt" --imgsz 1280 --output-dir "runs\airborne\demo_01"
```

Replace the example paths with actual files. A numeric `--source`, such as `0`, selects an OpenCV camera index. This option does not establish a connection to a physical UAV or Gazebo.

The code currently receives model weights and several settings through the command. `--imgsz auto` tries image sizes on sampled positions in a recorded video. A camera or unknown-length source uses the fallback image size. This heuristic is separate from the full automatic AI setup in the report's required design.

Output filenames are fixed. Use a separate `--output-dir` for each run to preserve earlier results. A preview is shown unless `--no-show` is passed. The script saves MP4 and CSV files; it does not provide a saved-session review interface.

The dependency versions record the development environment. A clean installation on another machine has not been verified in this documentation review. Recognition capabilities depend on the loaded weights and their class definitions.

For the separate recorded-video workflow without a model/image-size picker, see [automation](automation.md).
