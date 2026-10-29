# Dataset reference for the four-class study

The requested classes are **airplane, helicopter, drone and bird**. A dataset
captured by a drone is not necessarily a dataset of flying objects.

| Source | Labelled objects | Use in this study |
| --- | --- | --- |
| [Halmstad author repository](https://github.com/DroneDetectionThesis/Drone-detection-dataset) | Airplane, Bird, Drone, Helicopter in video | Primary four-class reference. Visible and IR modalities must be handled separately. Ground-camera viewpoint; it does not establish airborne-camera performance. |
| [MOT-FLY author repository](https://github.com/CZC-123/MOT-FLY) | Flying UAVs, with tracking identities | Additional air-to-air UAV tracking reference; it does not supply all four classes. |
| [VisDrone documentation](https://docs.ultralytics.com/datasets/detect/visdrone/) | Pedestrians, people, bicycles and vehicles | Not training or evaluation data for the four requested flying-object classes. |

The Halmstad repository publishes CC0-1.0 and MATLAB video-label annotations.
Refer to its instructions and original publication for the contents and label
format. A bibliography entry is not evidence that a dataset has been downloaded,
converted, trained or evaluated locally. The local audit found VisDrone images;
no prepared four-class training set was found.

`prepare_dataset.py` accepts reviewed YOLO-format detection labels. It does not
read the original MATLAB annotation objects or generate reliable labels from
model predictions. Dataset conversion and validation are prerequisites to any
four-class training run. Keep each recording in a single train/validation/test
split to avoid near-duplicate frames leaking across evaluation partitions.

Reference checked: 28 October 2026.
