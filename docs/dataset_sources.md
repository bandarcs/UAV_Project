# Airborne dataset sources

The project labels are **airplane, helicopter, drone and bird**.

| Source | Viewpoint and labels | Role |
| --- | --- | --- |
| [Halmstad dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8) · [Authors' paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8573135/) | Ground camera. RGB videos have frame-level boxes for all four project classes. The published repository uses CC0-1.0. | Separate RGB recordings for the training pilot and validation. |
| [AOD-4](https://data.mendeley.com/datasets/cd5z895tr2/1) | Four aerial-object labels in still images from mixed sources and viewpoints. CC BY 4.0. | Initial training baseline. Excluded files identified as Halmstad derivatives or of unclear provenance. |
| [MOT-FLY](https://github.com/CZC-123/MOT-FLY) | Airborne camera. Labels for multiple UAV targets and their tracks. | Reference for air-to-air UAV tracking. |
| [Amazon AOT](https://registry.opendata.aws/airborne-object-tracking/) | Airborne camera with annotated flying objects and empty frames. | Source to review for viewpoint coverage and background tests. Verify category mapping before merging. |

The Halmstad training pilot uses 510 labelled RGB images sampled from 24 videos.
It contains 128 airplane, 127 helicopter, 128 drone and 135 bird boxes. The
conversion checked annotation timestamps against video timestamps and excluded
incompatible recordings. The 12 validation sequences were kept outside training,
along with their immediate neighbours in the published clip numbering. Validation
uses 753 frames from those 12 sequences. The recordings share a dataset and
ground-camera perspective, so this comparison applies to that setting. Flight
camera validation requires separately labelled footage from an airborne camera.

Split by recording before extracting frames. Review labels and compare all
candidate models on the same held-out validation clips. Keep a separate test split
for the final model.
