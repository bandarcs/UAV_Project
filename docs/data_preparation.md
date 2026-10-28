# Data preparation

This utility checks **YOLO detection labels**, not MOT annotations or segmentation labels. Convert other formats using their documented class definitions before using it. It does not download data, infer labels, train a detector, or change raw files.

Create a CSV next to the raw data:

```csv
image,label,sequence,split
images/seq01/0001.jpg,labels/seq01/0001.txt,seq01,train
images/seq02/0001.jpg,labels/seq02/0001.txt,seq02,val
images/seq03/0001.jpg,labels/seq03/0001.txt,seq03,test
```

Each row needs an existing image, an existing label file, and its **source video/sequence ID**. An empty label file explicitly marks an image with no labelled objects. A missing file is an error. The class list must use the original class-ID order.

Provide a split for every row to preserve a reviewed partition. If all split cells are empty, the tool assigns complete sequence groups to approximate 70/20/10 train/validation/test proportions. Exact duplicates are grouped together too. At least three independent groups are needed. Official benchmark test sets should retain their published membership.

```powershell
.\venv\Scripts\python.exe prepare_dataset.py --manifest "datasets\source\manifest.csv" --output "datasets\checked_run_01" --classes airplane helicopter bird --source-url "https://source.example/dataset" --license "LICENSE-AS-PUBLISHED"
```

The command writes `summary.json` and `checked_manifest.csv` into a **new** output directory. The example class names and source URL are placeholders, not a claim about a downloaded dataset.

Add `--materialize` to copy the checked files into `images/{split}` and `labels/{split}` and write `data.yaml`. This may require additional disk space. Images retain their resolution; no augmentation is applied to validation or test data. The utility rejects class-ID errors, invalid normalised coordinates, unreadable images, and sequences or duplicate images shared across splits.

The summary distinguishes **images containing a class** from **labelled object instances** and reports negative images and realised split sizes. It is a dataset inventory, not an accuracy result.

Tests use synthetic files and check label errors, missing annotations, negative images, materialised output and split leakage:

```powershell
.\venv\Scripts\python.exe -m unittest -v test_prepare_dataset
```

The tool does not establish that any aircraft class is present, balanced, correctly labelled, or representative of UAV camera footage. Those properties require inspection of the chosen real dataset.
