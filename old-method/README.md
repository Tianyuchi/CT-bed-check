# Table-Top Defect Detection
## Code
### Project Layout
```
topbed_detection
├── docs/              # Documents like slides.
├── app.py             # Application GUI.
├── ckpt               # Saved checkpoint.
├── collect.py         # Data cleaning and managing.
├── dict_learning.py   # Traing & validate models.
├── process.py         # Deprecated, can be safely removed.
├── utils_io.py        # Utilities for DICOM load/parse/save.
├── negative.csv       # Index of negative samples.
├── negative/          # DICOM files: negative samples.
├── positive.csv       # Index of positive samples.
├── positive/          # DICOM files: positive samples.
├── DICOM_files/       # Unprocessed DICOM, batch 2.
└── dicom-linag/       # Unprocessed DICOM, batch 1.
```

### Core Ideas
Defects in topbed images cannot be restored with sparse coding dictionary trained on normal ones.

## Dataset
Two batches of DICOMs are collected with details described in the following section.

### Batch 1
All selected DICOM files in this batch come with defects.
One of them (with ID 5875) excluded as it is not produced during screening.

### Batch 2
Exported all samples with ID from 6540 to 6860 (inclusive).
A total of 225 folders, usually one sample (3 shots) for each.
No labels or qaulity control provided.
Acquisition date (indicated by DICOM tags) ranges from Aug 13, 2021 to Oct 13, 2021.

#### Quality Control
Convert DICOM with SeriesDescription of `Topogram  0.6  T20f` and convert to PNG for manual check
with the following script. Note the `Dose Report` and `Patient Protocal` are discarded.
 ```bash
 # Reorganize.
 python3 ./collect.py ../DICOM_files ./negative/
 # Copy PNGs to the same folder for the convience of manual check. 
cd ./negative; mkdir all
 find -name *png -printf '%h\n' | while read i ; do cp ${i}/image.png ./all/${i}.png; done
 ```
 
Results:
1. `6595` deleted as it contains 194 DICOM files.
1. `6797` deleted due to corrupted image.
1. `6727`, `6612`, and `6702` deleted due to unknown / irrelevant images.
1. `6609` and `6673` deleted due to defects.

## Train & Val
### Training with normal images only.
```bash
python3 dict_learn.py --train ckpt --data positive.csv
```

### Evaluate with images with defect
```bash
python3 dict_learn.py --eval ckpt --data positive/1.3.12.2.1107.5.1.4.91603.30000021053101180053300002156
```
