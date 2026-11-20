# Data

No data are included in this repository.

## ds004998 — MEG + STN-LFP, OFF/ON levodopa (Düsseldorf)

Rassoulou et al. (2024), *Scientific Data* 11, 889 — OpenNeuro **ds004998** (CC0).
20 patients, simultaneous whole-head MEG, bipolar STN-LFP and EMG at 2 kHz, OFF and ON medication.

Download from OpenNeuro (<https://openneuro.org/datasets/ds004998>), e.g. with the OpenNeuro CLI or DataLad,
and set `DS004998_ROOT` to the dataset root (the folder containing `participants.tsv`).

## Oxford MRC BNDU — STN-LFP, OFF/ON levodopa and DBS

STN-LFP from externalised DBS leads, medication OFF and ON, plus a DBS dataset (`DBS/MATRIX_DBS.mat`), from the
MRC Brain Network Dynamics Unit Data Sharing Platform (<https://data.mrc.ox.ac.uk>). Access is by request and
subject to the platform's terms; **do not redistribute the files or per-patient derived values**.

Expected layout under `OXFORD_BNDU_ROOT` (file names `<subject>_<le|ri>STN_<OFF|ON>.mat`, MATLAB v7.3):

```
oxford_mrc_bndu/
  OFF/...            *_leSTN_OFF.mat, *_riSTN_OFF.mat
  ON part 1/...      *_leSTN_ON.mat,  *_riSTN_ON.mat
  ON part 2/...
  DBS/MATRIX_DBS.mat
```

Files are found recursively; subject, side and medication are parsed from the names. Set `RUN_OXFORD = False`
in §1 to run without this dataset.
