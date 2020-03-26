# Iterative consensus spectral clustering for brain functional modules
The code repo for the iterative consensus spectral clustering that performs detection of subject and group level brain functional modules.


## Setup

`pip3 install -r requirements.txt`

## Guide to run the code

1. The data for the subjects resides in the `data` directory. You can create subdir for the dataset in the `data` directory and put the connectivity matrices for each subject. We used the [HCP dataset](http://www.humanconnectomeproject.org). For your reference, we have added some dummy connectivity matrices (synthetic connectivity matrices with 264 nodes) to give an idea of the format we expect the subject data to be in. The data should be a numpy array for each subject, of the shape (N, N) where N is the number of regions of interest. Currently, we have:

* `data/multiple_subjects/` folder containing data for 25 synthetic samples in the form of `subject_*.npy`. You can use this data to obtain group-level modularizations (and individual-level modularizations).

* `data/subject_sessions/` folder containing data for 2 subjects, containing 64 synthetic scan session samples in the form of `data/subject_sessions/subject_*/S*_corr.npy`. You can use this data to obtain subject-level modularizations (and session/individual-level modularizations).

2. The codes reside in the  `codes` folder. We give two scripts to derive group-level modularizations and subject-level modularizations. You can modify the following parameters in the respective files:
- NUM_NODES: the number of ROIs considered
- NUM_RUNS: the number of independent runs you wish to consider
- NUM_THREADS: the number of cores you want to use. Is useless if NUM_RUNS < 2
- DATASET: the folder where the dataset is stored.
- MAX_LABELS, MIN_LABELS: The range for the number of modules for the subject/scans under considerations
- SAVE_DIR: The directory where the results will be saved. 

* `codes/group_level/` folder contains the code to obtain group-level modularizations from subject connectivity matrices. As a starting point we have added dummy data in `data/multiple_subjects/`. The code to derive group-level modularizations (and other metrics) can be run by

```
python3 ICSC_group_level.py
```

* `codes/subject_level/` folder contains the code to obtain subject-level modularizations from multiples scans of the same subject. As a starting point we have added dummy data in `data/subject_sessions/subject_*/S*_corr.npy`. The code to derive subject-level modularizations (and other metrics) can be run by:

```
python3 ICSC_subject_level.py
```

