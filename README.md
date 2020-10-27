# Learning joint Segmentation of Tissues And Brain Lesions (jSTABL) from task-specific hetero-modal domain-shifted datasets

Public PyTorch implementation of the paper [Learning joint segmentation of tissues and brain lesions from task-specific hetero-modal domain-shifted datasets](https://arxiv.org/abs/2009.04009) published in Medical Image Analysis.

If you find this code useful for your research, please cite the following paper:
```
@article{DORENT2021101862,
title = "Learning joint segmentation of tissues and brain lesions from task-specific hetero-modal domain-shifted datasets",
journal = "Medical Image Analysis",
volume = "67",
pages = "101862",
year = "2021",
issn = "1361-8415",
doi = "https://doi.org/10.1016/j.media.2020.101862",
url = "http://www.sciencedirect.com/science/article/pii/S1361841520302267",
author = "Reuben Dorent and Thomas Booth and Wenqi Li and Carole H. Sudre and Sina Kafiabadi and Jorge Cardoso and Sebastien Ourselin and Tom Vercauteren",
keywords = "Joint learning, Domain adaptation, Multi-Task learning, Multi-Modal",
}
```

## Installation
### 1. Create a conda environment (recommended)
Within a conda environment:
```
ENVNAME="jstablenv"
conda create -n $ENVNAME python -y
conda activate $ENVNAME
```
### 2. Install PyTorch
Given your CUDA Toolkit version, please install [PyTorch](https://pytorch.org/) within the conda environment.

### 3. Install jSTABL
Within the conda environment:
```
pip install -e  git+https://github.com/ReubenDo/jSTABL#egg=jSTABL
```

### (Optionnal) 4. Install MRIPreprocessor
If your data is not pre-processed (skull-stripped and co-register) you could use the `MRIPreprocessor`:
Within the conda environment:
```
pip install git+https://github.com/ReubenDo/MRIPreprocessor#egg=MRIPreprocessor
```

## Usage
### Glioma

- If the T1, T1-c, T2 and FLAIR scans are pre-processed (skull-stripped and coregistered):
  
```
(jstablenv):~$jstabl_glioma  -t1 subj_T1.nii.gz -t1c subj_T1c.nii.gz -t2 subj_T2.nii.gz  -fl subj_FLAIR.nii.gz -res segmentation.nii.gz
INFO] GPU available.
[INFO] Reading data.
[INFO] Building model.
[INFO] Loading model.
[INFO] Starting Inference.
100%|███████████████████████████████████████████████████████| 27/27 [01:03<00:00,  2.35s/it]
[INFO] Inference done in 69s. Segmentation saved here: segmentation.nii.gz
Have a good day!
```
  
- If not, you can install `MRIPreprocessor` and perform the preprocessing as follows:
```
(jstablenv):~$jstabl_glioma  -t1 subj_T1.nii.gz -t1c subj_T1c.nii.gz -t2 subj_T2.nii.gz  -fl subj_FLAIR.nii.gz -res segmentation.nii.gz --preprocess
[INFO] GPU available.
[INFO] Reading data.
[INFO] Performing Coregistration
T1 is used as reference
Registration performed for T2
Registration performed for T1c
Registration performed for Flair
[INFO] Performing Skull Stripping using HD-BET
File: subj_T1.nii.gz 
preprocessing...
image shape after preprocessing:  (107, 160, 113)
prediction (CNN id)...
0
1
2
3
4
exporting segmentation...
[INFO] Building model.
[INFO] Loading model.
[INFO] Starting Inference.
100%|███████████████████████████████████████████████████████| 27/27 [01:03<00:00,  2.35s/it]
[INFO] Inference done in 180s. Segmentation saved here: segmentation.nii.gz
Have a good day!
```

### White Matter Lesions

- If the T1 and FLAIR scans are pre-processed (skull-stripped and coregistered):
  
```
(jstablenv):~$ jstabl_wmh  -t1 subj_T1.nii.gz -fl subj_FLAIR.nii.gz -res segmentation.nii.gz
[INFO] GPU available.
[INFO] Reading data.
[INFO] Building model.
[INFO] Loading model.
[INFO] Starting Inference.
[INFO] Inference done in 6s. Segmentation saved here: segmentation.nii.gz
Have a good day!
```
  
- If not, you can install `MRIPreprocessor` and perform the preprocessing+segmentation as follows:
```
(jstablenv):~$ jstabl_glioma  -t1 subj_T1.nii.gz -fl subj_FLAIR.nii.gz -res segmentation.nii.gz --preprocess
[INFO] GPU available.
[INFO] Reading data.
[INFO] Performing Coregistration
T1 is used as reference
Registration performed for Flair
[INFO] Performing Skull Stripping using HD-BET
File: subj_T1.nii.gz
preprocessing...
image shape after preprocessing:  (96, 153, 153)
prediction (CNN id)...
0
1
2
3
4
exporting segmentation...
[INFO] Building model.
[INFO] Loading model.
[INFO] Starting Inference.
[INFO] Inference done in 76s. Segmentation saved here: segmentation.nii.gz
Have a good day!
```
