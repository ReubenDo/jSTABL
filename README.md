# Learning joint Segmentation of Tissues And Brain Lesions (jSTABL) from task-specific hetero-modal domain-shifted datasets

Public PyTorch implementation of [Learning joint segmentation of tissues and brain lesions from task-specific hetero-modal domain-shifted datasets](https://arxiv.org/abs/2009.04009) published in Medical Image Analysis.

This work proposed a technique to perform joint brain tissue and lesion segmentation. Due to the lack of fully-annotated data, the framework has been trained  using hetero-modal (missing modalities), task-specific (tissue or lesion annotations) and domain-shifted (different acquisition protocols) datasets.

Two types of lesions are considered: gliomas and white matter lesions.

### Example of joint brain tissues and glioma segmentation:
![glioma_example](https://github.com/ReubenDo/reubendo.github.io/blob/master/images/together_optimised_loop.gif)

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
### 1. Create a [conda](https://docs.conda.io/en/latest/) environment (recommended)
```
ENVNAME="jstablenv"
conda create -n $ENVNAME python -y
conda activate $ENVNAME
```
### 2. Install [PyTorch](https://pytorch.org/)
Please install [PyTorch](https://pytorch.org/) for your CUDA toolkit within the conda environment:

### 3. Install jSTABL
Within the conda environment:
```
(jstablenv):~ pip install -e  git+https://github.com/ReubenDo/jSTABL#egg=jSTABL
```

### (Optionnal) 4. Install [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor)
If your data isn't preprocessed (skull-stripped and co-registered) you could use [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor). 

To install it within the conda environment:
```
(jstablenv):~ pip install git+https://github.com/ReubenDo/MRIPreprocessor#egg=MRIPreprocessor
```

## Usage
Using jSTABL is straightforward in any terminal on your linux system. The following examples show how to perform: 1. joint brain tissues and glioma segmentation; 2. joint brain tissues and white matter lesion segmentation.

The framework has been trained on preprocessed data, i.e. on skull-stripped and coregistered scans.
Additionnally, the data has been cropped to avoid performing inference on patchs containing only zeros.

If the data is not already preprocessed, an external library [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor) can be directly employed in the framework to perform this preprocessing. Note that other options could be considered such as the [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit).

### 1. Gliomas

- If the T1, T1c, T2 and FLAIR scans are already preprocessed (skull-stripped, coregistered and cropped):
  
```
(jstablenv):~$jstabl_glioma  -t1 subj_T1.nii.gz -t1c subj_T1c.nii.gz  \
-t2 subj_T2.nii.gz  -fl subj_FLAIR.nii.gz -res segmentation.nii.gz
INFO] GPU available.
[INFO] Reading data.
[INFO] Building model.
[INFO] Loading model.
[INFO] Starting Inference.
100%|███████████████████████████████████████████████████████| 27/27 [01:03<00:00,  2.35s/it]
[INFO] Inference done in 69s. Segmentation saved here: segmentation.nii.gz
Have a good day!
```
  
- If the data isn't preprocessed, you can install [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor) and perform the preprocessing+segmentation as follows:
```
(jstablenv):~$jstabl_glioma  -t1 subj_T1.nii.gz -t1c subj_T1c.nii.gz \
 -t2 subj_T2.nii.gz  -fl subj_FLAIR.nii.gz -res segmentation.nii.gz --preprocess
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

### 2. White Matter Lesions

- If the T1 and FLAIR scans are preprocessed (skull-stripped, coregistered and cropped):
  
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
  
- If the data isn't preprocessed, you can install [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor) and perform the preprocessing+segmentation as follows:
```
(jstablenv):~$ jstabl_glioma  -t1 subj_T1.nii.gz -fl subj_FLAIR.nii.gz \
-res segmentation.nii.gz --preprocess
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
# Copyright
* Copyright (c) 2020, King's College London. All rights reserved.
