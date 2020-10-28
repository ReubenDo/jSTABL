# Installation
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
pip install -e  git+https://github.com/ReubenDo/jSTABL#egg=jSTABL
```
# Data
### Control data
Consists of 35 T1 scans from the OASIS project with annotations of 143 structures of the brain provided by [Neuromorphometrics, Inc.](http://Neuromorphometrics.com/) under academic subscription. From the 143 structures, we  deducted the 6 tissue  classes. Additionnaly, 25 T1 control scans from [ADNI-2](adni.loni.usc.edu) were added with bronze standard parcellation of the brain structures computed with the accurate but time-consuming [GIF algorithm](https://pubmed.ncbi.nlm.nih.gov/25879909/). The T1c, T2 and FLAIR scans are missing.

### Glioma data
Consists of 285 patients (210 with high grade glioma and 75 with low grade glioma) from the training set of [BraTS18](https://www.med.upenn.edu/sbia/brats2018/data.html). T1, T1c, T2 and FLAIR scans are provided for each patient. Three tumour structures are annotated. The tissue annotations are missing.

### White Matter Lesions data
Consists of 60 sets of brain MR imagesfrom the  [White  Matter Hyperintensities (WMH)](https://wmh.isi.uu.nl/) database. T1 and FLAIR scans are provided for each patient. The white matter lesions are annotated. The tissue annotations are missing.

# Training the models

## Glioma
Without Domain Adaptation:

```` python3 glioma/train_WMH_noDA.py --model_dir models/WMH/noDA/ ````

With Data Augmentation:

```` python3 glioma/train_BRATS_augmentation.py --model_dir models/BRATS/augm/ ````

With Adversarial Domain Adaptation:

```` python3 glioma/train_BRATS_adversarial.py --model_dir models/BRATS/adv/ ````

With Annotated Pseudo-Healthy Scans:

```` python3 glioma/train_BRATS_pseudohealthy.py --model_dir models/BRATS/pseudohealthy/ ````

## White Matter Lesions
To train:

```` python3 wmh/train_BRATS_noDA.py --model_dir models/BRATS/noDA/ ````

