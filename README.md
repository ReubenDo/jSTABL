# Scribble-based Domain Adaptation via Co-segmentation

Public pytorch implementation for our paper [Learning joint segmentation of tissues and brain lesions from task-specific hetero-modal domain-shifted datasets], (https://arxiv.org/abs/2009.04009)
which was invited in the MIDL 2019 Special Issue - Medical Image Analysis. 

If you find this code useful for your research, please cite the following paper:



## Virtual Environment Setup

The code is implemented in Python 3.6 using using the PyTorch library. 
Requirements:

 * Set up a virtual environment (e.g. conda or virtualenv) with Python 3.6
 * Install all requirements using:
  
  ````pip install -r requirements.txt````
  
## Data

## Training the Models

### White Matter Lesions
To train:

```` python3 train_BRATS_noDA.py --model_dir ./models/BRATS/noDA/ ````
### Glioma
Without Domain Adaptation:

```` python3 train_WMH_noDA.py --model_dir ./models/WMH/noDA/ ````

With Data Augmentation:

```` python3 train_WMH_noDA.py --model_dir ./models/BRATS/augm/ ````

With Adversarial Domain Adaptation:

```` python3 train_WMH_noDA.py --model_dir ./models/BRATS/adv/ ````


With Annotated Pseudo-Healthy Scans:

```` python3 train_BRATS_pseudohealthy.py --model_dir ./models/BRATS/pseudohealthy/ ````
