#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
import numpy as np
from tqdm import tqdm
import nibabel as nib   
import pandas as pd 
import SimpleITK as sitk


# pytorch 
import torch
import torch.nn 
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
from torchvision.transforms import Compose

# TorchIO
import torchio
from torchio import ImagesDataset, Image, Subject, DATA
from torchio.transforms import (    
        ZNormalization,
        CenterCropOrPad,   
        RandomNoise,    
        RandomFlip, 
        RandomAffine,   
        ToCanonical,
    )   

from network.UNetModalityMeanTogether import Generic_UNet
from utilities.jaccard import *    
from utilities.sampling import GridSampler, GridAggregator  
     

MODALITIES = ['T1','FLAIR']


def inference_padding(paths_dict, 
                      model,
                      transformation,
                      device, 
                      pred_path,
                      cp_path,
                      opt):
    print("[INFO] Loading model.")
    model.load_state_dict(torch.load(cp_path))
    model.to(device)
    model.eval()

    subjects_dataset_inf = ImagesDataset(paths_dict, transform=transformation)

    nb_subject = len(subjects_dataset_inf)

    border = (0,0,0)

    print("[INFO] Starting Inference.")
    print('Number of subjects to infer: {}'.format(nb_subject))

    for param in model.parameters():
        print(2,param.data.reshape(-1)[0])
        break

    for index, batch in enumerate(subjects_dataset_inf):
        name = batch['T1']['stem'].split('T1')[0]
        affine = batch['T1']['affine']
        reference = torchio.utils.nib_to_sitk(batch['T1'][DATA].numpy(), affine)

        batch_pad = CenterCropOrPad((144,192,48))(batch)

        affine_pad = batch_pad['T1']['affine']
        nb_modalities = len([k for k in batch_pad.keys() if k in opt.modalities])
        assert nb_modalities in [1,2], 'Incorrect number of modalities for {}'.format(name.split('T1')[0])

        data = {'T1':batch_pad['T1'][DATA].unsqueeze(0).to(device)}
        if nb_modalities==2:
            data['FLAIR'] = torch.cat([batch_pad['T1'][DATA],batch_pad['FLAIR'][DATA]],0).unsqueeze(0).to(device) 

        with torch.no_grad():
            logits, _ = model(data)
            labels = logits.argmax(dim=1, keepdim=True)
            labels = labels[0,0,...].cpu().numpy()
        # output = np.zeros(shape[2:])
        # output[:x_shape,:y_shape,:z_shape] = labels
        output = labels
        #output = nib.Nifti1Image(output.astype(float), affine_pad)
        output = torchio.utils.nib_to_sitk(output.astype(float), affine_pad)
        output = sitk.Resample(
            output,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
        )
        sitk.WriteImage(output, pred_path.format(name))
        print('{}/{} - Inference done for {} with {} modalities'.format(index, nb_subject, name, nb_modalities))


def main():
    opt = parsing_data()


    
    print("[INFO] Reading data.")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        

    
    # FOLDERS
    fold_dir = opt.model_dir
    checkpoint_path = os.path.join(fold_dir,'models', './CP_{}.pth')
    checkpoint_path = checkpoint_path.format(opt.epoch_infe)
    assert os.path.isfile(checkpoint_path), 'no checkpoint found'
    
    output_path = opt.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'output_{}.nii.gz')
    
    # SPLITS
    split_path = opt.dataset_split
    assert os.path.isfile(split_path), 'split file not found'
    print('Split file found: {}'.format(split_path))
    

    # Reading csv file
    df_split = pd.read_csv(split_path,header =None)
    list_file = dict()
    list_split = ['inference', 'validation']
    for split in list_split:
        list_file[split] = df_split[df_split[1].isin([split.lower()])][0].tolist()
 
    # filing paths
    paths_dict = {split:[] for split in list_split}
    for split in list_split:
        for subject in list_file[split]:
            subject_data = []
            for modality in MODALITIES:
                subject_modality = opt.path_file+subject+modality+'.nii.gz'
                if os.path.isfile(subject_modality):
                    subject_data.append(Image(modality, subject_modality, torchio.INTENSITY))
            if len(subject_data)>0:
                paths_dict[split].append(Subject(*subject_data))
    
        

    transform_inference = (
            ToCanonical(),
            ZNormalization(),      
        )   
    transform_inference = Compose(transform_inference) 
     
    # MODEL 
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}  
    dropout_op_kwargs = {'p': 0, 'inplace': True}   
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}   

    print("[INFO] Building model.")  
    model= Generic_UNet(input_modalities=['T1', 'FLAIR'],   
                        base_num_features=32,   
                        num_classes=opt.nb_classes,     
                        num_pool=4, 
                        num_conv_per_stage=2,   
                        feat_map_mul_on_downscale=2,    
                        conv_op=torch.nn.Conv3d,    
                        norm_op=torch.nn.InstanceNorm3d,    
                        norm_op_kwargs=norm_op_kwargs,  
                        nonlin=net_nonlin,  
                        nonlin_kwargs=net_nonlin_kwargs,      
                        convolutional_pooling=False,     
                        convolutional_upsampling=False,  
                        final_nonlin=lambda x: x)
  

    paths_inf = paths_dict['inference']+paths_dict['validation']
    inference_padding(paths_inf, model, transform_inference, device, output_path, checkpoint_path, opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='3D Segmentation Using PyTorch and NiftyNet')

    parser.add_argument('-epoch_infe',
                        type=str,
                        default = 'best')

    parser.add_argument('-model_dir',
                        type=str)

    parser.add_argument('-dataset_split',
                        type=str,
                        default='dataset_split.csv')

    parser.add_argument('-path_file',
                        type=str,
                        default='/raid/rdorent/T1_sk_crop/')


    parser.add_argument('-output_dir',
                        type=str)

    parser.add_argument('-nb_classes',
                        type=int,
                        default=8)

    parser.add_argument('-modalities',
                        default=MODALITIES)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()



