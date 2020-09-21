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

# pytorch 
import torch
import torch.nn 
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
from torchvision.transforms import Compose

# TorchIO
import torchio
from torchio import ImagesDataset, Image, Subject, DATA, Queue
from torchio.data.sampler import ImageSampler
from torchio.transforms import (    
        ZNormalization,
        CenterCropOrPad,   
        RandomNoise,    
        RandomFlip, 
        RandomAffine,   
        ToCanonical,
    )   

from jstabl.networks.UNetModalityMeanTogether import Generic_UNet
from jstabl.utilities.jaccard import *

  
# Define training and patches sampling parameters
patch_size = (112,112,112)
queue_length = 8
samples_per_volume = 1


# Training parameters
val_eval_criterion_alpha = 0.95
train_loss_MA_alpha = 0.95
nb_patience = 10
patience_lr = 5
weight_decay = 1e-5

nb_classes = 10

MODALITIES_CONTROL = ['T1']
MODALITIES_LESION = ['T1','T1c','T2','Flair']
DATASETS = ['control', 'lesion']
MODALITIES = {'control':MODALITIES_CONTROL, 'lesion':MODALITIES_LESION}

batch_size = {'control':2, 'lesion':2}



def infinite_iterable(i):
    while True:
        yield from i

def train(paths_dict, model, transformation, device, save_path, opt):
    
    since = time.time()

    dataloaders = dict()
    # Define transforms for data normalization and augmentation
    for dataset in DATASETS:
        subjects_dataset_train = ImagesDataset(
            paths_dict[dataset]['training'], 
            transform=transformation['training'][dataset])

        subjects_dataset_val = ImagesDataset(
            paths_dict[dataset]['validation'], 
            transform=transformation['validation'][dataset])

        # Number of qoekwea
        workers = 10

        # Define the dataset data as a queue of patches
        queue_dataset_train = Queue(
            subjects_dataset_train,
            queue_length,
            samples_per_volume,
            patch_size,
            ImageSampler,
            num_workers=workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        queue_dataset_val = Queue(
            subjects_dataset_val,
            queue_length,
            samples_per_volume,
            patch_size,
            ImageSampler,
            num_workers=workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        
        batch_loader_dataset_train = infinite_iterable(DataLoader(queue_dataset_train, batch_size=batch_size[dataset]))
        batch_loader_dataset_val = infinite_iterable(DataLoader(queue_dataset_val, batch_size=batch_size[dataset]))

        dataloaders_dataset = dict()
        dataloaders_dataset['training'] = batch_loader_dataset_train
        dataloaders_dataset['validation'] = batch_loader_dataset_val
        dataloaders[dataset] = dataloaders_dataset

    df_path = os.path.join(opt.model_dir,'log.csv')
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
        epoch = df.iloc[-1]['epoch']
        best_epoch = df.iloc[-1]['best_epoch']

        val_eval_criterion_MA = df.iloc[-1]['MA']
        best_val_eval_criterion_MA = df.iloc[-1]['best_MA']

        initial_lr = df.iloc[-1]['lr']

        model.load_state_dict(torch.load(save_path.format('best')))

    else:
        df = pd.DataFrame(columns=['epoch','best_epoch', 'MA', 'best_MA', 'lr'])
        val_eval_criterion_MA = None
        best_epoch = 0
        epoch = 0

        initial_lr = opt.learning_rate

    # Optimisation policy
    optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate, weight_decay=weight_decay, amsgrad=True)
    lr_s = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                           patience=patience_lr,
                                                           verbose=True, 
                                                           threshold=1e-3,
                                                           threshold_mode="abs")

    
    model = model.to(device)

    continue_training = True
    
    
    ind_batch_train = np.arange(0, samples_per_volume*len(paths_dict['lesion']['training']), batch_size['lesion'])
    ind_batch_val = np.arange(0, samples_per_volume*len(paths_dict['lesion']['validation']),  batch_size['lesion'])
    ind_batch= dict()
    ind_batch['training'] = ind_batch_train
    ind_batch['validation'] = ind_batch_val

    while continue_training:
        epoch+=1
        print('-' * 10)
        print('Epoch {}/'.format(epoch))
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
            
        # Each epoch has a training and validation phase
        for phase in ['training','validation']:
        #for phase in ['validation','training']:
            print(phase)
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_lesion = 0.0
            running_loss_control = 0.0
            running_loss_intermod = 0.0

            epoch_samples = 0

            # Iterate over data
            for _ in tqdm(ind_batch[phase]):
                # Control data (T1)
                batch_control = next(dataloaders['control'][phase])
                labels_control = batch_control['label'][DATA].to(device).type(torch.cuda.IntTensor)
                inputs_control = {mod:batch_control[mod][DATA].to(device) for mod in MODALITIES_CONTROL}

                # Lesion data (T1 + full set of modalities) 
                batch_lesion = next(dataloaders['lesion'][phase])
                labels_lesion = batch_lesion['label'][DATA].to(device).type(torch.cuda.IntTensor)
                inputs_lesion =  {'T1':batch_lesion['T1'][DATA].to(device), 'all': torch.cat([batch_lesion[mod][DATA] for mod in MODALITIES_LESION],1).to(device)}
                inputs_lesion_T1 = {mod:batch_lesion[mod][DATA].to(device) for mod in MODALITIES_CONTROL}

                # Batch sizes (useful when the batch is not full at the end of the epoch)
                bs_control = inputs_control['T1'].shape[0]
                bs_lesion = inputs_lesion['T1'].shape[0]


                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    # Concatenating the 2 predictions made on the T1 for a faster computation
                    input_T1 = {'T1':torch.cat([inputs_control['T1'],inputs_lesion_T1['T1']],0)}
                    outputs_T1, _ = model(input_T1)

                    # Deconcatenating the 2 predictions
                    outputs_control = outputs_T1[:bs_control,...]
                    outputs_lesion_T1 = outputs_T1[bs_control:bs_control+bs_lesion,...]

                    # Prediction using the full set of modalities (lesion data)
                    outputs_lesion, _ = model(inputs_lesion)

                    # Loss computation using the Jaccard distance
                    loss_control = jaccard_tissue(outputs_control, labels_control)
                    loss_lesion = jaccard_lesion(outputs_lesion, 6.0+labels_lesion)
                    loss_intermod = jaccard_tissue(outputs_lesion, outputs_lesion_T1, is_prob=True)


                    if epoch>opt.warmup and phase=='training':
                        loss = loss_control  + loss_lesion + loss_intermod  
                    else:
                        loss = loss_control + loss_lesion                      
                    
                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                epoch_samples += 1
                running_loss += loss.item()
                running_loss_control += loss_control.item()
                running_loss_lesion += loss_lesion.item()
                running_loss_intermod += loss_intermod.item()
   

            epoch_loss = running_loss / epoch_samples
            epoch_loss_control = running_loss_control / epoch_samples
            epoch_loss_lesion = running_loss_lesion / epoch_samples
            epoch_loss_intermod = running_loss_intermod / epoch_samples

            
            print('{}  Loss Seg Control: {:.4f}'.format(
                phase, epoch_loss_control))
            print('{}  Loss Seg Lesion: {:.4f}'.format(
                phase, epoch_loss_lesion))
            print('{}  Loss Seg Intermod: {:.4f}'.format(
                phase, epoch_loss_intermod))                    
                     

            if phase == 'validation':
                if val_eval_criterion_MA is None: #first iteration
                    val_eval_criterion_MA = epoch_loss
                    best_val_eval_criterion_MA = val_eval_criterion_MA
                    print(val_eval_criterion_MA)

                else: #update criterion
                    val_eval_criterion_MA = val_eval_criterion_alpha * val_eval_criterion_MA + (
                                1 - val_eval_criterion_alpha) * epoch_loss
                    print(val_eval_criterion_MA)

                lr_s.step(val_eval_criterion_MA)
                df = df.append({'epoch':epoch,
                                'best_epoch':best_epoch,
                                'MA':val_eval_criterion_MA,
                                'best_MA':best_val_eval_criterion_MA,  
                                'lr':param_group['lr']}, ignore_index=True)
                df.to_csv(df_path)

                if val_eval_criterion_MA < best_val_eval_criterion_MA:
                    best_val_eval_criterion_MA = val_eval_criterion_MA
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format('best'))

                else:
                    if epoch-best_epoch>nb_patience:
                        continue_training=False

                if epoch==opt.warmup:
                    torch.save(model.state_dict(), save_path.format(opt.warmup))

    
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best epoch is {}'.format(best_epoch))


def main():
    opt = parsing_data()


    
    print("[INFO]Reading data")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        
    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model = os.path.join(fold_dir,'models')
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    save_path = os.path.join(fold_dir_model,'./CP_{}.pth')

    output_path = os.path.join(fold_dir,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'output_{}.nii.gz')
    
    
    # LOGGING
    orig_stdout = sys.stdout
    if os.path.exists(os.path.join(fold_dir,'out.txt')):
        compt = 0
        while os.path.exists(os.path.join(fold_dir,'out_'+str(compt)+'.txt')):
            compt+=1
        f = open(os.path.join(fold_dir,'out_'+str(compt)+'.txt'), 'w')
    else:
        f = open(os.path.join(fold_dir,'out.txt'), 'w')
    sys.stdout = f


    # SPLITS
    split_path = dict()
    split_path['control'] = opt.split_control
    split_path['lesion'] = opt.split_lesion
    for dataset in DATASETS:
        assert os.path.isfile(split_path[dataset]), f'{dataset}: split not found'

    path_file = dict()
    path_file['control'] = opt.path_control
    path_file['lesion'] = opt.path_lesion

    splits = ['training', 'validation']
    paths_dict = dict()

    for dataset in DATASETS:
        df_split = pd.read_csv(split_path[dataset],header =None)
        list_file = dict()
        for split in splits:
            list_file[split] = df_split[df_split[1].isin([split])][0].tolist()
        
        paths_dict_dataset = {split:[] for split in splits}
        for split in splits:
            for subject in list_file[split]:
                subject_data = []
                for modality in MODALITIES[dataset]:
                    subject_data.append(Image(modality, path_file[dataset]+subject+modality+'.nii.gz', torchio.INTENSITY))
                if split in ['training', 'validation']:
                    subject_data.append(Image('label', path_file[dataset]+subject+'Label.nii.gz', torchio.LABEL))
                paths_dict_dataset[split].append(Subject(*subject_data))
            print(f'{dataset} / {split}: {len(paths_dict_dataset[split])} subjects')
        paths_dict[dataset] = paths_dict_dataset
            

    # PREPROCESSING
    transform_training = dict()
    transform_validation = dict()
    for dataset in DATASETS:
        transform_training[dataset] = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((144,192,144)),
            RandomAffine(scales=(0.9, 1.1), degrees=10),
            RandomNoise(std_range=(0, 0.10)),
            RandomFlip(axes=(0,)), 
            )   
        transform_training[dataset] = Compose(transform_training[dataset])   

        transform_validation[dataset] = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((144,192,144)),
            )
        transform_validation[dataset] = Compose(transform_validation[dataset]) 


    transform = {'training': transform_training, 'validation':transform_validation}   
    
    # MODEL 
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}  
    dropout_op_kwargs = {'p': 0, 'inplace': True}   
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}   

    print("[INFO] Building model")  
    model= Generic_UNet(input_modalities=['T1', 'all'],   
                        base_num_features=32,   
                        num_classes=nb_classes,     
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
                        final_nonlin=torch.nn.Softmax(1),
                        input_features={'T1':1, 'all':4})
  

    print("[INFO] Training")
    train(paths_dict, 
        model, 
        transform,
        device, 
        save_path,
        opt)


    sys.stdout = orig_stdout
    f.close()


def parsing_data():
    parser = argparse.ArgumentParser(
        description='Brain Tissue and Lesion segmentation')

    parser.add_argument('--model_dir',
                        type=str)

    parser.add_argument('--split_control',
                        type=str,
                        default='./splits/Neuro/1.csv')

    parser.add_argument('--split_lesion',
                        type=str,
                        default='./splits/BRATS/1.csv')

    parser.add_argument('--path_control',
                        type=str,
                        default='../data/T1_sk_crop/')

    parser.add_argument('--path_lesion',
                        type=str,
                        default='../data/BRATS2018_crop_renamed/')

    parser.add_argument('-learning_rate',
                    type=float,
                    default=5*1e-4)

    parser.add_argument('-warmup',
                    type=int,
                    default=30)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()



