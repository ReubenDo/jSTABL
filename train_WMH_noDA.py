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

    
# Define training and patches sampling parameters   
num_epochs_max = 10000  
patch_size = {'source':(144,192,48), 'target':(144,192,48)}

nb_voxels = {d:np.prod(v) for d,v in patch_size.items()}
queue_length = 16
samples_per_volume = 1
batch_size = 2

nb_classes = 8

# Training parameters
val_eval_criterion_alpha = 0.95
train_loss_MA_alpha = 0.95
nb_patience = 10
patience_lr = 5
weight_decay = 1e-5

MODALITIES_SOURCE = ['T1']
MODALITIES_TARGET = ['T1','FLAIR']

MODALITIES = {'source':MODALITIES_SOURCE, 'target':MODALITIES_TARGET}


def infinite_iterable(i):
    while True:
        yield from i

def train(paths_dict, model, transformation, device, save_path, opt):
    
    since = time.time()

    dataloaders = dict()
    # Define transforms for data normalization and augmentation
    for domain in ['source', 'target']:
        subjects_domain_train = ImagesDataset(
            paths_dict[domain]['training'], 
            transform=transformation['training'][domain])

        subjects_domain_val = ImagesDataset(
            paths_dict[domain]['validation'], 
            transform=transformation['validation'][domain])


        batch_loader_domain_train = infinite_iterable(DataLoader(subjects_domain_train, batch_size=batch_size))
        batch_loader_domain_val = infinite_iterable(DataLoader(subjects_domain_val, batch_size=batch_size))

        dataloaders_domain = dict()
        dataloaders_domain['training'] = batch_loader_domain_train
        dataloaders_domain['validation'] = batch_loader_domain_val
        dataloaders[domain] = dataloaders_domain


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
    optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=weight_decay, amsgrad=True)
    lr_s = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                           patience=patience_lr,
                                                           verbose=True, 
                                                           threshold=1e-3,
                                                           threshold_mode="abs")

  
    model = model.to(device)
    continue_training = True

    ind_batch_train = np.arange(0, samples_per_volume*len(paths_dict['source']['training']), batch_size)
    ind_batch_val = np.arange(0, samples_per_volume*max(len(paths_dict['source']['validation']),len(paths_dict['target']['validation'])), batch_size)
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
            running_loss_target = 0.0
            running_loss_source = 0.0
            running_loss_intermod = 0.0
            epoch_samples = 0

            # Iterate over data
            for _ in tqdm(ind_batch[phase]):
                batch_source = next(dataloaders['source'][phase])
                labels_source = batch_source['label'][DATA].to(device).type(torch.cuda.IntTensor)
                inputs_source = {mod:batch_source[mod][DATA].to(device) for mod in MODALITIES_SOURCE}

                batch_target= next(dataloaders['target'][phase])
                labels_target = batch_target['label'][DATA].to(device)
                inputs_target =  {'T1':batch_target['T1'][DATA].to(device), 'FLAIR': torch.cat([batch_target['T1'][DATA],batch_target['FLAIR'][DATA]],1).to(device)}


                inputs_target_assource = {mod:batch_target[mod][DATA].to(device) for mod in MODALITIES_SOURCE}


                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    
                    outputs_source, _ = model(inputs_source)
                    outputs_target, _ = model(inputs_target)

                    # Predictions using the shared modalities between the control and lesion data
                    outputs_target_assource, _ = model(inputs_target_assource)

                
                    loss_source = jaccard_tissue(outputs_source, labels_source)
                    loss_target = jaccard_lesion(outputs_target, 6.0+labels_target)
                    loss_intermod = jaccard_tissue(outputs_target, outputs_target_assource, is_prob=True)
                    #loss_intermod = loss_target

                    if epoch>opt.warmup and phase=='training':
                        loss = loss_source + loss_target + loss_intermod
                    else:
                        loss = loss_source + loss_target 


                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                epoch_samples += 1
                running_loss += loss.item()
                running_loss_source += loss_source.item()
                running_loss_target += loss_target.item()
                running_loss_intermod += loss_intermod.item()  

            epoch_loss = running_loss / epoch_samples
            epoch_loss_source = running_loss_source / epoch_samples
            epoch_loss_target = running_loss_target / epoch_samples
            epoch_loss_intermod = running_loss_intermod / epoch_samples

            
            print('{}  Loss Seg Source: {:.4f}'.format(
                phase, epoch_loss_source))
            print('{}  Loss Seg Target: {:.4f}'.format(
                phase, epoch_loss_target))
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
                    if  epoch-best_epoch>nb_patience:

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
    split_path_source = opt.dataset_split_source
    assert os.path.isfile(split_path_source), 'source file not found'

    split_path_target = opt.dataset_split_target
    assert os.path.isfile(split_path_target), 'target file not found'

    split_path = dict()
    split_path['source'] = split_path_source
    split_path['target'] = split_path_target

    path_file = dict()
    path_file['source'] = opt.path_source
    path_file['target'] = opt.path_target

    list_split = ['training', 'validation',]
    paths_dict = dict()

    for domain in ['source','target']:
        df_split = pd.read_csv(split_path[domain],header =None)
        list_file = dict()
        for split in list_split:
            list_file[split] = df_split[df_split[1].isin([split])][0].tolist()

        paths_dict_domain = {split:[] for split in list_split}
        for split in list_split:
            for subject in list_file[split]:
                subject_data = []
                for modality in MODALITIES[domain]:
                    subject_data.append(Image(modality, path_file[domain]+subject+modality+'.nii.gz', torchio.INTENSITY))
                if split in ['training', 'validation']:
                    subject_data.append(Image('label', path_file[domain]+subject+'Label.nii.gz', torchio.LABEL))

                    #subject_data[] = 
                paths_dict_domain[split].append(Subject(*subject_data))
            print(domain, split, len(paths_dict_domain[split]))
        paths_dict[domain] = paths_dict_domain
            


    # PREPROCESSING
    transform_training = dict()
    transform_validation = dict()
    for domain in ['source','target']:
        transform_training[domain] = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((144,192,48)),
            RandomAffine(scales=(0.9, 1.1), degrees=10),
            RandomNoise(std_range=(0, 0.10)),
            RandomFlip(axes=(0,)), 
            )  

        transform_training[domain] = Compose(transform_training[domain])   

        transform_validation[domain] = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((144,192,48)),
            )
        transform_validation[domain] = Compose(transform_validation[domain]) 

    
    transform = {'training': transform_training, 'validation':transform_validation}   
    
    # MODEL 
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}  
    dropout_op_kwargs = {'p': 0, 'inplace': True}   
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}   

    
    
    print("[INFO] Building model")  
    model= Generic_UNet(input_modalities=MODALITIES_TARGET,   
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
                        final_nonlin=torch.nn.Softmax(1))
  
    

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

    parser.add_argument('--dataset_split_target',
                        type=str,
                        default='dataset_split.csv')

    parser.add_argument('--dataset_split_source',
                        type=str,
                        default='dataset_split.csv')

    parser.add_argument('--path_source',
                        type=str,
                        default='../data/MEDIA/NeuroTissue_sk/')

    parser.add_argument('--path_target',
                        type=str,
                        default='../data/MEDIA/WMHLesion_sk/')

    parser.add_argument('--learning_rate',
                    type=float,
                    default=5*1e-4)

    parser.add_argument('--warmup',
                    type=int,
                    default=15)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()



