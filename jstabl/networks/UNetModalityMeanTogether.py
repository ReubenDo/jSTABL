#!/usr/bin/env python3
#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvNormNonlin, self).__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op


        self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([ConvNormNonlin(input_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs_first_conv,
                                     self.norm_op, self.norm_op_kwargs,
                                     self.nonlin, self.nonlin_kwargs)] +
              [ConvNormNonlin(output_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs,
                                     self.norm_op, self.norm_op_kwargs, 
                                     self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)



class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Generic_UNet(nn.Module):

    def __init__(self, input_modalities, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, 
                 final_nonlin=torch.nn.Softmax(1), weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, input_features=None):


        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits


        self.conv_kwargs = {'stride':1, 'dilation':1, 'bias':True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin

        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d

        conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        pool_op_kernel_sizes = [(2, 2, 2)] * num_pool


        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])


        self.conv_first_block = dict()
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        if input_features is None:
            input_features = {'T1':1, 'FLAIR':2}

        output_features = base_num_features

        # FIRST BLOCK

        for modality in input_modalities:
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[0]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[0]
            # add convolutions
            self.conv_first_block[modality] = StackedConvLayers(input_features[modality], output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs
                                                              )
        
        self.conv_blocks_context.append(self.conv_first_block[modality]) #small hack
        self.td.append(pool_op(pool_op_kernel_sizes[0]))
        input_features = output_features
        output_features = int(np.round(output_features * feat_map_mul_on_downscale))



        for d in range(1,num_pool):
            # determine the first stride

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs
                                                              ))

            self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))


        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs,  self.nonlin, self.nonlin_kwargs),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs)))




        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u+1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u+1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                   self.nonlin, self.nonlin_kwargs),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs)
            ))

        
        self.final_conv = conv_op(self.conv_blocks_localization[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)


        # register all modules properly
        self.conv_first_block = nn.ModuleDict(self.conv_first_block)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            #self.apply(print_module_training_status)

    def forward(self, x, skip_only=False):
        first_block_outputs = []
        skips = []
        seg_outputs = []

        for modality in x.keys():
            first_block_outputs.append(self.conv_first_block[modality](x[modality]))

        x = sum(first_block_outputs)/len(x.keys())

        skips.append(x)
        x = self.td[0](x)

        
        for d in range(1,len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        if not skip_only:
            x = self.conv_blocks_context[-1](x)
            #print(x.shape)

            for u in range(len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)

                #print(x.shape)

            features = x
            seg_outputs = self.final_nonlin(self.final_conv(x))

            return seg_outputs, skips
        else:
            return skips
                




