# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:00:46 2022

@author: user
"""

#import _init_paths
import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param

import torch.nn.functional as F
from .common import *

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class UNet_Nested(nn.Module):
    
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
    #     super(UNet, self).__init__()
    
    # def __init__(self, num_input_channels=3, num_output_channels=3, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.num_input_channels = num_input_channels
        self.feature_scale = feature_scale
        #self.is_deconv = is_deconv
        #self.is_batchnorm = is_batchnorm
        #self.is_ds = is_ds
        
        self.more_layers = more_layers
        self.concat_x = concat_x

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)

        self.conv10 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad)
        self.conv20 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad)
        self.conv30 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad)
        self.conv40 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad)
        
        # self.conv00 = unetConv2(self.num_input_channels, filters[0], self.is_batchnorm)
        # self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        # upsampling
        self.up_concat01 = unetUp(filters[0], upsample_mode, need_bias, pad)
        self.up_concat11 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up_concat21 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up_concat31 = unetUp(filters[3], upsample_mode, need_bias, pad)
        
        # self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        # self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        # self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        
        self.up_concat02 = unetUp(filters[0], upsample_mode, need_bias, pad, n_concat=3)
        self.up_concat12 = unetUp(filters[1], upsample_mode, need_bias, pad, n_concat=3)
        self.up_concat22 = unetUp(filters[2], upsample_mode, need_bias, pad, n_concat=3)

        # self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        # self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        # self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        
        self.up_concat03 = unetUp(filters[0], upsample_mode, need_bias, pad, n_concat=4)
        self.up_concat13 = unetUp(filters[1], upsample_mode, need_bias, pad, n_concat=4)

        # self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        # self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        
        self.up_concat04 = unetUp(filters[0], upsample_mode, need_bias, pad, n_concat=5)

        #self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        
        # final conv (without any concat)
        self.final_1 = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.final_2 = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.final_3 = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.final_4 = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        
        #self.final_1 = nn.Conv2d(filters[0], num_output_channels, 1)
        #self.final_2 = nn.Conv2d(filters[0], num_output_channels, 1)
        #self.final_3 = nn.Conv2d(filters[0], num_output_channels, 1)
        #self.final_4 = nn.Conv2d(filters[0], num_output_channels, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpool(X_00)    # 16*256*256
        X_10= self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1+final_2+final_3+final_4)/4

        return final

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        print(pad)
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False, n_concat=2):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2+(n_concat-2)*out_size, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size * 2+(n_concat-2)*out_size, out_size, None, need_bias, pad)
        else:
            assert False
# =============================================================================
#     def forward(self, inputs1, inputs2):
#         in1_up= self.up(inputs1)
#         
#         if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
#             diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
#             diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
#             inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
#         else:
#             inputs2_ = inputs2
# 
#         output= self.conv(torch.cat([in1_up, inputs2_], 1))
# 
#         return output
# =============================================================================
    
    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

      
