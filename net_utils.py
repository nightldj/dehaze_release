# my autoencoder for images
# Zheng Xu, xuzhustc@gmail.com, Jan 2018


#reference:
# WCT AE: https://github.com/sunshineatnoon/PytorchWCT/blob/master/modelsNIPS.py
# WCT torch/TF: https://github.com/Yijunmaverick/UniversalStyleTransfer, https://github.com/eridgd/WCT-TF



# -*- coding: utf-8 -*-

import torch as th
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as func

import torch.backends.cudnn as cudnn
from torch.utils.serialization import load_lua

import numpy as np

import os
import time
from datetime import datetime
import shutil


cfg = {
        5: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512],#vgg19, block 5, 14 cnvs
        4: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512],#vgg19, block 4
        3: [64, 64, 'M', 128, 128, 'M', 256],#vgg19, block 3
        2: [64, 64, 'M', 128],#vgg19, block 2
        1: [64],#vgg19, block 1
        }
dec_cfg = {
        5: [512, 512, 'M', 512, 512, 512, 256, 'M',  256, 256, 256, 128, 'M', 128, 64, 'M', 64],
        4: [512, 256, 'M',  256, 256, 256, 128, 'M', 128, 64, 'M', 64],
        3: [256, 128, 'M', 128, 64, 'M', 64],
        2: [128, 64, 'M', 64],
        1: [64],
        }


th_cfg = {
        5:[0, 2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42],
        4:[0, 2, 5, 9, 12, 16, 19, 22, 25, 29],
        3:[0, 2, 5, 9, 12, 16],
        2:[0, 2, 5, 9],
        1:[0, 2],
        }
th_dec_cfg = {
        5:[1, 5, 8, 11, 14, 18, 21, 24, 27, 31, 34, 38, 41],
        4:[1, 5, 8, 11, 14, 18, 21, 25, 28],
        3:[1, 5, 8, 12, 15],
        2:[1, 5, 8],
        1:[1],
        }


def make_vgg_enc_layers(cfg):
    layers = [nn.Conv2d(3, 3, kernel_size=1, padding=0)]
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            layers += [nn.ReflectionPad2d((1,1,1,1)), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_vgg_aux_enc_layers(cfg, aux_cfg):
    assert(len(cfg) < len(aux_cfg))
    layers = []
    i = 0
    in_channels = None
    while i < len(cfg):
        assert(cfg[i] == aux_cfg[i])
        v = cfg[i]
        if v!= 'M':
            in_channels = v
        i += 1
    while i < len(aux_cfg):
        v = aux_cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            layers += [nn.ReflectionPad2d((1,1,1,1)), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        i +=1
    return nn.Sequential(*layers)



def make_tr_dec_layers(cfg, in_channels=0, use_bn='b', use_sgm='tanh'):   #trainable decoder
    assert(in_channels == cfg[0]*2)
    decs = []
    layers = [ nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=0),
            nn.ReLU(True)]  #first layer without BN
    in_channels = cfg[0]
    i = 1
    first=True
    use_bias = False
    while i < len(cfg):
        v = cfg[i]
        if use_bn == 'in':
            layers += [nn.InstanceNorm2d(in_channels, affine=True)]
        elif use_bn == 'b':
            layers += [nn.BatchNorm2d(in_channels)]
        else:
            use_bias=True
            print 'make_tr_dec: no norm', use_bn
        if v == 'M':
            i += 1
            v = cfg[i]
            if first:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, stride=2, padding=0, bias=use_bias)
                first = False
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1, bias=use_bias)
            layers += [conv2d, nn.ReLU(True)]
            decs.append(nn.Sequential(*layers))
            layers = []
            in_channels = 2*v
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0, bias=(not use_bn))
            layers += [nn.ReflectionPad2d((1,1,1,1)), conv2d, nn.ReLU(True)]
            in_channels = v
        i += 1

    layers += [nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)]  #last layer, create image
    if use_sgm == 'sigmoid':  #constrained the pixel value to be 0~1
        layers += [nn.Sigmoid()]
    elif use_sgm == 'tanh':
        layers += [nn.Tanh()]
    elif use_sgm == 'hard':
        layers += [nn.Hardtanh(min_val=0)]
    elif use_sgm.lower() != 'none':
        print 'unknow last decoder layer flag:', use_sgm
    decs.append(nn.Sequential(*layers))
    return nn.ModuleList(decs)


def make_pred_layers(cfg, in_channels=0):   #
    assert(in_channels == cfg[0]*2)
    decs = [nn.Linear(in_channels, in_channels)]
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == 'M':
            i += 1
            v = cfg[i]
            in_channels = 2*v
            decs.append(nn.Linear(in_channels, in_channels))
        i += 1
    return nn.ModuleList(list(reversed(decs)))


def make_in_layers(cfg, in_channels=0):   #
    assert(in_channels == cfg[0]*2)
    decs = [nn.InstanceNorm2d(cfg[0], affine=True)]
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == 'M':
            i += 1
            v = cfg[i]
            in_channels = 2*v
            decs.append(nn.InstanceNorm2d(v, affine=True))
        i += 1
    return nn.ModuleList(list(reversed(decs)))


def make_bn_layers(cfg, in_channels=0):   #
    assert(in_channels == cfg[0]*2)
    decs = [nn.BatchNorm2d(cfg[0])]
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == 'M':
            i += 1
            v = cfg[i]
            in_channels = 2*v
            decs.append(nn.BatchNorm2d(v))
        i += 1
    return nn.ModuleList(list(reversed(decs)))



def pred_mv(bases, preders):  #get mean variance of each filter
    assert(len(preders)==len(bases))
    outs = []
    for i in xrange(len(bases)):  #for each layer
        #whitening
        base = bases[i]
        bn,cn,wn,hn=base.size()
        bv = base.view(bn, cn, wn*hn)  #vectorize feature map
        mu = th.mean(bv, dim=2, keepdim=True) #get mean
        ss = th.std(bv, dim=2, keepdim=True)
        b = (bv - mu)/th.clamp(ss, min=1e-6)  #normalize

        #pred
        muss = th.cat([mu,ss], dim=1)
        muss2 = preders[i](muss.view(muss.size(0), -1))
        mu2,ss2 = muss2.unsqueeze(2).chunk(2, dim=1) #get mean
        #print muss.size(), muss2.size(), mu2.size(), ss2.size()
        #raw_input('debug pred_mv')

        bvst = b*ss2 + mu2
        outs.append(bvst.view(bn,cn,wn,hn))

        #print mu.size(), ss.size()
        #print bvst.size(), bv.size()
        #print bv[0, 0, 0:10], b[0, 0, 0:10], bvst[0, 0, 0:10], bv2[0, 0, 0:10]
        #raw_input('debug adin')

    #print len(outs),len(bases)
    #for i in xrange(len(bases)):
    #    print outs[i].size(), bases[i].size()
    #raw_input('debug adin whiten')
    return outs


def pred_in(bases, preders):  #get mean variance of each filter
    assert(len(preders)==len(bases))
    outs = []
    for i in xrange(len(bases)):  #for each layer
        outs.append(preders[i](bases[i]))

    return outs

