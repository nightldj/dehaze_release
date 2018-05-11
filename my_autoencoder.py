# my autoencoder for images
# Zheng Xu, xuzhustc@gmail.com, Jan 2018


#reference:
# WCT AE: https://github.com/sunshineatnoon/PytorchWCT/blob/master/modelsNIPS.py



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

from net_utils import *






class unet(nn.Module):
    def __init__(self, dropout = 0.3, trans_flag='pred', use_sgm='tanh', use_bn = 'b'):
        #trans_flag: pred/in/res
        super(unet, self).__init__()
        self.trans_flag = trans_flag
        self.dp = dropout  #dropout for texture part
        self.dep = 4 #up to conv4

        self.encs = [make_vgg_enc_layers(cfg[1])]  #conv1
        for i in range(2, self.dep+1):
            self.encs.append(make_vgg_aux_enc_layers(cfg[i-1], cfg[i])) #conv2~5
        self.encs = nn.ModuleList(self.encs)  #compatible with DataParallel

        self.decs = make_tr_dec_layers(dec_cfg[self.dep], cfg[self.dep][-1]*2, use_sgm=use_sgm, use_bn=use_bn)

        if trans_flag == 'pred':
            self.preds = make_pred_layers(dec_cfg[self.dep], cfg[self.dep][-1]*2)
        elif trans_flag == 'in':
            self.preds = make_in_layers(dec_cfg[self.dep], cfg[self.dep][-1]*2)
        elif trans_flag == 'bn':
            self.preds = make_bn_layers(dec_cfg[self.dep], cfg[self.dep][-1]*2)
        else:
            self.preds = None
            print 'no transform, simple unet'

        print 'unet stacks', len(self.encs), len(self.decs),  'of', self.dep


    def freeze_base(self):
        for enc in self.encs:
            for param in enc.parameters():
                param.requires_grad = False


    def load_from_torch(self, ptm, thm, th_cfg):
        print ptm, thm
        i = 0
        for layer in list(ptm):
            if isinstance(layer, nn.Conv2d):
                print i, '/', len(th_cfg), ':', th_cfg[i]
                layer.weight = th.nn.Parameter(thm.get(th_cfg[i]).weight.float())
                layer.bias = th.nn.Parameter(thm.get(th_cfg[i]).bias.float())
                i += 1
        print 'unet load torch #convs', len(th_cfg), i


    def load_aux_from_torch(self, ptm, thm, th_cfg, aux_cfg):
        #print ptm, thm
        assert(len(th_cfg) < len(aux_cfg))
        i = 0
        while i < len(th_cfg):
            assert(th_cfg[i] == aux_cfg[i])
            i += 1

        for layer in list(ptm):
            if isinstance(layer, nn.Conv2d):
                print i, '/', len(aux_cfg), ':', aux_cfg[i]
                layer.weight = th.nn.Parameter(thm.get(aux_cfg[i]).weight.float())
                layer.bias = th.nn.Parameter(thm.get(aux_cfg[i]).bias.float())
                i += 1
        print 'unet load aux torch #convs', len(th_cfg), '-', len(aux_cfg), i


    def load_model(self, enc_model = 'models/vgg_normalised_conv4_1.t7', dec_model = None):
        if True:
            print 'load encoder from:', enc_model
            vgg = load_lua(enc_model)
            self.load_from_torch(self.encs[0], vgg, th_cfg[1]) #conv1
            for i in range(2, self.dep+1):
                self.load_aux_from_torch(self.encs[i-1], vgg, th_cfg[i-1], th_cfg[i])
        else:
            print 'unet encoder: load: flag not supported', flag


    def get_base_perc(self, img):
        code = img
        bases = []
        grams = []
        for i in range(len(self.encs)):
            code = self.encs[i](code)
            bases.append(code)
            if (i+2) == self.dep: #conv3
                out = code
        return bases,out


    def forward(self, img):
        bases, perc = self.get_base_perc(img)
        bases = [x.detach() for x in bases]

        if self.trans_flag == 'pred':
            trans= pred_mv(bases, self.preds)
        elif self.trans_flag == 'in' or self.trans_flag=='bn':
            trans= pred_in(bases, self.preds)
        else:
            trans = bases

        x = bases[self.dep-1]
        for i in range(self.dep):
            x = th.cat([x, trans[self.dep-1-i]], dim=1)
            x = self.decs[i](x)
        out = x

        return out, perc.detach()

    def load_pred_model(self, load_model):
        checkpoint = th.load(load_model)
        if self.preds is not None:
            self.preds.load_state_dict(checkpoint['pred'])
        self.decs.load_state_dict(checkpoint['dec'])
        print 'unet: trained layers loaded from:', load_model

