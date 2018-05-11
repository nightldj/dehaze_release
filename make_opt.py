#June 9, 2017, Zheng Xu, xuzhustc@gmail.com

#optimizer


# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torchvision as thv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim
import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from datetime import datetime
import shutil
import math




def get_optimizer_var(var, args, flag, lr): #for variables
    if flag == 'adam':
        optimizer = optim.Adam(var, lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay)
    elif flag == 'padam':
        optimizer = PredAdam(var, lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay, pred_gm=args.pred_gm)
    elif flag == 'rmsp':
        optimizer = optim.RMSprop(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif flag == 'sgd':
        optimizer = optim.SGD(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print 'unknown optimizer name, use SGD!'
        optimizer = optim.SGD(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer



def adjust_learning_rate(optimizer, init_lr, args, epoch, flag='linear'):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""
    if flag == 'segment':
        if epoch % args.lr_freq == 0:
            lr = init_lr * (0.1 ** (epoch // args.lr_freq))
            print 'epoch %d learning rate schedule to %f'%(epoch, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif flag == 'linear':
        s = epoch//args.lr_freq  #starting from lr_freq, learning rate drop linearly to 0.1 for each lr_freq
        p =  (s+1)*args.lr_freq - epoch
        elr = init_lr * (0.1**s)
        lr = elr + ( min(init_lr, elr*10) - elr)*float(p)/float(args.lr_freq)
        print 'epoch %d learning rate schedule to %f'%(epoch, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        print 'make_opt: unknown flag for adjusting learning rates'


