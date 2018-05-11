# VGG-based U-net with Instance normalizatin for single image dehazing
# Zheng Xu, xuzhustc@gmail.com, Apr 2018
# Reference paper:  The Effectiveness of Instance Normalization: a Strong Baseline for Single Image Dehazing (https://arxiv.org/abs/1805.03305)
# Please kindly consider cite our paper if you find the code is helpful for your research.


# We acknowledge the following open-source repo:
# https://github.com/sunshineatnoon/PytorchWCT


#usage:
#training
#python main.py --trans-flag in --use-bn in --batch-size 16 --test-batch-size 8 --optm sgd --lr 0.1 --lr-freq 30 --epochs 60 --rec-w 1 --per-w 1  --print-freq 200 --gpuid 0,1,2,3
#testing
#python main.py --trans-flag in --use-bn in  --test-flag --test-batch-size 8 --gpuid 0 --load-model models/dehaze_release.pth --save-image output


# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import torch as th
from torch.autograd import Variable

import torchvision as thv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim
import torch.backends.cudnn as cudnn


import folder
from my_autoencoder import *
from net_utils import *
import make_opt as mko


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

import argparse
import os
import time
from datetime import datetime
import shutil



parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--tr-haze-data', default='data/RESIDE_standard/ITS/hazy', help='train hazy image')
parser.add_argument('--tr-gt-data', default='data/RESIDE_standard/ITS/clear', help='clean images')
parser.add_argument('--tedata-flag', default='reside', type=str, help='the testing data: reside')
parser.add_argument('--te-haze-data', default='data/RESIDE_standard/SOTS/indoor/hazy', help='test hazy image')
parser.add_argument('--te-gt-data', default='data/RESIDE_standard/SOTS/indoor/gt', help='clean images')
parser.add_argument('--num-workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1 encoder')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--test-batch-size', default=16, type=int, help='test minibatch size')
parser.add_argument('--gpuid', type=str, default='0', help="which gpu to run on.  default is 0")
parser.add_argument('--save-image',default='output', help='path for saving the dehazed image')
parser.add_argument('--print-freq', default=100, type=int, help='print every x minibatches')
parser.add_argument('--test-flag', action='store_true', help='testing')
parser.add_argument('--save-model', default='models/', help='folder to save model')
parser.add_argument('--load-model', default=None, type=str, help='load model for fine tuning or testing')

parser.add_argument('--trans-flag', default='pred', type=str, help='the transform method: pred | in |none')
parser.add_argument('--rec-w', default=1, type=float, help='weight for recosntruction loss')
parser.add_argument('--per-w', default=1, type=float, help='weight for perceptron loss')

parser.add_argument('--seed', default=2017, type=int, help='random seed')
parser.add_argument('--optm', default='sgd', help='optimizer: sgd | adam | rmsp')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr-freq', default=30, type=int, help='learning rate scheduler, 0.1 every x epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay ')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs')
parser.add_argument('--use-bn', default='in', type=str, help='batch norm/instance norm/none for dec: b | in | none')

args = parser.parse_args()

print datetime.now(), args, '\n============================'


#############################################

tag='dehze_%s_%d_%s_%s_r%.3f_p%.3f_%s%.4f_wd%.1f_ep%d_%d_mb%d_%d'%(
        datetime.now().strftime("%m%d%H"), args.seed, args.trans_flag,
        args.use_bn, args.rec_w, args.per_w,
        args.optm, args.lr, args.weight_decay,  args.epochs, args.lr_freq, args.batch_size, args.test_batch_size)


def get_save_file(args):
    best_file1 = '%s/%s'%(args.save_model, tag)
    return best_file1

if not os.path.exists(args.save_model):
    os.makedirs(args.save_model)
    #os.chmod(args.save_model, 0o777)

def save_model(epoch, wrap_net,  optimizer, save_file):
    if len(gids) > 1:
        if wrap_net.module.net.preds is not None:
            th.save({'epoch':epoch,
                'pred':wrap_net.module.net.preds.state_dict(),
                'dec':wrap_net.module.net.decs.state_dict(),
                'optimizer':optimizer.state_dict(),
                }, save_file)
        else:
            th.save({'epoch':epoch,
                'dec':wrap_net.module.net.decs.state_dict(),
                'optimizer':optimizer.state_dict(),
                }, save_file)
    else:
        if wrap_net.net.preds is not None:
            th.save({'epoch':epoch,
                'pred':wrap_net.net.preds.state_dict(),
                'dec':wrap_net.net.decs.state_dict(),
                'optimizer':optimizer.state_dict(),
                }, save_file)
        else:
            th.save({'epoch':epoch,
                'dec':wrap_net.net.decs.state_dict(),
                'optimizer':optimizer.state_dict(),
                }, save_file)
    os.chmod(save_file, 0o777)



use_cuda = th.cuda.is_available()
dtype = th.cuda.FloatTensor if use_cuda else th.FloatTensor


gids = args.gpuid.split(',')
gids = [int(x) for x in gids]
print 'deploy on GPUs:', gids

th.manual_seed(args.seed)
if use_cuda:
    if len(gids) == 1:
        th.cuda.set_device(gids[0])
    th.cuda.manual_seed(args.seed)



################# model
class unet_withloss(nn.Module):
    def __init__(self, args):
        super(unet_withloss, self).__init__()

        net = unet(trans_flag = args.trans_flag, use_bn=args.use_bn)
        net.load_model(enc_model=args.vgg4)

        #load pre-trained
        if args.load_model is not None and args.load_model != 'none' and args.load_model != 'None':
            net.load_pred_model(args.load_model)

        print '================== net \n', net

        #loss and optim
        criterion = nn.MSELoss()

        net.freeze_base()

        self.net = net
        self.criterion = criterion

    def forward(self, img, gt, per_w):
        x,p = self.net(img)
        l1 = self.criterion(x,gt)
        if per_w > 0:
            _,p2 = self.net(x)
            l2 = self.criterion(p2,p)
        else:
            l2 = Variable(l1.data.clone().zero_())

        return (l1,l2), x


wrap_net = unet_withloss(args)
if args.load_model is not None:
    wrap_net.net.load_pred_model(args.load_model)
if args.trans_flag == 'pred' or args.trans_flag == 'in':
    optimizer = mko.get_optimizer_var([{'params':wrap_net.net.preds.parameters(),
        'params':wrap_net.net.decs.parameters()}],
        args, args.optm, args.lr)
else:
    optimizer = mko.get_optimizer_var([{'params':wrap_net.net.decs.parameters()}], args, args.optm, args.lr)
print optimizer

if use_cuda:
    if len(gids) > 1:
        wrap_net = nn.DataParallel(wrap_net, device_ids=gids)
    wrap_net.cuda() #use GPU
    cudnn.benchmark = True

# ============================ load data
def load_reside():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    tr_set = folder.HazeImageFolder(hazy_path=args.tr_haze_data, gt_path=args.tr_gt_data, transform=transform_test)
    tr_loader = th.utils.data.DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    if args.tedata_flag == 'reside':
        te_set = folder.HazeImageFolder(hazy_path=args.te_haze_data, gt_path=args.te_gt_data, transform=transform_test)
    else:
        print 'other test data not supported in this release'
    te_loader = th.utils.data.DataLoader(te_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    print 'imgs #:', len(tr_set.imgs), len(te_set.imgs)
    print tr_set.imgs[0], te_set.imgs[0]
    #raw_input('debug load reside')
    return tr_loader,te_loader

tr_loader,te_loader = load_reside()
print datetime.now(), 'data loaded!\n'


def save_himgs(save_folder, bi, hinputs, ginputs, target):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    def imsave(tensor, savefile):
        image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
        image = unloader(image)
        image.save(savefile)
        os.chmod(savefile, 0o777)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.chmod(save_folder, 0o777)
    for j in range(hinputs.size(0)):
        tmp = hinputs.data[j].clamp_(0, 1)
        imsave(tmp, '%s/h%d_%d_%s.jpg'%(save_folder, bi, j, 'hazy'))
        tmp = ginputs.data[j].clamp_(0, 1)
        imsave(tmp, '%s/h%d_%d_%s.jpg'%(save_folder, bi, j, 'gt'))
        tmp = target.data[j].clamp_(0, 1)
        imsave(tmp, '%s/h%d_%d_%s.jpg'%(save_folder, bi, j, 'cl'))



####################################################

epoch = 0
best_mse = 1.0e10
best_epoch = 0

def train(epoch):
    wrap_net.train()
    running_loss = 0.0
    running_time = 0.0
    loading_time = 0.0
    end = time.time()
    for bi,(hinputs, ginputs) in enumerate(tr_loader):
        #print 'mb', bi
        if use_cuda:
            hinputs, ginputs = hinputs.cuda(async=True), ginputs.cuda(async=True)
        #hinputs,ginputs = Variable(hinputs,volatile=True), Variable(ginputs,volatile=True)
        hinputs,ginputs = Variable(hinputs), Variable(ginputs)


        loading_time += time.time() - end
        optimizer.zero_grad()

        #loss
        loss, target = wrap_net(hinputs, ginputs, args.per_w)
        sumloss = args.rec_w * loss[0] + args.per_w * loss[1]
        if len(gids) > 1:
            sumloss.backward(th.ones(len(loss[0])))
            running_loss += th.sum(sumloss).data[0]
        else:
            sumloss.backward()
            running_loss += sumloss.data[0]

        optimizer.step()

        running_time += time.time() - end
        end = time.time()


        if bi % args.print_freq == 1:
            print 'training epoch: %d, minibatch: %d, loss: %f,  total time/mb: %f ms, running time/mb: %f ms'%(
                    epoch, bi, running_loss/(bi+1),
                    running_time/(bi+1)*1000.0, (running_time-loading_time)/(bi+1)*1000.0)
            print 'ep%d mb%d  loss details: '%(epoch, bi), [x.data[0] for x in loss]
    return running_loss/len(tr_loader), running_time, loading_time


def test(epoch, save_folder=None):
    wrap_net.eval()
    running_loss = 0.0
    running_time = 0.0
    loading_time = 0.0
    end = time.time()
    for bi,(hinputs, ginputs) in enumerate(te_loader):
        if use_cuda:
            hinputs, ginputs = hinputs.cuda(async=True), ginputs.cuda(async=True)
        hinputs,ginputs = Variable(hinputs,volatile=True), Variable(ginputs,volatile=True)


        loading_time += time.time() - end
        #loss
        loss, target = wrap_net(hinputs, ginputs, args.per_w)
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                os.chmod(save_folder, 0o777)
            print 'saving mb', bi
            save_himgs(save_folder, bi, hinputs, ginputs, target)
        if len(gids) > 1:
            running_loss += th.sum(loss[0]).data[0]
        else:
            running_loss += loss[0].data[0]

        running_time += time.time() - end
        end = time.time()

    return running_loss/len(te_loader), running_time, loading_time



####################################################  main #######################

if args.test_flag:
    save_folder = args.save_image
    te_l,running_time,loading_time=test(0, save_folder)
    print '**testing, mb: %d * %d, loss: %f,  total time/mb: %f ms, running time/mb: %f ms, total time/epoch: %f s,'%(
            args.test_batch_size, len(te_loader), te_l,
            running_time/len(te_loader)*1000.0, (running_time-loading_time)/len(te_loader)*1000.0, running_time)
    print 'images saved to ', save_folder
else:
    save_file = get_save_file(args)
    while epoch < args.epochs:
        epoch += 1
        print 'taining epoch', epoch
        mko.adjust_learning_rate(optimizer, args.lr, args, epoch)
        tr_l, running_time, loading_time = train(epoch)
        print '**training epoch: %d, mb: %d * %d, loss: %f,  total time/mb: %f ms, running time/mb: %f ms, total time/epoch: %f s,'%(
                epoch, args.batch_size, len(tr_loader), tr_l,
                running_time/len(tr_loader)*1000.0, (running_time-loading_time)/len(tr_loader)*1000.0, running_time)
        if math.isnan(tr_l) or math.isinf(tr_l) or tr_l > 1e5:
            print 'stop for abnormal  training'
            break
    te_l,running_time,loading_time = test(epoch)
    print '**validating epoch: %d, mb: %d * %d, loss: %f,  total time/mb: %f ms, running time/mb: %f ms, total time/epoch: %f s,'%(
        epoch, args.test_batch_size, len(te_loader), te_l,
        running_time/len(te_loader)*1000.0, (running_time-loading_time)/len(te_loader)*1000.0, running_time)
    save_model(epoch, wrap_net,  optimizer, save_file)
    print 'model saved to ', save_file, 'epoch', epoch, 'loss', te_l

print 'complete!', datetime.now()
