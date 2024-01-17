#!/usr/bin/python3

import argparse
import sys
import os
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from datasets2  import TestDatasetFromFolder22,ImageDataset1,ImageDataset2,TestDatasetFromFolder1,TestDatasetFromFolder2
from skimage.metrics import structural_similarity as ski_ssim
# from skimage.measure import compare_ssim as ski_ssim
import  numpy as np
# from model.networks import *
import torchvision.utils as vutils
import math
# from GFN11221126 import *
from  GFN0503o1channelsor import *
import cv2
# import  tensorboard  as tb
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
#parser.add_argument('--dataroot', type=str, default='/home/omnisky/volume/9cyclegan/datasets/haze', help='root directory of the dataset')#/ datasets/horse2zebra/ /home/omnisky/volume/9cyclegan/PyTorch-CycleGAN-master0507/datasets/haze/

parser.add_argument('--dataroot', type=str, default='/home/omnisky/4t/JTY/testdataset', help='root directory of the dataset')#/home/omnisky/4t/dataset
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default='Ture',help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
#parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B_60.pth', help='A2B generator checkpoint file')
#parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A_60.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks


net_DISH= Net_DIS()
# print(net_DIS)
net_RECC= Net_RECC()


net_DISH.cuda()

net_RECC.cuda()


# Load state dicts  `
net_DISH.load_state_dict(torch.load('./output/net_DIS_11.pth'))#unsupervised
net_RECC.load_state_dict(torch.load('./output/net_RECC_11.pth'))#unsupervised


# net_DISH.load_state_dict(torch.load('./net_DIS.pth'))#unsupervised
# net_RECC.load_state_dict(torch.load('./net_RECC.pth'))#unsupervised

# print("net_DISH have {} parameters in total".format(sum(x.numel() for x in net_DISH.parameters())))
# print("net_RECC have {} parameters in total".format(sum(x.numel() for x in net_RECC.parameters())))
# print("netG_content have {} parameters in total".format(sum(x.numel() for x in netG_content.parameters())))
# net_RECC.eval()
# net_DISH.eval()
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor


transforms_ = [ transforms.ToTensor
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
dataloader = DataLoader(TestDatasetFromFolder22('/home/omnisky/4t/JTY/testdataset/Remote sensing'))
# dataloader = DataLoader(TestDatasetFromFolder1('/home/omnisky/4t/JTY/testdataset') )
# dataloader = DataLoader(TestDatasetFromFolder2('/home/omnisky/4t/JTY/testdataset'))
################################### /home/omnisky/4t/JTY/testdataset/test_new
###### Testing######
#meta = torch.load("/home/omnisky/volume/3meta-opera/model/dehaze_80.pth",map_location="cuda:0")
# Create output dirs if they don't exist
if not os.path.exists('./output/A'):
    os.makedirs('./output/A')
if not os.path.exists('./output/B'):
    os.makedirs('./output/B')
if not os.path.exists('output/C'):
    os.makedirs('./output/C')


test_ite = 0
test_psnr = 0
test_ssim = 0
eps = 1e-10
tt = 0
with torch.no_grad():
 for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(batch['A']).cuda()
    real_B = Variable(batch['B']).cuda()
    h, w = real_B.size(2), real_B.size(3)

    pad_h = h % 8
    pad_w = w % 8
    real_B = real_B[:, :, 0:h - pad_h, 0:w - pad_w]
    real_A = real_A[:, :, 0:h - pad_h, 0:w - pad_w]

    # t0 = time.time()
    content_B, mask_B  = net_DISH(real_B)

    dehaze_BBB = net_RECC(content_B  )  #dehaze_B1, dehaze_B2,
    # t1 = time.time()
    # tt = tt + t1 - t0
    # print('time:', str((t1 - t0)))

    output = dehaze_BBB#.resize(h,w)#cv2.resize(dehaze_BBB, (dehaze_BBB.size(0),dehaze_BBB.size(1),h,w), interpolation=cv2.INTER_CUBIC) #.resize_(:,:,h,w) #.resize[:,:,h,w]

    hr_patch = (real_A)

    vutils.save_image(real_A.data, './output/A/%04d.png' % (int(i)), padding=0, normalize=True)  # False
    vutils.save_image(real_B.data, './output/B/%04d.png' % (int(i)), padding=0, normalize=True)
    vutils.save_image(dehaze_BBB.data, './output/C/%04d.png' % (int(i)), padding=0, normalize=True)#True
    #print(dehaze_B)
    #print(real_A)


    output = output.data.cpu().numpy()[0]
    output[output >1] = 1
    output[output < 0] = 0
    output = output.transpose((1, 2, 0))
    #print(output)
    hr_patch = real_A.data.cpu().numpy()[0]
    hr_patch[hr_patch > 1] = 1
    hr_patch[hr_patch < 0] = 0
    hr_patch = hr_patch.transpose((1, 2, 0))
    #print(hr_patch)
    # SSIM
    ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    test_ssim += ssim  # ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    # PSNR
    imdf = (output - hr_patch) ** 2
    mse = np.mean(imdf) + eps
    psnr = 10 * math.log10(1.0 / mse)
    #psnr = compare_psnr(output,hr_patch)

    test_psnr += psnr  # 10 * math.log10(1.0/mse)
    test_ite += 1
    print('PSNR: {:.4f}'.format(psnr))
    print('SSIM: {:.4f}'.format(ssim))
    print('m_PSNR: {:.4f}'.format(test_psnr / test_ite))
    print('m_SSIM: {:.4f}'.format(test_ssim / test_ite))



    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
 print('mtt',str(tt/500))
 print('m_PSNR: {:.4f}'.format(test_psnr/test_ite))
 print('m_SSIM: {:.4f}'.format(test_ssim/test_ite))
 sys.stdout.write('\n')
###################################
