#!/usr/bin/python3

import argparse
import sys
import os
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from datasets2  import TestDatasetFromFolder22,ImageDataset1,ImageDataset2,TestDatasetFromFolder1,TestDatasetFromFolder2
from skimage.metrics import structural_similarity as ski_ssim
import  numpy as np
import torchvision.utils as vutils
from  GFN0430o1channelsuninl1 import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/omnisky/4t/JTY/testdataset', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default='Ture',help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
net_DISH= Net_DIS()
net_RECC= Net_RECC()
net_DISH.cuda()
net_RECC.cuda()


# Load state dicts  `
#for real-world hazy image dehazing
net_DISH.load_state_dict(torch.load('./output/net_DIS_8.pth'))
net_RECC.load_state_dict(torch.load('./output/net_RECC_8.pth'))

#for synthetic hazy image dehazing
# net_DISH.load_state_dict(torch.load('./output/net_DIS_2.pth'))
# net_RECC.load_state_dict(torch.load('./output/net_RECC_2.pth'))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

dataloader = DataLoader(TestDatasetFromFolder22('/home/omnisky/4t/JTY/testdataset/test_new'))

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
    dehaze_BBB = net_RECC(content_B  )

    output = dehaze_BBB
    hr_patch = (real_A)

    vutils.save_image(real_A.data, './output/A/%04d.png' % (int(i)), padding=0, normalize=True)  # False
    vutils.save_image(real_B.data, './output/B/%04d.png' % (int(i)), padding=0, normalize=True)
    vutils.save_image(dehaze_BBB.data, './output/C/%04d.png' % (int(i)), padding=0, normalize=True)#True


    output = output.data.cpu().numpy()[0]
    output[output >1] = 1
    output[output < 0] = 0
    output = output.transpose((1, 2, 0))
    hr_patch = real_A.data.cpu().numpy()[0]
    hr_patch[hr_patch > 1] = 1
    hr_patch[hr_patch < 0] = 0
    hr_patch = hr_patch.transpose((1, 2, 0))
    # SSIM
    ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    test_ssim += ssim
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
