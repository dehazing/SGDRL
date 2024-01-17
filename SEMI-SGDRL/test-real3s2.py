#!/usr/bin/python3
import argparse
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets2  import ImageDataset,ImageDataset1,ImageDataset2,TestDatasetFromFolder22,TestDatasetFromFolder2
from skimage.metrics import structural_similarity as ski_ssim
import  numpy as np
import torchvision.utils as vutils
from GFN1012o1channels import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/omnisky/4t/JTY/testdataset', help='root directory of the dataset')#/home/omnisky/4t/dataset
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
net_DIS1= Net_DISS ( )
net_RECC= Net_RECC(stage=1 )
net_RECC.cuda()
net_DIS1.cuda()
# Load state dicts  `
#for real world hazy images
net_DIS1.load_state_dict(torch.load('./output/net_DIS_3.pth'))#
net_RECC.load_state_dict(torch.load('./output/net_RECC_3.pth'))#
# for synthetic hazy images
# net_DIS1.load_state_dict(torch.load('./output/net_DIS_4.pth'))#
# net_RECC.load_state_dict(torch.load('./output/net_RECC_4.pth'))#
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
dataloader = DataLoader(TestDatasetFromFolder22('/home/omnisky/4t/JTY/testdataset/test_new'))
###################################
###### Testing######
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
    # h, w = real_B.size(2), real_B.size(3)

    # pad_h = h % 8
    # pad_w = w % 8
    # real_B = real_B[:, :, 0:h - pad_h, 0:w - pad_w]
    # real_A = real_A[:, :, 0:h - pad_h, 0:w - pad_w]
    content_B, mask_B = net_DIS1(real_B, real_B, real_B)

    dehaze_B = net_RECC(content_B)
    content_BB, mask_BB = net_DIS1(content_B, mask_B, dehaze_B,s=2)
    dehaze_BB = net_RECC(content_BB)
    output = dehaze_B
    hr_patch = (real_A)

    vutils.save_image(real_A.data, './output/A/%04d.png' % (int(i)), padding=0, normalize=True)  # False
    vutils.save_image(dehaze_BB.data, './output/B/%04d.png' % (int(i)), padding=0, normalize=True)
    vutils.save_image(dehaze_B.data, './output/C/%04d.png' % (int(i)), padding=0, normalize=True)#True



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
    test_ssim += ssim
    # PSNR
    imdf = (output - hr_patch) ** 2
    mse = np.mean(imdf) + eps
    psnr = 10 * math.log10(1.0 / mse)
    test_psnr += psnr
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
