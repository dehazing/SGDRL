
import csv
import argparse
import itertools
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.models import vgg16
from perceptual import LossNetwork
from datasets222 import TrainDatasetFromFolder4,TrainDatasetFromFolder3,TrainDatasetFromFolder2,TestDatasetFromFolder1
from skimage.metrics import structural_similarity as ski_ssim
from GFN1012o1channels import *
from  utils import *
from  ECLoss import *
from CAPLOSS import *
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=4, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')#256
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default='Ture',help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
net_DIS = Net_DISS( )
net_RECC= Net_RECC(stage=1 )
net_RECH= Net_RECH(stage=1 )
net_DC= netDC( )

net_DIS.cuda()
net_RECC.cuda()
net_RECH.cuda()
net_DC.cuda( )

# Lossess
adversarial_loss = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(vgg_model).cuda()
loss_network.eval()

optimizer_DC= torch.optim.Adam(net_DC.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_G= torch.optim.Adam(itertools.chain(net_DIS.parameters() ,net_RECC.parameters() ,net_RECH.parameters()), lr=opt.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=100)

dataloader1 = DataLoader(TrainDatasetFromFolder3('/home/omnisky/4t/RESIDE/ITS-V2/trainA_new',
                                     '/home/omnisky/4t/RESIDE/ITS-V2/trainB_new',  '/home/omnisky/4t/realWorldHazeDataSet/trainB_newsize_128',crop_size= 128), batch_size=opt.batchSize,shuffle=True )  #SIDMS   /home/omnisky/volume/ITSV2/clear
dataloader2 = DataLoader(TrainDatasetFromFolder4('/home/omnisky/4t/RESIDE/OTS_BETA/clear/clear_newsize',
                                             '/home/omnisky/4t/RESIDE/OTS_BETA/haze/hazy7',  '/home/omnisky/4t/realWorldHazeDataSet/trainA_newsize_128', crop_size=128), batch_size=opt.batchSize,shuffle=True )

val_data_loader = DataLoader(TestDatasetFromFolder1('/home/omnisky/4t/JTY/testdataset'),
                       batch_size=1, shuffle=False, num_workers=opt.n_cpu)

# Loss plot
logger1 = Logger(opt.n_epochs, len(dataloader1))
logger2 = Logger(opt.n_epochs, len(dataloader2))
###################################
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('./results'):
    os.makedirs('./results/Inputs')
    os.makedirs('./results/Outputs')
    os.makedirs('./results/Targets')
FloatTensor = torch.cuda.FloatTensor

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    if not epoch % 2:
        dataloader = dataloader1
    else:
        dataloader = dataloader2
    ite = 0
    adjust_learning_rate(optimizer_G, epoch)

    for i, batch in enumerate(dataloader):

      real_A = Variable(batch['A']).cuda(0)
      real_B = Variable(batch['R']).cuda(0)
      real_R = Variable(batch['RR']).cuda(0)
      if real_A.size(1) == 3 and real_A.size(1) == 3:
        ite += 1
        valid = Variable(FloatTensor(opt.batchSize, 1).fill_(1.0), requires_grad=False).cuda( )
        fake = Variable(FloatTensor(opt.batchSize, 1).fill_(0.0), requires_grad=False).cuda( )

        content_B, mask_B = net_DIS(real_B,real_B,real_B)
        content_A, mask_A  = net_DIS(real_A,real_A,real_A)

        fake_H = net_RECH( content_A,mask_B )
        fake_C =net_RECH(content_B,mask_A)
        dehaze_B  = net_RECC(content_B )
        dehaze_A = net_RECC(content_A)

        content_BB,mask_BB = net_DIS(content_B,mask_B,dehaze_B,s=2)
        dehaze_BB  = net_RECC(content_BB)

        content_AA, mask_AA = net_DIS(content_A,mask_A,dehaze_A,s=2)
        dehaze_AA = net_RECC(content_AA)

        fake_H2 = net_RECH(content_AA, mask_BB)
        fake_C2 = net_RECH(content_BB, mask_AA)

        optimizer_G.zero_grad()

        loss_dehaze =F.smooth_l1_loss(dehaze_A,real_A)+loss_network(dehaze_A,real_A) * 0.04 + \
                     F.smooth_l1_loss(dehaze_AA,real_A)+loss_network(dehaze_AA,real_A) * 0.04 +\
                     F.smooth_l1_loss(dehaze_B,real_A)+loss_network(dehaze_B,real_A) * 0.04 +\
                     F.smooth_l1_loss(dehaze_BB,real_A)+loss_network(dehaze_BB,real_A) * 0.04 +\
                     F.smooth_l1_loss(content_B[2], content_A[2])+ F.smooth_l1_loss(content_B[1], content_A[1])+ F.smooth_l1_loss(content_B[0], content_A[0])\
                     +F.smooth_l1_loss(content_BB[2], content_AA[2])+ F.smooth_l1_loss(content_BB[1], content_AA[1])+ F.smooth_l1_loss(content_BB[0], content_AA[0])

        loss_fake = F.smooth_l1_loss(fake_H, real_B) + loss_network(fake_H, real_B) * 0.04 +\
                     F.smooth_l1_loss(fake_H2, real_B) + loss_network(fake_H2, real_B) * 0.04 + \
                    F.smooth_l1_loss(fake_C, real_A) + loss_network(fake_C, real_A) * 0.04 +\
                     F.smooth_l1_loss(fake_C2, real_A) + loss_network(fake_C2, real_A) * 0.04 \

        loss_M =  10* loss_dehaze + 1*  loss_fake
        content_R, mask_R = net_DIS(real_R,real_R,real_R)
        dehaze_R  = net_RECC(content_R )

        content_RR, mask_RR = net_DIS(content_R,mask_R, dehaze_R,s=2)
        dehaze_RR = net_RECC(content_RR)

        net_DC.zero_grad()  # gradient to 0
        pred_dehazeR = net_DC(dehaze_RR)
        loss_DC_R = adversarial_loss(pred_dehazeR, fake)
        pred_real = net_DC(real_A)
        loss_DC_real = adversarial_loss(pred_real, valid)
        dC_lossG = (loss_DC_real + loss_DC_R) * 0.5
        dC_lossG.backward(retain_graph=True)
        optimizer_DC.step()

        Cpred_fakeR = net_DC(dehaze_RR)

        loss_GAN = adversarial_loss(Cpred_fakeR, valid) *0.5

        loss_DER =  loss_network(dehaze_R , real_R)  +loss_network(dehaze_RR , dehaze_R)
        loss_G1 =   loss_M  +0.001*loss_DER +0.1*loss_GAN
        # if loss_G <=100000:
        loss_G1.backward( )
        optimizer_G.step()

        if not epoch % 2:
            logger = logger1
        else:
            logger = logger2
        ###################################

        logger.log( {'loss_G1': loss_G1,'loss_M': loss_M, 'loss_DER': loss_DER,'loss_GAN': loss_GAN,'loss_dehaze': loss_dehaze,' loss_fake':  loss_fake })
        if ite % 100 == 0:
            # print(output)
            vutils.save_image(real_A.data, './real_A.png' , normalize=True)
            vutils.save_image(real_B.data, './real_B.png', normalize=True)
            vutils.save_image(dehaze_A.data, './dehaze_A.png', normalize=True)
            vutils.save_image(dehaze_B.data, './dehaze_B.png', normalize=True)
            vutils.save_image(dehaze_BB.data, './dehaze_BB.png', normalize=True)
            vutils.save_image(dehaze_R.data, './dehaze_R.png', normalize=True)
            vutils.save_image(dehaze_RR.data, './dehaze_RR.png', normalize=True)
            vutils.save_image(real_R.data, './real_R.png', normalize=True)
            vutils.save_image(fake_H.data, './fake_H.png', normalize=True)
            torch.save(net_DIS.state_dict(), 'output/net_DIS.pth' )
            torch.save(net_RECC.state_dict(), 'output/net_RECC.pth')

    lr_scheduler_G.step()

    torch.save(net_DIS.state_dict(), 'output/net_DIS_%d.pth' % int(epoch+1 ))
    torch.save(net_RECC.state_dict(), 'output/net_RECC_%d.pth' % int(epoch+1 ))
    torch.save(net_RECH.state_dict(), 'output/net_RECH_%d.pth' % int(epoch + 1 ))
    torch.save(net_DC.state_dict(), 'output/net_DC_%d.pth' % int(epoch + 1 ))

    if epoch % 1 == 0:
        with torch.no_grad():
            print('------------------------')
            test_psnr = 0
            test_ssim = 0
            eps = 1e-10
            test_ite = 0
            for i, batch in enumerate(val_data_loader):
                # Set model input
                real_A = Variable(batch['A']).cuda(0)  # clear
                real_B = Variable(batch['B']).cuda(0)
                h, w = real_B.size(2), real_B.size(3)

                pad_h = h % 8
                pad_w = w % 8
                real_B = real_B[:, :, 0:h - pad_h, 0:w - pad_w]
                real_A = real_A[:, :, 0:h - pad_h, 0:w - pad_w]
                content_B, mask_B  = net_DIS(real_B,real_B,real_B)
                dehaze_B = net_RECC(content_B)
                content_BB, mask_BB = net_DIS(content_B,mask_B, dehaze_B,s=2)
                dehaze_BB = net_RECC(content_BB)

                vutils.save_image(real_A.data, './results/Targets/%05d.png' % (int(i)), padding=0,
                                  normalize=True)  # False
                vutils.save_image(real_B.data, './results/Inputs/%05d.png' % (int(i)), padding=0, normalize=True)
                vutils.save_image(dehaze_BB.data, './results/Outputs/%05d.png' % (int(i)), padding=0, normalize=True)
                output = dehaze_BB.data.cpu().numpy()[0]
                output[output > 1] = 1
                output[output < 0] = 0
                output = output.transpose((1, 2, 0))
                hr_patch = real_A.data.cpu().numpy()[0]
                hr_patch[hr_patch > 1] = 1
                hr_patch[hr_patch < 0] = 0
                hr_patch = hr_patch.transpose((1, 2, 0))
                # SSIM
                test_ssim += ski_ssim(output, hr_patch, data_range=1, multichannel=True)
                # PSNR
                imdf = (output - hr_patch) ** 2
                mse = np.mean(imdf) + eps
                test_psnr += 10 * math.log10(1.0 / mse)
                test_ite += 1
            test_psnr /= (test_ite)
            test_ssim /= (test_ite)
            print('Valid PSNR: {:.4f}'.format(test_psnr))
            print('Valid SSIM: {:.4f}'.format(test_ssim))
            f = open('PSNR.txt', 'a')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([epoch, test_psnr, test_ssim])
            f.close()
            print('------------------------')

###################################

