

#res_con3 = self.relu32(self.conv32(self.up32(content3*content[5][:, 0, ::]+content2*content[4][:, 0, ::]+content1*content[3][:, 0, ::])))

import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os
from torch import cat
# from SearchTransfer0703 import *
# from transformer0826 import *
import functools
import  torch.nn.functional as F

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 3, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 3, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel, 1, int((kernel-1)/2), bias=True),
            nn.ReLU(inplace=True),#nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel, 1,int((kernel-1)/2), bias=True)
        )
        self.calayer = CALayer(outchannel)
        # self.palayer = PALayer(outchannel)

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = self.calayer(out)
        # out = self.palayer(out)
        out = torch.add(residual, out)
        return out
class _ResBlockSR(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,kernel, 1,int((kernel-1)/2), bias=True),
            nn.ReLU(inplace=True),#nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel, 1, int((kernel-1)/2), bias=True)
        )

        self.calayer = CALayer(outchannel)
        # self.palayer = PALayer(outchannel)
    #
    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = self.calayer(out)
        # out = self.palayer(out)
        out = torch.add(residual, out)
        return out


class _ResBLockRE(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(_ResBLockRE, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel, 1, int((kernel-1)/2), bias=True),
            nn.ReLU(inplace=True),#nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel, 1, int((kernel-1)/2), bias=True)
        )
        self.calayer = CALayer(outchannel)
        # self.palayer = PALayer(outchannel)

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = self.calayer(out)
        # out = self.palayer(out)
        out = torch.add(residual, out)
        return out



class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        # x = x
        return x
class OALayer(nn.Module):
    def __init__(self, num_ops=3,channel=224,k=1 ):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        # self.batch =batch
        self.output =  k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops ))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        # y = y.view(-1, self.k, self.num_ops)#y = y.view(-1, 3, 2)
        return y
class Net_DIS(nn.Module):
    def __init__(self ):
        super(Net_DIS, self).__init__()


        self.conv1 = nn.Conv2d(3 , 16, (3, 3), 1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), 1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.resBlockC1 = self._makelayersC(32, 32, 4, 3)
        self.conv22 = nn.Conv2d(32, 64, (1, 1), 1, padding=0)
        self.relu22 = nn.ReLU(inplace=True)

        self.resBlockC2 = self._makelayersC(64, 64, 4, 3)

        self.conv32 = nn.Conv2d(64, 128, (1, 1), 1, padding=0)
        self.relu32 = nn.ReLU(inplace=True)

        self.resBlockC3 = self._makelayersC(128, 128, 4, 3)

        self.caD1 = nn.Sequential(*[
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(32, 32// 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32//4, 32 , 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.caD11 = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32 // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // 4, 32, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.caD2 = nn.Sequential(*[
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(96,96// 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(96// 4, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.caD22 = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96,96 // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(96// 4, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.caD3= nn.Sequential(*[
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(192,192// 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192// 4, 128, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.caD33= nn.Sequential(*[
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(192, 192 // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192 // 4, 128, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.attention3 = OALayer(num_ops=3,channel=224 )
        self.conv13 = nn.Conv2d(32, 32, (1, 1), 1, padding=0)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv23 = nn.Conv2d(32, 32, (1, 1), 1, padding=0)
        self.relu23 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)
        self.relu33 = nn.ReLU(inplace=True)
        self.conv43 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)
        self.relu43 = nn.ReLU(inplace=True)

    def _makelayersC(self, inchannel, outchannel, block_num, kernal,stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel,kernal))
        return nn.Sequential(*layers)
  

    def forward(self, x ):
        # feat = []
        content = []
        mask = []
        # if s <2:
        con1 = self.relu1(self.conv1(x))
        res_con1 = self.relu2(self.conv2(con1))
        res_con1 = self.resBlockC1(res_con1)
        # else:
        # res_con1 = self.resBlockC1(res_con1)

        wD1 = self.caD1(res_con1)
        content1 = wD1  * res_con1
        wD11 = self.caD11(res_con1)
        mask1 = wD11  * res_con1
        content.append(content1)
        mask.append(mask1)

        res_con2 =  self.relu22(self.conv22( (res_con1 )))# self.down22
        res_con2 = self.resBlockC2(res_con2)#64

        content1 = self.relu13(self.conv13((content1)))#self.down13self.down13
        mask1 = self.relu23(self.conv23((mask1)))#self.down23

        wD2 = self.caD2(cat([res_con2,content1 ],1))
        content2 = wD2  *res_con2
        wD22 = self.caD22(cat([res_con2, mask1], 1))
        mask2 = wD22 *res_con2
        content.append(content2)
        mask.append((mask2))

        res_con3 = self.relu32(self.conv32((res_con2 )))# self.down32 self.down32
        res_con3 = self.resBlockC3(res_con3 )#128

        content2 = self.relu33(self.conv33((content2)))#self.down33
        mask2 = self.relu43(self.conv43((mask2)))#self.down43

        wD3 = self.caD3(cat([res_con3,content2],1))
        content3 = wD3 *  res_con3

        wD33 = self.caD33(cat([res_con3, mask2], 1))
        mask3 = wD33  *  res_con3
        content.append(content3)
        mask.append((mask3))

        a1 =((wD11 - wD1) )
        a2 =((wD22 - wD2) )
        a3 =((wD33 - wD3) )


        fea3 = torch.cat([a1, a2, a3], 1)
        weights2 = F.softmax(self.attention3(fea3), dim=-1)
        content.append(weights2)#4

        content.append(fea3 )#6

        return content,mask#, feat



class Net_DISS(nn.Module):
    def __init__(self ):
        super(Net_DISS, self).__init__()
        self.dis  =Net_DIS()
        self.attention2 = OALayer(num_ops=3, channel=448)
        self.conv13 = nn.Conv2d(128* 2, 128 , (1, 1), 1, padding=0)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv23 = nn.Conv2d(64  * 2, 64 , (1, 1), 1, padding=0)
        self.relu23 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(32  * 2, 32 , (1, 1), 1, padding=0)
        self.relu33 = nn.ReLU(inplace=True)

        self.conv43 = nn.Conv2d(128 * 2, 128, (1, 1), 1, padding=0)
        self.relu43 = nn.ReLU(inplace=True)
        self.conv53 = nn.Conv2d(64 * 2, 64, (1, 1), 1, padding=0)
        self.relu53 = nn.ReLU(inplace=True)
        self.conv63 = nn.Conv2d(32 * 2, 32, (1, 1), 1, padding=0)
        self.relu63 = nn.ReLU(inplace=True)

    def forward(self,  con ,mask, x,s=1 ):
        if s>1:
            content2=[]
            mask2=[]
            content21, mask21 = self.dis(x )
            content22 = con
            mask22 = mask

            content2.append(self.relu33(self.conv33(cat([content21[0],content22[0]],1))))
            content2.append(self.relu23(self.conv23(cat([content21[1], content22[1]], 1))))
            content2.append(self.relu13(self.conv13(cat([content21[2], content22[2]], 1))))

            feat2 = cat([content21[4], content22[4]], 1)
            weight2 = self.attention2(feat2)
            content2.append(weight2)#4
            content2.append( content21[4])

            mask2.append(self.relu63(self.conv63(cat([mask21[0], mask22[0]], 1))))
            mask2.append(self.relu53(self.conv53(cat([mask21[1], mask22[1]], 1))))
            mask2.append(self.relu43(self.conv43(cat([mask21[2], mask22[2]], 1))))
        else:
            content2, mask2 = self.dis(x )

        return  content2, mask2#, feat2


class Net_RECC(nn.Module):
    def __init__(self,stage=1):
        super(Net_RECC, self).__init__()

        self.resBlockG4 = self._makelayersG(128*stage, 128*stage, 4, 3)
        # self.up12 = SkipUpSample(128*stage, 0)
        self.conv12 = nn.Conv2d(128*stage,64*stage, (1, 1), 1, padding=0)
        self.relu12 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.resBlockG3 = self._makelayersG(64*stage, 64*stage, 4, 3)
        # self.up22 = SkipUpSample(64*stage, 0)
        self.conv22 = nn.Conv2d(64*stage,32*stage, (1, 1), 1, padding=0)
        self.relu22 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.resBlockG2 = self._makelayersG(32*stage, 32*stage,4, 3)
        self.conv1 = nn.Conv2d(32*stage, 16, (3, 3), 1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(16, 3, (3, 3), 1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.calayer = CALayer(32*stage)
        self.palayer = PALayer(32*stage)



    def _makelayersG(self, inchannel, outchannel, block_num, kernel):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockRE(inchannel, outchannel, kernel))
        return nn.Sequential(*layers)

    def forward(self, content ):

        content3 = self.resBlockG4(  content[2]*content[3][:, :, 2].view([-1, 1, 1, 1]))
        res_con3 = self.relu12(self.conv12((content3)))# self.up12
        content3 = self.resBlockG3(res_con3 +content[1]*content[3][:, :,1].view([-1, 1, 1, 1]) )#[:,0:32,:,:]
        res_con3 = self.relu22(self.conv22((content3)))# self.up22
        content3= self.resBlockG2(res_con3 +content[0]*content[3][:, :, 0].view([-1, 1, 1, 1]) )#[:,0:32,:,:]

        res_con3 = self.calayer(content3 )
        res_con3 = self.palayer(res_con3)
        dehaze3 = self.relu1(self.conv1(res_con3))
        dehaze3 = self.relu2(self.conv2( dehaze3))#+ x self.relu2

        return     dehaze3

class Net_RECH(nn.Module):
    def  __init__(self,stage):
        super(Net_RECH, self).__init__()

        self.resBlockG4 = self._makelayersG(64*2, 64*2, 4, 3)
        # self.up12 = SkipUpSample(64*2, 0)
        self.conv12 = nn.Conv2d(64*2, 64, (1, 1), 1, padding=0)
        self.relu12 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.resBlockG3 = self._makelayersG(64, 64, 4, 3)
        # self.up22 = SkipUpSample(64, 0)
        self.conv22 = nn.Conv2d(64, 32, (1, 1), 1, padding=0)
        self.relu22 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.resBlockG2 = self._makelayersG(32, 32, 4, 3)
        self.conv1 = nn.Conv2d(32, 16, (3, 3), 1, padding=1)
        self.relu1 =nn.ReLU(inplace=True)# nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(16, 3, (3, 3), 1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)

        self.calayer = CALayer(32*stage)
        self.palayer = PALayer(32*stage)

        self.conv23 = nn.Conv2d(128  *2, 64 * 2, (1, 1), 1, padding=0)
        self.relu23 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)


    def _makelayersG(self, inchannel, outchannel, block_num, kernel):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockRE(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self,  content,mask,s=1 ):

        rapair2 = cat([content[2],mask[2]],1)
        rapair2 = (self.relu23(self.conv23(rapair2)))

        res_con3 = self.resBlockG4(  rapair2 )  # 32
        res_con3 = self.relu12(self.conv12( (res_con3)))#self.up12
        res_con3 = self.resBlockG3(res_con3 )  # [:,0:32,:,:]
        res_con3 = self.relu22(self.conv22((res_con3)))#self.up22
        res_con3 = self.resBlockG2(res_con3 )
        res_con3 = self.calayer(res_con3)
        res_con3 = self.palayer(res_con3)
        recover3 = self.relu1(self.conv1(res_con3))
        recover3  = self.relu2(self.conv2(recover3))

        return  recover3



class netDC(nn.Module):
    def __init__(self):
        super(netDC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(16, 8)#(144,4)#(16, 8)#(144,1)

    def forward(self, input):
        output = self.layer1(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output.view(-1)

class netDH(nn.Module):
    def __init__(self):
        super(netDH, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(16, 8)#(144,4)#(16, 8)#(144,1

    def forward(self, input):
        output = self.layer1(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output.view(-1)

