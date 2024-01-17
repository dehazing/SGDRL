import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb
import cv2




def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return loss

# def EDLoss(img, patch_size):
#     """
#     calculating bright channel of image, the image shape is of N*C*W*H
#     """
#     patch_size = 35
#     dc = maxpool(img[:, None, :, :, :])
#
#     target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1)
#     loss = L1Loss(size_average=False)(dc, target)
#     return loss
    
if __name__=="__main__":
    img = Image.open('real_B.png')
    totensor = transforms.ToTensor()
    
    img = totensor(img)
    
    img = Variable(img[None, :, :, :].cuda(), requires_grad=True)    
    loss = DCLoss(img, 16)
    print(loss)
    
    # loss.backward()
def DELoss(dehaze,hazy):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    unloader = transforms.ToPILImage()
    image = dehaze.cpu().clone()
    image = image.squeeze(0)
    dehaze = unloader(image)

    unloader = transforms.ToPILImage()
    image = hazy.cpu().clone()
    image = image.squeeze(0)
    hazy = unloader(image)

    totensor = transforms.ToTensor()

    img1 = cv2.GaussianBlur(dehaze, (3, 3), 0)
    canny1 = cv2.Canny(img1, 50, 150)

    img2 = cv2.GaussianBlur(hazy, (3, 3), 0)
    canny2 = cv2.Canny(img2, 50, 150)

    # 形态学：边缘检测
    _, Thr_img = cv2.threshold(img1, 210, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度


    # l1loss = L1Loss()
    # loss = l1loss(ldehaze, lhazy )



    # loss = L1Loss(size_average=True)(S, V)
    return loss

    



