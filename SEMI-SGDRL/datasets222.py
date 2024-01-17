import glob
import random
import os

from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose,ToPILImage, RandomCrop, CenterCrop, Resize  ,ToTensor, Normalize
import torchvision.transforms as transforms
import  natsort
#import  h5py
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp','.BMP'])


def calculate_valid_crop_size(crop_size):
    return crop_size



def train_h_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),

        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

def train_s_transform(crop_size):
    return Compose([
        # CenterCrop(crop_size),
        RandomCrop(crop_size),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    ])

def train_trans_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        # RandomCrop(crop_size),
        ToTensor(),

    ])
def test_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='OTS_B'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/clear' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset1(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='real'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/gt' % mode) + '/*.*'))#testA
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))#testB

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset2(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='test-rrrrrr'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/h' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/t' % mode) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/testA'#'/hazy'
        self.s_path = dataset_dir + '/testB'#'/clear'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return {'A': h_image, 'B': s_image}

    def __len__(self):
        return len(self.h_filenames)

class TestDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder1, self).__init__()

        self.h_path = dataset_dir + '/outdoor' + '/hazy'  # '/hazy' indoor
        self.s_path = dataset_dir + '/outdoor' + '/gt'  # '/clear'

        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))  if is_image_file(x)]

        self.s_transform = test_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image =  self.s_transform((Image.open(self.h_filenames[index])))
        s_image = self.s_transform( (Image.open(self.s_filenames[index])))
        return {'A': s_image, 'B': h_image}
        # return {'A': h_image, 'B': s_image}

    def __len__(self):
        return len(self.h_filenames)

class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder2, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/indoor'+'/hazy'#'/hazy' indoor
        self.s_path = dataset_dir + '/indoor'+'/gt'#'/clear'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if is_image_file(x)]#


    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image =  ToTensor()(Image.open(self.h_filenames[index]))
        s_image =  ToTensor()(Image.open(self.s_filenames[index]))
        return {'A': s_image, 'B': h_image}

    def __len__(self):
        return len(self.h_filenames)


class TrainDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s,dataset_dir_trans, crop_size):
        super(TrainDatasetFromFolder1, self).__init__()
        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:2061] if is_image_file(x)]
        self.image_filenames_trans = [join(dataset_dir_trans, x) for x in natsort.natsorted(listdir(dataset_dir_trans))[0:2061] if is_image_file(x)]

        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:2061] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)
        self.trans_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        trans_image = self.trans_transform(Image.open(self.image_filenames_trans[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        #return h_image, s_image

        return {'A': h_image, 'B': s_image,'T':trans_image}

    def __len__(self):
        return  len(self.image_filenames_h) #max(len(self.image_filenames_h), len(self.image_filenames_s))



class TrainDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,crop_size ):
        super(TrainDatasetFromFolder2, self).__init__()
        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:1399] for p in range(10) if is_image_file(x)]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:13990]  if is_image_file(x)]


        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_s_transform(crop_size)
        self.h_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.h_transform(Image.open(self.image_filenames_B[index]))
        #return h_image, s_image

        return {'A': A_image, 'B':B_image}#'B': s_image,

    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))

class TrainDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,dataset_real,crop_size ):
        super(TrainDatasetFromFolder3, self).__init__()
        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:1399] for p in range(10) if is_image_file(x)]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:13990]  if is_image_file(x)]
        self.image_filenames_C = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real))[0:13990]   if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_s_transform(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.r_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.h_transform(Image.open(self.image_filenames_B[index]))
        R_image = self.r_transform(Image.open(self.image_filenames_C[index]))

        #return h_image, s_image

        return {'A': A_image, 'R':B_image,'RR':R_image}#'B': s_image,

    def __len__(self):
        return  len(self.image_filenames_A)


class TrainDatasetFromFolder4(Dataset):
    def __init__(self, dataset_dir_c, dataset_dir_h,dataset_real, crop_size):
        super(TrainDatasetFromFolder4, self).__init__()

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:2061] for   p in range(7) if is_image_file(  x)]  # for p in range(10)#5687    27327   2000  #is_image_file(10*x)] [0:5687]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:14427] if   is_image_file(x)]  # [0:5687]  54600
        self.image_filenames_R = [join(dataset_real, x) for x in natsort.natsorted(listdir(dataset_real))[0:14427] if is_image_file(x)]


        crop_size = calculate_valid_crop_size(crop_size)
        self.r_transform = train_s_transform(crop_size)
        self.c_transform = train_trans_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        R_image = self.c_transform(Image.open(self.image_filenames_B[index]))

        RR_image = self.r_transform(Image.open(self.image_filenames_R[index]))

        return {'A': A_image, 'R':R_image, 'RR':RR_image}#'B': s_image,

    def __len__(self):
        return  len(self.image_filenames_A)
class TrainDatasetFromFolder5(Dataset):
    def __init__(self, dataset_dir_c, dataset_dir_h, crop_size):
        super(TrainDatasetFromFolder5, self).__init__()

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:2061] for p in range(11) if is_image_file(x)]   # for p in range(10)#5687    27327   2000  #is_image_file(10*x)] [0:5687]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:22671] if is_image_file(x)]  #[0:5687]  54600 22671 14427

        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.c_transform(Image.open(self.image_filenames_B[index]))
        return {'A': A_image,'B': B_image }#'B': s_image,


    def __len__(self):
        return  len(self.image_filenames_A) #max(len(self.image_filenames_h), len(self.image_filenames_s))
