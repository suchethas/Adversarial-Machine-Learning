from __future__ import print_function, division
import numpy
from tqdm import tqdm
import pandas as pd
import xlwt
from xlwt import Workbook
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torchvision.utils as utils
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import csv
plt.ion()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
PATH = '/home/shihong/imagenet-10/'
class Data(Dataset):
    __xs = []
    __ys = []
    __zs = []
    def __init__(self, transform):
        self.transform = transform
        with open("imageval.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0]) # original image
                self.__ys.append(row[2]) # adv image
                self.__zs.append(row[1]) # mask image

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1 = Image.open(PATH + self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        img1 = torch.from_numpy(np.asarray(img1))

        img2 = Image.open(PATH + self.__ys[index])
        img2 = img2.convert('RGB')
        if self.transform is not None:
            img2 = self.transform(img2)
        img2 = torch.from_numpy(np.asarray(img2))

        img3 = Image.open(PATH + self.__zs[index])
        img3 = img3.convert('L')
        if self.transform is not None:
            img3 = self.transform(img3)
        img3 = torch.from_numpy(np.asarray(img3))
        return img1, img2, img3, PATH + self.__xs[index]

    def __len__(self):
        return len(self.__xs)

res_dir = 'imagenet_autoencoder_result'

dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 50, shuffle = True, num_workers=4)

model = models.vgg16_bn(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
modified_model = nn.Sequential(*list(model.features.children())[:-1])


def make_layers_decoder(cfg_decoder, batch_norm=True):
    layers_decoder = []
    s = [512, 512, 256, 128, 64]
    si = 0
    inn_channels = s[si]
    for v in range(len(cfg_decoder)):
        if cfg_decoder[v] == 'M':
            trans_conv2d = nn.ConvTranspose2d(inn_channels, s[si], kernel_size=2, stride = 2, padding=0)
            layers_decoder += [trans_conv2d]
            si = si+1
        else:
            trans_conv2d = nn.ConvTranspose2d(inn_channels, cfg_decoder[v], kernel_size=3, padding=1)
            if batch_norm:
                layers_decoder += [trans_conv2d, nn.BatchNorm2d(cfg_decoder[v]), nn.ReLU(inplace=True)]
            else:
                if v<=len(cfg_decoder)-2:
                    if cfg_decoder[v+1]=='M':
                        layers_decoder += [trans_conv2d]
                    else:
                        layers_decoder += [trans_conv2d, nn.ReLU(inplace=True)]
                else:
                    layers_decoder += [trans_conv2d]
            inn_channels = cfg_decoder[v]
    layers_decoder = layers_decoder[1:-2]
    return nn.Sequential(*layers_decoder)

cfg_decoder = {
    'D' : ['M', 512, 512, 512, 'M', 512, 512,  256, 'M', 256, 256,  128, 'M', 128,  64, 'M', 64, 1],
}

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = modified_model
        self.decoder = make_layers_decoder(cfg_decoder['D'])
        
    def forward(self,x):
        x = modified_model(x)
        for i,module in enumerate(self.decoder.children()):
                x = module(x)
        # x = self.handlesequential_decoder(self.decoder,x)
        return x
    
    # def handlesequential_decoder(self,layer,x):
    #     for i,module in enumerate(layer.children()):
    #             x = module(x)
    #     return x
    
ae = Autoencoder()

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     ae = nn.DataParallel(ae)
criterion = nn.MSELoss(reduction='sum')
ae.to('cuda')
ae.load_state_dict(torch.load('./model/ae_fixed_m.pth'))


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
ae.eval()
with torch.no_grad():
    loss_orig = 0
    loss_adv = 0
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        inputs = inp_data[0].to('cuda')
        adv = inp_data[1].to('cuda')
        targets = inp_data[2].to('cuda')
        paths = inp_data[3]
        output0 = ae(inputs)
        output1 = ae(adv)

        loss_orig += criterion(output0, targets).item()/224.0/224.0
        loss_adv += criterion(output1, targets).item()/224.0/224.0

        for xi in range(len(output0)):
            save_path = os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
            create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
            utils.save_image(inv_normalize(output0[xi,:,:,:]), save_path + '_orig.png', padding=0)
            utils.save_image(inv_normalize(output1[xi,:,:,:]), save_path + '_adv.png', padding=0)
    loss_orig /= len(dset_loaders.dataset)
    loss_adv /= len(dset_loaders.dataset)
    print('\nTest set: Average loss for orig image: {:.4f}, Test set: Average loss for adv image: {:.4f}'.format(loss_orig, loss_adv))
