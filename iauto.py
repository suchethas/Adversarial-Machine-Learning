from __future__ import print_function, division
import numpy
from tqdm import tqdm
import pandas as pd
import xlwt
from xlwt import Workbook
from data_loader import ImageFolder
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
import os
import csv
plt.ion()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])

class DriveData(Dataset):
    __xs = []
    __ys = []
    __path = []

    def __init__(self, transform):
        self.transform = transform
        with open("image.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])
                self.__ys.append(row[1])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1 = Image.open(self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        # Convert image and label to torch tensors
        img1 = torch.from_numpy(np.asarray(img1))

        img2 = Image.open(self.__ys[index])
        img2 = img2.convert('L')
        if self.transform is not None:
            img2 = self.transform(img2)
        # Convert image and label to torch tensors
        img2 = torch.from_numpy(np.asarray(img2))

        return img1, img2, self.__xs[index]

    def __len__(self):
        return len(self.__xs)

res_dir = 'imagenet_autoencoder_notpretrained'

dsets = DriveData(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 25, shuffle = True, num_workers=4)


model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)

#model.load_state_dict(torch.load('model_imageclass10.pth'))
modified_model = nn.Sequential(*list(model.features.children()))

def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False

#freeze_layer(modified_model)
#modified_model.to(device)
def make_layers_decoder(cfg_decoder, batch_norm=False):
    layers_decoder = []
    s = [512, 512, 256, 128, 64]
    si = 0
    inn_channels = s[si]
    for v in range(len(cfg_decoder)):
        if cfg_decoder[v] == 'M':
            trans_conv2d = nn.ConvTranspose2d(inn_channels, s[si], kernel_size=2, stride = 2, padding=0)
            layers_decoder += [trans_conv2d, nn.ReLU(inplace=True)]
            si = si+1
        else:
            trans_conv2d = nn.ConvTranspose2d(inn_channels, cfg_decoder[v], kernel_size=3, padding=1)
            if batch_norm:
                if v<=len(cfg_decoder)-2:
                    if cfg_decoder[v+1]=='M':
                        #layers_decoder += [trans_conv2d, nn.BatchNorm2d(cfg_decoder[v])]
                        layers_decoder += [trans_conv2d]
                    else:
                        layers_decoder += [trans_conv2d, nn.BatchNorm2d(cfg_decoder[v]), nn.ReLU(inplace=True)]
                else:
                    layers_decoder += [trans_conv2d]
                #layers_decoder += [trans_conv2d, nn.BatchNorm2d(cfg_decoder[v]), nn.ReLU(inplace=True)]
            else:
                if v<=len(cfg_decoder)-2:
                    if cfg_decoder[v+1]=='M':
                        layers_decoder += [trans_conv2d]
                    else:
                        layers_decoder += [trans_conv2d, nn.ReLU(inplace=True)]
                else:
                    layers_decoder += [trans_conv2d]
            inn_channels = cfg_decoder[v]
    return nn.Sequential(*layers_decoder)


# class Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode

#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x

# def make_layers_decoder(cfg_decoder, batch_norm=False):
#     layers_decoder = []
#     s = [14, 28, 56, 112, 224]
#     si = 0
#     inn_channels = 512
#     for v in range(len(cfg_decoder)):
#         if cfg_decoder[v] == 'M':
#             #trans_conv2d = nn.UpsamplingBilinear2d(size=(s[si], s[si]))
#             trans_conv2d = Interpolate(size=(s[si], s[si]), mode='bilinear')
#             layers_decoder += [trans_conv2d, nn.ReLU(inplace=True)]
#             si = si+1
#         else:
#             trans_conv2d = nn.ConvTranspose2d(inn_channels, cfg_decoder[v], kernel_size=3, padding=1)

#             if v<=len(cfg_decoder)-2:
#                 if cfg_decoder[v+1]=='M':
#                     layers_decoder += [trans_conv2d]
#                 else:
#                     layers_decoder += [trans_conv2d, nn.ReLU(inplace=True)]
#             else:
#                 layers_decoder += [trans_conv2d]
#             inn_channels = cfg_decoder[v]
#     return nn.Sequential(*layers_decoder)

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
        x = self.handlesequential_decoder(self.decoder,x)
        return x
    
    def handlesequential_decoder(self,layer,x):
        for i,module in enumerate(layer.children()):
                x = module(x)
        return x
    
ae = Autoencoder()
ae.cuda()
def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
        
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), weight_decay=1e-5)
num_epoch = 20
for epoch in range(num_epoch):
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        inputs = inp_data[0].cuda()
        targets = inp_data[1].cuda()
        #print(inputs)
        paths = inp_data[2]

        outputs = ae(inputs)
        if epoch == num_epoch-1 or epoch==5:
            with torch.no_grad():
                for xi in range(25):
                    save_path = os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
                    create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
                    utils.save_image(outputs[xi,:,:,:], save_path + '_auto.png',  normalize=True, padding=0)
                    utils.save_image(targets[xi,:,:,:], save_path + '_orig.png',  normalize=True, padding=0)
        #print(outputs.shape, targets.shape)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epoch, loss.item()))
        #print('batch [{}/{}], loss:{:.4f}'.format(batch_idx, len(dsets)/25, loss.item()))
