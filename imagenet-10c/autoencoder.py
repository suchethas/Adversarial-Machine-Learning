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
        img1 = Image.open(PATH + self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        # Convert image and label to torch tensors
        img1 = torch.from_numpy(np.asarray(img1))

        img2 = Image.open(PATH + self.__ys[index])
        img2 = img2.convert('L')
        if self.transform is not None:
            img2 = self.transform(img2)
        # Convert image and label to torch tensors
        img2 = torch.from_numpy(np.asarray(img2))

        return img1, img2, PATH + self.__xs[index]

    def __len__(self):
        return len(self.__xs)

res_dir = 'imagenet_autoencoder_pretrained'

dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 32, shuffle = True, num_workers=4)

model = models.vgg16_bn(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
modified_model = nn.Sequential(*list(model.features.children())[:-1])
model.load_state_dict(torch.load('./model/model_imageclass10bn.pth'))

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
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

ae.to('cuda')
freeze_layer(ae.encoder)

def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
        
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epoch = 120
train_loss = []
for epoch in range(num_epoch):
    scheduler.step()
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        inputs = inp_data[0].to('cuda')
        targets = inp_data[1].to('cuda')
        paths = inp_data[2]
        outputs = ae(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        plt.plot(np.arange(len(train_loss)), train_loss)
        plt.ylim(0, 5)
        plt.savefig('./train_curve_ae_fixed.jpg')
        plt.close()
    with torch.no_grad():
        for xi in range(len(outputs)):
            save_path = os.path.join(res_dir, paths[xi].split('/')[4],paths[xi].split('/')[5].split(".")[0])
            create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
            utils.save_image(inv_normalize(outputs[xi,:,:,:]), save_path + '_auto.png', padding=0)
            utils.save_image(inv_normalize(targets[xi,:,:,:]), save_path + '_orig.png', padding=0)
    torch.save(ae.state_dict(), './model/ae_fixed_m.pth')
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epoch, loss.item()))

