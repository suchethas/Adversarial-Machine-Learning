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


class Data(Dataset):
    def __init__(self,  transform):
        self.__xs = [] #input
        self.__ys = [] #gradcam-mask
        self.__zs = [] #ae-mask
        self.__ls = [] #label

        self.transform = transform
        with open("input_train_gcae_classifier.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])
                self.__ys.append(row[1])
                self.__zs.append(row[2])
                if "orig" in row[0]:
                    self.__ls.append(0)
                else: #perturb
                    self.__ls.append(1)

    def __getitem__(self, index):
        img1 = Image.open(self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)

        img2 = Image.open(self.__ys[index])
        img2 = img2.convert('L')
        if self.transform is not None:
            img2 = self.transform(img2)
        
        img3 = Image.open(self.__zs[index])
        img3 = img3.convert('L')
        if self.transform is not None:
            img3 = self.transform(img3)

        a = np.zeros(shape=(5,224,224))
        a[0,:,:]=img1[0,:,:]
        a[1,:,:]=img1[1,:,:]
        a[2,:,:]=img1[2,:,:]
        a[3,:,:]=img2[0,:,:]
        a[4,:,:]=img3[0,:,:]

        # Convert image and label to torch tensors
        a = torch.from_numpy(np.asarray(a))
        label = torch.from_numpy(np.asarray(self.__ls[index]))

        return a,  label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 64, shuffle = True, num_workers=4)

model = models.vgg16_bn(pretrained=True)
first_conv_layer = [nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1)]
first_conv_layer.extend(list(model.features))  
model.features= nn.Sequential(*first_conv_layer ) 
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=2)

freeze_layer(model)
for param in model.features[0].parameters():
    param.requires_grad = True
for param in model.classifier[6].parameters():
    param.requires_grad = True
for param in model.classifier[3].parameters():
    param.requires_grad = True
for param in model.classifier[0].parameters():
    param.requires_grad = True

model.cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epoch = 45
best_acc = 0
train_loss = []
for epoch in range(num_epoch):
    correct = 0
    l = []
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):        
        inputs = inp_data[0].cuda()
        inputs = inputs.float()
        target = inp_data[1].cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, target)
       
        if batch_idx%100==0 or batch_idx==len(dsets)/64:
            l.append(loss.data)

        correct += torch.sum(preds == target.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        plt.plot(np.arange(len(train_loss)), train_loss)
        plt.ylim(0, 5)
        plt.savefig('train_curve_gcae_classifier.jpg')
        plt.close()
    acc= (correct.double()/len(dsets))*100
    torch.save(model.module.state_dict(), 'model_gcae_classifier.pth')
    print('loss=', l)	
    print('{} Accuracy: {:.4f}'.format(epoch+1, acc))
