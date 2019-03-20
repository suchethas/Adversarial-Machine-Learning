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

PATH = '/home/shihong/imagenet-10/'
class Data(Dataset):
    def __init__(self,  transform):
        self.__xs = []
        self.__ys = []

        self.transform = transform
        with open("image.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])
                if "n01440764" in row[0]:
                    self.__ys.append(0)
                elif "n01689811" in row[0]:
                    self.__ys.append(1)
                elif "n01806143" in row[0]:
                    self.__ys.append(2)
                elif "n02138441" in row[0]:
                    self.__ys.append(3)
                elif "n02490219" in row[0]:
                    self.__ys.append(4)
                elif "n03841143" in row[0]:
                    self.__ys.append(5)
                elif "n03866082" in row[0]:
                    self.__ys.append(6)
                elif "n04154565" in row[0]:
                    self.__ys.append(7)
                elif "n04507155" in row[0]:
                    self.__ys.append(8)
                else:
                    self.__ys.append(9)

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1 = Image.open(PATH + self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        # Convert image and label to torch tensors
        img1 = torch.from_numpy(np.asarray(img1))
        label = torch.from_numpy(np.asarray(self.__ys[index]))

        return img1,  label

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
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)

# freeze all layers except the last layer
freeze_layer(model)
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
# optimizer = optim.Adam(model.parameters())
num_epoch = 20
best_acc = 0
train_loss = []
for epoch in range(num_epoch):
    correct = 0
    l = []
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):        
        inputs = inp_data[0].cuda()
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
        plt.savefig('./train_curve_classifier.jpg')
        plt.close()
    acc= (correct.double()/len(dsets))*100
    torch.save(model.module.state_dict(), './model/model_imageclass10bn.pth')
    print('loss=', l)	
    print('{} Accuracy: {:.4f}'.format(epoch+1, acc))