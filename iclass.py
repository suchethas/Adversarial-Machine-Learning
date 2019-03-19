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
import data_loader as dl
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv

class DriveData(Dataset):
    __xs = []
    __ys = []
    #__path = []

    def __init__(self,  transform):
        self.transform = transform
        with open("image.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1 = Image.open(self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        # Convert image and label to torch tensors
        img1 = torch.from_numpy(np.asarray(img1))

        return img1,  self.__xs[index]#self.__path

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
dsets = DriveData(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 20, shuffle = True, num_workers=4)

#dsets = dl.ImageFolder('/newhd/suchetha/suchetha/train1',transform=trans)
#dset_loader = torch.utils.data.DataLoader(dsets,batch_size=10,num_workers=2,shuffle=False)

model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
model.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epoch = 20
best_acc = 0
for epoch in range(num_epoch):
    correct = 0
    l = []
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        #print(epoch, batch_idx)
        inputs = inp_data[0].cuda()
        #inpu = inputs.to(device)
        #print(inputs)
        paths = inp_data[1]
        target = []
        for i in range(len(paths)):
            c = paths[i].split("/")[5]
            if c=="n01440764":
                target.append(0)
            elif c=="n01689811":
                target.append(1)
            elif c == "n01806143":
                target.append(2)
            elif c== "n02138441":
                target.append(3)
            elif c == "n02490219":
                target.append(4)
            elif c == "n03841143":
                target.append(5)
            elif c == "n03866082":
                target.append(6)
            elif c == "n04154565":
                target.append(7)
            elif c== "n04507155":
                target.append(8)
            else:
                target.append(9)

        #target.to_tensor().cuda()
        t=torch.FloatTensor(target)
        t=t.long().cuda()
        #print(t)
        outputs = model(inputs)
        #print(outputs)
        _, preds = torch.max(outputs, 1)
        #print(preds)
        #print(preds, t.data)

        loss = criterion(outputs, t)
        #print(loss)
        if batch_idx%100==0 or batch_idx==len(dsets)/20:
            l.append(loss.data)
        with torch.no_grad():
            correct += torch.sum(preds == t.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc= (correct.double()/len(dsets))*100
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '/newhd/suchetha/suchetha/model_imageclass10.pth')
        torch.save(optimizer.state_dict(),'/newhd/suchetha/suchetha/optim_class_10.pth')
    print('loss=', l)	
    print('{} Accuracy: {:.4f}'.format(epoch+1, acc))
