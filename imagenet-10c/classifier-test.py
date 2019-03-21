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

PATH = '/home/shihong/imagenet-10/'
class Data(Dataset):
    def __init__(self,  transform):
        self.__xs = []
        self.__ys = []

        self.transform = transform
        with open("imageval.csv", 'r+') as csvfile:
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 128, shuffle = True, num_workers=4)

model = models.vgg16_bn(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
model.cuda()
model.load_state_dict(torch.load('./model/model_imageclass10bn.pth'))
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
best_acc = 0
model.eval()
correct = 0
l = []
with torch.no_grad():
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):        
        inputs = inp_data[0].cuda()
        target = inp_data[1].cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, target)
        correct += torch.sum(preds == target.data)
    acc = (correct.double()/len(dsets))*100
	
print('Accuracy: {:.4f}'.format(acc))