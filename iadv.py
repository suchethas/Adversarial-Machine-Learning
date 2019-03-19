from __future__ import print_function, division
from tqdm import tqdm
import numpy
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
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import csv
plt.ion()

res_dir = 'imagenet_adv' 
def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

class DriveData(Dataset):
    __xs = []
    #__ys = []
    #__path = []

    def __init__(self,  transform):
        self.transform = transform
        with open("advimg.csv", 'r+') as csvfile:
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
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 25, shuffle = True, num_workers=4)

#dsets = dl.ImageFolder(data_dir,transform=trans)
#dset_loader = torch.utils.data.DataLoader(dsets,batch_size=25,num_workers=2,shuffle=True)

model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
model.load_state_dict(torch.load('model_imageclass10.pth'))

def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False

freeze_layer(model)

model.cuda()
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def fgsm_attack(image, epsilon, data_grad, paths):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad

    for xi in range(25):
        save_path = os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
        create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
        utils.save_image(image[xi,:,:,:], save_path + '_orig.png',  normalize=True, padding=0)
        utils.save_image(perturbed_image[xi,:,:,:], save_path + '_perturb.png',  normalize=True, padding=0)
    return perturbed_image

def attack(model, optimizer, epsilon, criterion):
    correct_before = 0
    correct_after = 0

    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        #print(epoch, batch_idx)
        inputs = inp_data[0].cuda()
        #inputs.cuda()
        inputs.requires_grad=True
        #inputs.cuda()
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
        outputs_before = model(inputs)
        _, preds_before = torch.max(outputs_before, 1)
        correct_before += torch.sum(preds_before == t.data)
        loss_before = criterion(outputs_before, t)
        
        optimizer.zero_grad()
        loss_before.backward()
        
        data_grad = inputs.grad.data

        perturbed_data = fgsm_attack(inputs, epsilon, data_grad, paths)
        outputs_after = model(perturbed_data)
        _, preds_after = torch.max(outputs_after, 1)
        correct_after += torch.sum(preds_after == t.data)
    
    final_acc_before = correct_before.double()/len(dsets)
    final_acc_after = correct_after.double()/len(dsets)
    print('{} Acc_before: {:.4f}'.format(epsilon, final_acc_before))
    print('{} Acc_after: {:.4f}'.format(epsilon, final_acc_after))

a = attack(model, optimizer, 0.01, criterion)
