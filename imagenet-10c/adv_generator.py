from __future__ import print_function, division
from tqdm import tqdm
import numpy
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
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pdb
import csv

plt.ion()
PATH = '/home/shihong/imagenet-10/val_s'
res_dir = 'imagenet_adv' 
def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

class Data(Dataset):
    def __init__(self,  transform):
        self.__xs = []
        self.__ys = []

        self.transform = transform
        with open("input_val_gcae3.csv", 'r+') as csvfile:
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

        return img1, label, PATH + self.__xs[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(), normalize])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])
dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 25, shuffle = True, num_workers=4)


model = models.vgg16_bn(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
model.load_state_dict(torch.load('./model/model_imageclass10bn.pth'))

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
    image0 = image.cpu().data
    perturbed_image0 = perturbed_image.cpu().data
    for xi in range(25):
        save_path = os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
        create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
        utils.save_image(inv_normalize(image0[xi,:,:,:]), save_path + '_orig.png', padding=0)
        utils.save_image(inv_normalize(perturbed_image0[xi,:,:,:]), save_path + '_perturb.png', padding=0)

    for i in range(len(perturbed_image)):
        perturbed_image[i] = inv_normalize(perturbed_image[i])
        perturbed_image[i] = perturbed_image[i].mul_(255).round_().clamp_(0, 255).div_(255)
        perturbed_image[i] = normalize(perturbed_image[i])
    return perturbed_image

def attack(model, optimizer, epsilon, criterion):
    correct_before = 0
    correct_after = 0

    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        
        inputs = inp_data[0].cuda()
        
        inputs.requires_grad=True
        
        paths = inp_data[2]
        target = inp_data[1].to('cuda')

        outputs_before = model(inputs)
        _, preds_before = torch.max(outputs_before, 1)
        correct_before += torch.sum(preds_before == target.data)
        loss_before = criterion(outputs_before, target)
        
        optimizer.zero_grad()
        loss_before.backward()
        
        data_grad = inputs.grad.data

        perturbed_data = fgsm_attack(inputs, epsilon, data_grad, paths)
        outputs_after = model(perturbed_data.to('cuda'))
        _, preds_after = torch.max(outputs_after, 1)
        correct_after += torch.sum(preds_after == target.data)
    
    final_acc_before = correct_before.double()/len(dsets)
    final_acc_after = correct_after.double()/len(dsets)
    print('{} Acc_before: {:.4f}'.format(epsilon, final_acc_before))
    print('{} Acc_after: {:.4f}'.format(epsilon, final_acc_after))

a = attack(model, optimizer, 0.03, criterion)