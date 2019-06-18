from __future__ import print_function, division
from tqdm import tqdm
import numpy
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

#dataset-loader
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])
trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])

class Data(Dataset):
    __xs = [] #input image
    __ls = [] #target-label
    __ps = [] #classifier-label
    def __init__(self, transform):
        self.transform = transform
        with open("/mnt/ssd/suchetha/classifier/input_val_gcae1.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0]) # original/adv image
                if "orig_orig" in row[0]:
                    #self.__ls.append(0)
                    continue
                else: #perturb
                    #self.__ls.append(1)
                    self.__ls.append(0)
                if "n01440764" in row[0]:
                    self.__ps.append(0)
                elif "n01689811" in row[0]:
                    self.__ps.append(1)
                elif "n01806143" in row[0]:
                    self.__ps.append(2)
                elif "n02138441" in row[0]:
                    self.__ps.append(3)
                elif "n02490219" in row[0]:
                    self.__ps.append(4)
                elif "n03841143" in row[0]:
                    self.__ps.append(5)
                elif "n03866082" in row[0]:
                    self.__ps.append(6)
                elif "n04154565" in row[0]:
                    self.__ps.append(7)
                elif "n04507155" in row[0]:
                    self.__ps.append(8)
                else:
                    self.__ps.append(9) 


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1 = Image.open(self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        img1 = torch.from_numpy(np.asarray(img1))

        label_target = torch.from_numpy(np.asarray(self.__ls[index]))
        label_classifier = torch.from_numpy(np.asarray(self.__ps[index]))
        return img1, self.__xs[index], label_target, label_classifier

    def __len__(self):
        return len(self.__xs)

dsets = Data(transform=trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 10, shuffle = False, num_workers=4)

#classifier-part

classif = models.vgg16_bn(pretrained=True)
num_ftrs = classif.classifier[6].in_features
classif.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
classif.cuda()
classif.load_state_dict(torch.load('model_imageclass10bn.pth'))

#autoencoder-part

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
        return x

ae = Autoencoder()
criterion = nn.MSELoss(reduction='sum')
ae.to('cuda')
ae.load_state_dict(torch.load('ae_gradcam_1.pth'))
ae.eval()

#detector-part

model = models.vgg16_bn(pretrained=True)
first_conv_layer = [nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)]
first_conv_layer.extend(list(model.features))
model.features= nn.Sequential(*first_conv_layer )
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=2)
model.cuda()
model.load_state_dict(torch.load('model_ae_classifier.pth'))
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
model.eval()

#criterion = nn.MSELoss(reduction='sum')
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epoch = 1
for i in range(num_epoch):
    correct = 0
    correct_after = 0
    for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):
        inputs = inp_data[0].cuda()
        inputs.requires_grad=True
        inputs2 = inputs.clone()
        paths = inp_data[1]
        target = inp_data[2].cuda()#label(0 or 1)
        classifier_label = inp_data[3].cuda()
        ae_mask=ae(inputs)
        #a = np.zeros(shape=(10,4,224,224))
        #a[0,:,:]=inputs[:,0,:,:]
        #a[1,:,:]=inputs[:,1,:,:]
        #a[2,:,:]=inputs[:,2,:,:]
        #a[3,:,:]=ae_mask[:,0,:,:]
        a = torch.cat((inputs,ae_mask),1)
        outputs = model(a)
        _, preds = torch.max(outputs, 1)
        with torch.no_grad():
            class_outputs = classif(inputs)
            _, class_preds = torch.max(class_outputs, 1)
            correct += torch.sum(class_preds == classifier_label.data)
            acc = (correct.double()/((batch_idx+1)*len(inputs)))*100
            print('Accuracy before: {:.4f}'.format(acc))
        idx = (preds == 1).nonzero()#.type(torch.LongTensor)
        idx = idx.view(-1)
        for i in idx:
            i = i.item()
            loss = criterion(outputs[i:i+1,:], target[i:i+1])
            
            #optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #print(torch.sum(inputs.grad,dim=0).shape)
            data_grad = inputs.grad.data[i,:,:,:]
            inputs2[i,:,:,:] = inputs[i,:,:,:]-0.03*data_grad.sign()
            inputs2[i,:,:,:] = torch.clamp(inputs2[i,:,:,:],0,1)
            #optimizer.step()
        #loss.backward() 
        with torch.no_grad():
            ae_mask_after=ae(inputs2)
            a_new = torch.cat((inputs2,ae_mask_after),1)
            outputs_after = model(a_new)
            _, preds_after = torch.max(outputs_after, 1)
            class_outputs_after = classif(inputs2)
            _, class_preds_after = torch.max(class_outputs_after, 1)
            correct_after += torch.sum(class_preds_after == classifier_label.data)
            acc_after = (correct_after.double()/((batch_idx+1)*len(inputs)))*100
            print('Accuracy after: {:.4f}'.format(acc_after))
    #correct += torch.sum(preds == target.data)
    #acc = (correct.double()/len(dsets))*100
    #print('Accuracy: {:.4f}'.format(acc))
