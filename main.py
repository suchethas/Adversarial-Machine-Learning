from __future__ import print_function
from PIL import Image
from tqdm import tqdm
import copy
import os.path as osp
import os
import csv
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pdb
import torch.nn as nn
from grad_cam import (
    # BackPropagation,
    # Deconvnet,
    GradCAM,
    # GuidedBackPropagation,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])

def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("samples/synset_words1.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename1, filename2, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy() * 255
    cv2.imwrite(filename1, np.uint8(gcam))
    cv2.imwrite(filename2, raw_image)


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

class Data(Dataset):
    __xs = []

    def __init__(self):
        with open("/newhd/suchetha/suchetha/classifier/input_train_naive.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img1, raw_img1 = preprocess(self.__xs[index])
        return img1, raw_img1, self.__xs[index]

    def __len__(self):
        return len(self.__xs)

@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
#@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)

def demo1(target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=num_ftrs,out_features=10)
    model.load_state_dict(torch.load('model_imageclass10bn.pth'))
    #print(model)
    model.to(device)
    model.eval()
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #trans = transforms.Compose([transforms.ToTensor(), normalize])
    dsets = Data()
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 10, shuffle = False, num_workers=2)
    # =========================================================================
    
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
    #bp = BackPropagation(model=model)
    for batch, inp in enumerate(tqdm(dset_loaders),1):
        #print(images)
        i = inp[0]
        raw_images = inp[1].numpy()
        #print(raw_images)
        paths = inp[2]
        images = i.cuda()
        #raw_images = r.cuda()
        #probs, ids = bp.forward(images)
        gcam = GradCAM(model=model)
        probs, ids = gcam.forward(images)
        #_ = gcam.forward(images)

        #gbp = GuidedBackPropagation(model=model)
        #_ = gbp.forward(images)

        for i in range(topk):
            # Guided Backpropagation
            #gbp.backward(ids=ids[:, [i]])
            #gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=ids[:, [i]], image=images)
            regions = gcam.generate(target_layer=target_layer)

            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
                # Grad-CAM
                save_path = os.path.join(output_dir, paths[j].split('/')[6],paths[j].split('/')[7].split(".")[0])
                #save_path = os.path.join(output_dir, paths[j].split('/')[6],paths[j].split('/')[7].split(".")[0])
                create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
                save_gradcam(
                    filename1=
                        save_path+"gradcam.png".format(classes[ids[j, i]]
                        ),
                    filename2=save_path+".png",
                    gcam=regions[j, 0],
                    raw_image=raw_images[j],
                )
        gcam.remove_hook()
        gcam.model.zero_grad()
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
