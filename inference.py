import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import numpy as np

import libs as lib
import libs.plot

from wgan import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
from PIL import Image
import math
from shutil import rmtree
import torch.nn.init as init

import config as CONFIG

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('Run on device = ', device)

PATH_TO_DISC_CKPT = 'checkpoints_gaussian/discriminator_31799.pt'
PATH_TO_GEN_CKPT = 'checkpoints_gaussian/generator_31799.pt'
NUM_IMAGES = 10000
BATCH_SIZE = 100
TOLERATE_TH = 1 # Critic of ganerated image must be larger than mean on real + the tolerate thresholh
GEN_DIR = 'generated_10kimages'
PATH_TO_NOISE_VEC = 'noise.npy'

if os.path.exists(GEN_DIR):
    rmtree(GEN_DIR)
os.makedirs(GEN_DIR)

def load_data(path_to_folder):
    data_transform = transforms.Compose([
                 transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), interpolation=Image.BICUBIC), # Possible values are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC and Image.ANTIALIAS
                 #transforms.CenterCrop(CONFIG.IMAGE_SIZE),
                 #transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])
    
    dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=5, drop_last=True, pin_memory=True)
    return dataset_loader

VAL_DIR_PATH = '/media/HDD/SangTV/Zalo_AI_challenge/training_dataset/validationset'
val_loader = load_data(VAL_DIR_PATH)

def gen_rand_noise():
    noise = torch.randn(BATCH_SIZE, CONFIG.NOISE_DIM)
    noise = noise.to(device)

    return noise


def generate_image(netG, netD, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    samples = netG(noise)
    disc_values = netD(samples)
    disc_values = disc_values.view(BATCH_SIZE, -1)
    samples = samples.view(BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    samples = samples * 0.5 + 0.5
    return samples, disc_values, noise

aG = torch.load(PATH_TO_GEN_CKPT)
print('Restore Generator done')
aD = torch.load(PATH_TO_DISC_CKPT)
print('Restore Discriminator done')

aG.eval()
aD.eval()

"""
Measure D_th is incorrect now ==> fix bug later
"""

with torch.no_grad():
    """
    # Measure the evaluation of Discriminator on Validation set
    dev_disc_costs = []
    for _, images in enumerate(val_loader):
        imgs = torch.Tensor(images[0])
        imgs = imgs.to(device)
        with torch.no_grad():
            imgs_v = imgs

        D = aD(imgs_v)
        _dev_disc_cost = D.mean().cpu().data.numpy()
        dev_disc_costs.append(_dev_disc_cost)

    Disc_th = np.mean(dev_disc_costs)
    print('Mean on Real = {} | Tol_th = {} ==> Threshold = {}'.format(Disc_th, TOLERATE_TH, Disc_th + TOLERATE_TH))
    Disc_th += TOLERATE_TH
    """
    Disc_th = 0

    i = 0
    enough_image = False
    noises = []
    while True:
        print(i, end='\r')
        gen_images, disc_values, noise = generate_image(aG, aD)
        for j in range(BATCH_SIZE):
            disc_val = disc_values[j].cpu().data.numpy()
            disval = disc_val[0]
            if disval >= Disc_th:
                i += 1
                save_name = '{}_{}'.format(i, disval)
                save_name = save_name.replace('.', '_') + '.png'
                torchvision.utils.save_image(gen_images[j, :, :, :], os.path.join(GEN_DIR, save_name))
                noises.append(noise[j].cpu().data.numpy())
                if i == NUM_IMAGES:
                    enough_image = True
                    break

        if enough_image is True:
            break

np.save(PATH_TO_NOISE_VEC, np.asarray(noises))

