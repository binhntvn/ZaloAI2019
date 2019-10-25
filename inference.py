import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

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

PATH_TO_CKPT = 'checkpoints/generator.pt'
NUM_IMAGES = 10000
BATCH_SIZE = 32
GEN_DIR = 'generated_10kimages'

if os.path.exists(GEN_DIR):
    rmtree(GEN_DIR)
os.makedirs(GEN_DIR)

def gen_rand_noise():
    noise = torch.randn(BATCH_SIZE, CONFIG.NOISE_DIM)
    noise = noise.to(device)

    return noise


def generate_image(netG, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
        noisev = noise 
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    samples = samples * 0.5 + 0.5
    return samples

aG = torch.load(PATH_TO_CKPT)

for i in range(int(math.ceil(NUM_IMAGES/BATCH_SIZE))):
    gen_images = generate_image(aG)
    for j in range(BATCH_SIZE):
        torchvision.utils.save_image(gen_images[j, :, :, :], os.path.join(GEN_DIR, '{}.png'.format(i * BATCH_SIZE + j)))

