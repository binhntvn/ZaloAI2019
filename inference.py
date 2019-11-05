import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
sys.path.append('/media/HDD/SangTV/Zalo_AI_challenge/evaluation_script/')
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
from pre_trained_classifier import GetClassifyScore

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('Run on device = ', device)

PATH_TO_GEN_CKPT = 'checkpoints_1101/generator_14399.pt'
NUM_IMAGES = 10000
BATCH_SIZE = 16
TOLERATE_TH = 1 # Critic of ganerated image must be larger than mean on real + the tolerate thresholh
GEN_DIR = 'generated_10kimages'
PATH_TO_NOISE_VEC = 'noise.npy'
SCORE_TH = 0#0.95

Extractor = GetClassifyScore()

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

    samples = netG(noise)

    samples_cp = samples.clone()
    samples_cp = samples_cp * 0.5 + 0.5
    samples_cp = samples_cp.view(BATCH_SIZE, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3)
    score_arr, feature_arr = Extractor.get_scores(samples_cp.cpu().data.numpy(), BATCH_SIZE)

    samples = samples.view(BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    samples = samples * 0.5 + 0.5

    return samples, noise, score_arr.flatten()

aG = torch.load(PATH_TO_GEN_CKPT)
print('Restore Generator done')

aG.eval()

with torch.no_grad():
    i = 0
    enough_image = False
    noises = []

    while True:
        print(i, end='\r')

        gen_images, noise, scores = generate_image(aG)
        # print(scores)

        for j in range(BATCH_SIZE):
            src = scores[j]
            if src >= SCORE_TH:
                i += 1
                save_name = '{}_{}'.format(i, src)
                save_name = save_name.replace('.', '_') + '.png'
                torchvision.utils.save_image(gen_images[j, :, :, :], os.path.join(GEN_DIR, save_name))
                noises.append(noise[j].cpu().data.numpy())
                if i == NUM_IMAGES:
                    enough_image = True
                    break

        if enough_image is True:
            break

np.save(PATH_TO_NOISE_VEC, np.asarray(noises))

