import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import time
import functools
import argparse

import numpy as np

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

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

import torch.nn.init as init

import config as CONFIG

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('Run on device = ', device)

if not os.path.exists(CONFIG.OUTPUT_PATH):
    os.makedirs(CONFIG.OUTPUT_PATH)


def weights_init(m):
    if isinstance(m, MyConv2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def load_data(path_to_folder):
    data_transform = transforms.Compose([
                 #transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), interpolation=Image.BICUBIC), # Possible values are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC and Image.ANTIALIAS
                #  transforms.CenterCrop(CONFIG.IMAGE_SIZE),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])
    
    dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=5, drop_last=True, pin_memory=True)
    return dataset_loader


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(CONFIG.BATCH_SIZE, 1)
    alpha = alpha.expand(CONFIG.BATCH_SIZE, int(real_data.nelement()/CONFIG.BATCH_SIZE)).contiguous()
    alpha = alpha.view(CONFIG.BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(CONFIG.BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * CONFIG.LAMBDA
    return gradient_penalty


def gen_rand_noise():
    noise = torch.randn(CONFIG.BATCH_SIZE, CONFIG.NOISE_DIM)
    noise = noise.to(device)

    return noise


def generate_image(netG, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
    	noisev = noise 
    samples = netG(noisev)
    samples = samples.view(CONFIG.BATCH_SIZE, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    samples = samples * 0.5 + 0.5
    return samples


fixed_noise = gen_rand_noise()


if CONFIG.RESTORE_MODE:
    aG = torch.load(os.path.join(CONFIG.OUTPUT_PATH, "generator_{}.pt".format(CONFIG.START_ITER)))
    aD = torch.load(os.path.join(CONFIG.OUTPUT_PATH, "discriminator_{}.pt".format(CONFIG.START_ITER)))
else:
    aG = GoodGenerator(CONFIG.DIM, CONFIG.OUTPUT_DIM)
    aD = GoodDiscriminator(CONFIG.IMAGE_SIZE)
    
    aG.apply(weights_init)
    aD.apply(weights_init)

print('Generator')
print(aG)

print('Discriminator')
print(aD)

optimizer_g = torch.optim.Adam(aG.parameters(), lr=CONFIG.LR, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=CONFIG.LR, betas=(0,0.9))
one = torch.tensor(1.0)
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)


writer = SummaryWriter()
#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    dataloader = load_data(CONFIG.TRN_DATA_DIR)
    dataiter = iter(dataloader)
    for iteration in range(CONFIG.START_ITER, CONFIG.END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        avg_gen_cost = []
        avg_dis_cost = []
        avg_w_dist = []

        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(CONFIG.GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()

            noise.requires_grad_(True)
            fake_data = aG(noise)

            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
            optimizer_g.step()
            avg_gen_cost.append(gen_cost.cpu().data.numpy())

        end = timer()
        print('---train G elapsed time: {}'.format(end - start))
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CONFIG.CRITIC_ITERS):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise()
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            end = timer(); print('---gen G elapsed time: {}'.format(end-start))
            start = timer()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading

            end = timer(); print('---load real imgs elapsed time: {}'.format(end-start))
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()

            avg_dis_cost.append(disc_cost.cpu().data.numpy())
            avg_w_dist.append(w_dist.cpu().data.numpy())
            
            #------------------VISUALIZATION----------
            if i == CONFIG.CRITIC_ITERS-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
                if iteration %200==199:
                    body_model = [i for i in aD.children()][0]
                    layer1 = body_model.conv
                    xyz = layer1.weight.data.clone()
                    tensor = xyz.cpu()
                    tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                    writer.add_image('D/conv1', tensors, iteration)

            end = timer(); print('---train D elapsed time: {}'.format(end-start))
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(os.path.join(CONFIG.OUTPUT_PATH, 'time'), time.time() - start_time)
        lib.plot.plot(os.path.join(CONFIG.OUTPUT_PATH, 'train_disc_cost'), np.mean(np.array(avg_dis_cost)))
        lib.plot.plot(os.path.join(CONFIG.OUTPUT_PATH, 'train_gen_cost'), np.mean(np.array(avg_gen_cost)))
        lib.plot.plot(os.path.join(CONFIG.OUTPUT_PATH, 'wasserstein_distance'), np.mean(np.array(avg_w_dist)))
        if iteration % 200 == 199:
            val_loader = load_data(CONFIG.VAL_DATA_DIR)
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
               	imgs = imgs.to(device)
                with torch.no_grad():
            	    imgs_v = imgs

                D = aD(imgs_v)
                _dev_disc_cost = D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(os.path.join(CONFIG.OUTPUT_PATH, 'dev_disc_cost'), np.mean(dev_disc_costs))
            lib.plot.flush()	
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, os.path.join(CONFIG.OUTPUT_PATH, 'samples_{}.png'.format(iteration)), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
	#----------------------Save model----------------------
            torch.save(aG, os.path.join(CONFIG.OUTPUT_PATH, "generator_{}.pt".format(iteration)))
            torch.save(aD, os.path.join(CONFIG.OUTPUT_PATH, "discriminator_{}.pt".format(iteration)))
        lib.plot.tick()

train()
