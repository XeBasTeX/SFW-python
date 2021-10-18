# from PIL import Image
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import tifffile

import geomloss as gloss

import time

from skimage.measure import label
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = tifffile.imread('data/sequence-as-stack-MT0.N1.LD-2D-Exp.tif')

im = im.astype(np.float16)

pil_to_tensor = torchvision.transforms.ToTensor()

im_tensor = torch.tensor(im).float()

#%%

im_select = im_tensor[9, :, :]
im_select = (im_select-im_select.min())/(im_select.max()-im_select.min())

plt.imshow(im_select)

nx, ny = 64, 64
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)

[X, Y] = np.meshgrid(x, y)

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

v_x = torch.stack((Y_tensor, X_tensor), dim=2).view(-1, 2).float()

#%%
factor = 5

start_time = time.time()
alpha = torch.exp(factor*im_select)/(torch.exp(factor*im_select).sum())
alphaf = alpha.view(-1, 1).squeeze().float()

loss = gloss.SamplesLoss("sinkhorn", p=2, blur=0.001, scaling=.9,
                         backend="tensorized")

Niter = 50

delta_x = torch.randn(3, 2).float()
delta_x.requires_grad = True

uniform_delta_distribution = torch.ones([delta_x.shape[0]]).float()
uniform_delta_distribution = uniform_delta_distribution / \
    uniform_delta_distribution.sum()

dt = 0.1

indic_nz = alphaf > 2*alphaf.mean()

v_xsamp = v_x[indic_nz, :]
alphafsamp = alphaf[indic_nz]/alphaf[indic_nz].sum()

print(time.time() - start_time)

for i in range(Niter):
    cost = loss(uniform_delta_distribution, delta_x, alphafsamp, v_xsamp)
    [g_x] = torch.autograd.grad(cost, [delta_x])

    delta_x.data = delta_x.data - dt*g_x

    if (i+1) % 10 == 0:
        plt.figure(1)
        plt.imshow(alpha)
        plt.scatter(nx*(delta_x[:, 1].detach()+1)/2,
                    ny*(delta_x[:, 0].detach()+1)/2, c='r')
        plt.pause(0.01)

    print(i + 1)
    print(cost)


#%%

plt.figure(1)
plt.imshow(alpha)
plt.scatter(nx*(delta_x[:, 1].detach()+1)/2, ny*(delta_x[:, 0].detach()+1)/2,
            c='r')

#%% iterating over images

L_output = []

loss = gloss.SamplesLoss("sinkhorn", p=1, blur=0.001, scaling=.6,
                         backend="tensorized")
Niter = 100

max_points_per_frame = 8

for image_number in range(4000):
    im_select = im_tensor[image_number, :, :]
    im_select = (im_select-im_select.min())/(im_select.max()-im_select.min())

    alpha = torch.exp(factor*im_select)/(torch.exp(factor*im_select).sum())
    alphaf = alpha.view(-1, 1).squeeze().float()

    indic_nz = alphaf > 4*alphaf.mean()

    ncc = label(indic_nz.view(nx, ny)).max()
    print(ncc)

    n_dirac = min(ncc, max_points_per_frame)

    delta_x = torch.randn(n_dirac, 2).float()
    delta_x.requires_grad = True

    uniform_delta_distribution = torch.ones([delta_x.shape[0]]).float()/n_dirac

    v_xsamp = v_x[indic_nz, :]
    alphafsamp = alphaf[indic_nz]/alphaf[indic_nz].sum()

    if ncc <= max_points_per_frame:
        for i in range(Niter):
            cost = loss(uniform_delta_distribution,
                        delta_x, alphafsamp, v_xsamp)
            [g_x] = torch.autograd.grad(cost, [delta_x])

            delta_x.data = delta_x.data - dt*g_x

        print('image {} OK, W-loss: {}'.format(image_number+1, cost.data))
        L_output.append(delta_x.detach())


#%%
point_stack = torch.vstack(L_output)

plt.figure(1)
plt.imshow(im_tensor.sum(dim=0))

# plt.figure(2)
plt.scatter(nx*(point_stack[:, 1]+1)/2, ny*(point_stack[:, 0]+1)/2, c='r',
            marker='.', s=5)
plt.xlim((0, nx))
plt.ylim((0, ny))

#%%
sr = 2

image_sr = torch.zeros(sr*nx, sr*ny)

for index in range(point_stack.shape[0]):
    if point_stack[index, 0] < 1 and point_stack[index, 0] > -1 and point_stack[index, 1] > -1 and point_stack[index, 1] < 1:
        image_sr[torch.floor(sr*nx*(point_stack[index, 0]+1)/2).long(),
                 torch.floor(sr*ny*(point_stack[index, 1]+1)/2).long()] += 1


plt.figure()
plt.imshow(image_sr)
