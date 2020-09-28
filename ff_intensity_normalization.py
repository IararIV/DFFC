#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:28:36 2020

@author: Gerard Jover Pujol

Dynamic intensity normalization usingeigen flat fields in X-ray imaging
"""

# reading i23 data
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.transform import rescale, resize, downscale_local_mean
import random
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR

vert_tuple = [i for i in range(500,1100)] # selection of vertical slice

h5py_list = h5py.File('/dls/science/users/lqg38422/data/13282/13282.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:40,vert_tuple,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][40:80,vert_tuple,:]

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][80:1881,vert_tuple,:]
angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
angles_rad = angles[80:1881]*np.pi/180.0

h5py_list.close()

#%% Image + DF + FF display

n_im = random.randrange(0, 40)
print(f"Image number {n_im}")

fig= plt.figure()
plt.rcParams.update({'font.size': 12})
plt.subplot(221)
plt.imshow(flats[n_im,:,:], vmin=250, vmax=47000, cmap="gray")
plt.title('Flat field image')
plt.subplot(222)
plt.imshow(darks[n_im,:,:], vmin=0, vmax=1000, cmap="gray")
plt.title('Dark field image')
plt.show()
plt.subplot(223)
plt.imshow(data_raw[n_im,:,:], cmap="gray")
plt.title('Raw data image')
plt.show()

#%% Traditional method

mean_d = np.mean(darks, axis=0)
mean_f = np.mean(flats, axis=0)

n_im = random.randrange(0, 40)
output = (data_raw[n_im] - mean_d) / (mean_f - mean_d)

fig= plt.figure()
plt.rcParams.update({'font.size': 12})
plt.subplot(121)
plt.title(f"Image number {n_im}")
plt.imshow(data_raw[n_im], cmap='gray')
plt.subplot(122)
plt.title("Image reconstruction")
plt.imshow(output, cmap='gray')

#%% Dynamic Flat Field Correction (Matlab translation)
# =============================================================================
# NOTES:
# · Why I only get 1 EEF?
# =============================================================================

# Load frames
meanDarkfield = np.mean(darks, axis=0)

whiteVect = np.zeros((flats.shape[0], flats.shape[1]*flats.shape[2]))
k = 0
for ff in flats:
    whiteVect[k] = ff.flatten() - meanDarkfield.flatten()

mn = np.mean(whiteVect, axis=0)
    
# Substract mean flat field
M, N = whiteVect.shape
Data = whiteVect - mn

# =============================================================================
# EEFs selection - Parallel Analysis
#      Selection of the number of components for PCA using parallel Analysis.
#      Each flat field is a single row of the matrix flatFields, different
#      rows are different observations.
# =============================================================================

def parallelAnalysis(flatFields, repetitions):
    stdEFF = np.std(flatFields, axis=0)
    H, W = flatFields.shape
    keepTrack = np.zeros((H, repetitions))
    stdMatrix = np.tile(stdEFF, (H, 1))
    for i in range(repetitions):
        print(f"Parallel Analysis - repetition {i}")
        sample = stdMatrix * np.random.rand(H, W)
        D1, _ = np.linalg.eig(np.cov(sample))
        keepTrack[:,i] = D1
    mean_flat_fields_EFF = np.mean(flatFields, axis=0)
    F = flatFields - mean_flat_fields_EFF
    D1, V1 = np.linalg.eig(np.cov(F))
    
    selection = np.zeros((1,H))
    selection[:,D1>(np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1))] = 1
    numberPC = np.sum(selection)
    return V1, D1, int(numberPC)

# Parallel Analysis
nrPArepetions = 10
print("Parallel Analysis:")
V1, D1, nrEigenflatfields = parallelAnalysis(Data,nrPArepetions)
print(f"{nrEigenflatfields} eigen flat fields selected!")

# calculation eigen flat fields
C, H, W = data_raw.shape
eig0 = mn.reshape((H,W))
EFF = np.zeros((nrEigenflatfields, H, W))
print("Calculating EFFs:")
for i in range(nrEigenflatfields):
    EFF[i] = (np.matmul(Data.T, V1[M-i-1]).T).reshape((H,W))
print("Done!")
# works

# Filter EFFs (we skip this step since we want to try DL)

# =============================================================================
# Normalize function: intensity values between [0, 1]
# =============================================================================

def normalize(im):
    return (im - im.min()) / (im.max() - im.min())

# Estimate abundance of weights in projections
# =============================================================================
# Conventional FFC
# out_path = "./outDIRDFFC/"
# n_im = len(data_raw)
# meanVector = np.zeros((1, n_im))
# for i in range(n_im):
#     projection = data_raw[i]
#     tmp = (projection - meanDarkfield)/EFF[0]
#     meanVector[i] = np.mean(tmp)
#     
#     tmp[tmp<0] = 0
#     tmp = -np.log(tmp)
#     tmp[np.isinf(tmp)] = 10^5
#     tmp = normalize(tmp)
#     tmp = np.uint16((2^16-1)*tmp)
#     tiff.imsave(f'{out_path}out_{i}.tiff', tmp)
# =============================================================================

# =============================================================================
# Function cost: objective function to optimize weights
# =============================================================================

def cost_func(projections, meanFF, FF, DF, x):
    FF_eff = np.zeros((FF.shape[1], FF.shape[2]))
    for i in range(len(FF)):
        FF_eff  = FF_eff + x[i] * FF[i]
    logCorProj=(projections-DF)/(meanFF+FF_eff)*np.mean(meanFF.flatten()+FF_eff.flatten());
    Gx, Gy = np.gradient(logCorProj)
    mag = (Gx**2 + Gy**2)**(1/2)
    cost = np.sum(mag.flatten())
    return cost
    
# =============================================================================
# CondTVmean function: finds the optimal estimates  of the coefficients of the 
# eigen flat fields.
# =============================================================================

def condTVmean(projections,meanFF,FF,DF,x,DS):
    # Downsample image
    projections = downscale_local_mean(projections, (DS, DS))
    meanFF = downscale_local_mean(meanFF, (DS, DS))
    FF2 = np.zeros((FF.shape[0], meanFF.shape[1], meanFF.shape[2]))
    for i in range(len(FF)):
        FF2[i] = downscale_local_mean(FF[i], (DS, DS))
    FF = FF2
    DF = downscale_local_mean(DF, (DS, DS))
    
    # Optimize coefficients
    cost = cost_func(projections, meanFF, FF, DF, x)
    print(f"Cost: {cost}")
    # optimization function
    
out_path = './outDIRDFFC/'
n_im = len(data_raw)
xArray = np.zeros((nrEigenflatfields, n_im))
downsample = 20
for i in range(n_im):
    print("Estimation projection:")
    projection = data_raw[i]
    # Estimate weights for a single projection
    weights = np.zeros(nrEigenflatfields)
    x=condTVmean(projection, EFF[0],\
                     EFF[1:nrEigenflatfields],\
                     meanDarkfield,weights,downsample)
    xArray[:,i]=x
    
    # Dynamic FFC
    FFeff = np.zeros(meanDarkfield.shape)
    for j in range(nrEigenflatfields):
        FFeff = FFeff + x[j] * EFF[j+1]
        
    tmp = (projection - meanDarkfield)/(EFF[0] + FFeff)
    tmp[tmp<0] = 0
    tmp = -np.log(tmp)
    tmp[np.isinf(tmp)] = 10^5
    tmp = normalize(tmp)
    tmp = np.uint16((2^16-1)*tmp)
    tiff.imsave(f'{out_path}out_{i}.tiff', tmp)
    
    
    


#%% Dynamic Flat Field Correction

def normalize(im):
    return (im - im.min()) / (im.max() - im.min())

# def vector_matrix(im):
#     c, h, w = im.shape
#     M = np.zeros((h*w, c))
#     for n,frame in enumerate(im):
#         M[:,n] = normalize(frame.flatten())
    return M

def flatten(im):
    return np.expand_dims(im.flatten(), axis=1)

data_raw = data_raw[:100]

mean_d = normalize(np.mean(darks, axis=0))
mean_f = normalize(np.mean(flats, axis=0))
flat_mean_d = flatten(mean_d)
flat_mean_f = flatten(mean_f)


# centered flat field matrix (columns: ff vectors)
M = vector_matrix(flats)

# centered FF matrix
A = M - flat_mean_f

# covariance matrix 
C = np.matmul(A.T, A)

# eigenvalues and eigenvectors
eigvals, eigvects = np.linalg.eig(C)

#%% Down-scale images
# =============================================================================
# The dynamic flat field weights{wjk}were estimated on 20 times down sampled 
# projections and afterwards applied to the full scaleprojections. Due to the 
# limited amount of truncation the dynamic FFC projections were rescaledusing the 
# first Helgason-Ludwig consistency condition
# =============================================================================

# Option 1: Resize
image = data_raw[0]
image_downscaled = downscale_local_mean(image, (10, 10))

#%% EFFs selection - variation percentil
variation = np.cumsum(eigvals)/np.cumsum(eigvals).max()
plt.plot(variation)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

p = np.percentile(variation, 90)
idx = np.argwhere(variation >= p)[0][0]
print('90% percentile:', p)
print('Idx:', idx)

EFF = np.matmul(A, eigvects)
uk = EFF[:,idx].reshape((800,1199))

# filtering
# we skip this since we're using DL

#%% Estimate (Gerard implementation)

# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# torch.manual_seed(1)

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__() #or super(Network, super).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
#         self.out = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5, padding=2)
        
#     def forward(self, t):
                
#         # (1) input layer
#         t = t
        
#         # (2) hidden conv layer
#         t = self.conv1(t)
#         t = F.relu(t)

#         # (3) hidden conv layer
#         t = self.conv2(t)
#         t = F.relu(t)
        
#         # (4) hidden conv layer
#         t = self.conv3(t)
#         t = F.relu(t)
        
#         # (4) output layer
#         t = self.out(t)
#         # t = F.softmax(t, dim=1)

#         return t
    
# def cost_function(projection, mean_f, mean_d, weights, uk):
#     c_w = torch.mean(mean_f + weights * uk - mean_d)
#     nj = (projection - mean_d)/(mean_f + weights * uk - mean_d)
#     grad = torch.tensor(np.gradient(nj.cpu().detach().numpy())).to('cuda')
#     magn = (grad[0]**2 + grad[1]**2)**(1/2)
#     cost = c_w * sum(magn)
#     return torch.mean(torch.tensor(cost, requires_grad=True))

#%%
# network = Network()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# network.to(device)
# mean_f_gpu = torch.tensor(mean_f).to(device)
# mean_d_gpu = torch.tensor(mean_d).to(device)
# uk_gpu = torch.tensor(uk).to(device)
# loss_list = []

# # Hyperparameters
# lr = 0.00001
# optimizer = optim.Adam(network.parameters(), lr=0.01)
# num_epoch = 100

# for n in range(num_epoch):
#     epoch_cost = []
#     total_loss = 0
#     total_correct = 0
    
#     for i in range(len(data_raw)):
#         data = torch.tensor(normalize(data_raw[i]).astype(np.float32), requires_grad=True, dtype=torch.float32)
#         data = data.unsqueeze(0).unsqueeze(0).to(device)
        
#         preds = network(data)
        
#         weight = preds[0][0]
        
#         loss = cost_function(data[0][0], mean_f_gpu, mean_d_gpu, weight, uk_gpu)
        
#         loss.backward()
#         optimizer.step()
#         epoch_cost.append(loss.item())
        
#         if i%500 == 0: print(f"Iteration {i} loss: {loss.item()}")
        
#         torch.cuda.empty_cache()
#         del data
#         del preds
#         del weight


#     print(f"Epoch {n} mean loss: {sum(epoch_cost)/len(epoch_cost)}")
    
#     loss_list.append(sum(epoch_cost)/len(epoch_cost))
























































































