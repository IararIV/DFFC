#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:28:36 2020

@author: Gerard Jover Pujol

Dynamic intensity normalization usingeigen flat fields in X-ray imaging
"""

# =============================================================================
# 12923 - darks [20] - flats [20] - data [1800] - flats [0]
# 13012 -
# 13282 - darks [40] - flats [40] - data [1801] - flats [0]
# =============================================================================

# reading i23 data
import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy
from skimage.transform import downscale_local_mean
import random
from data_loader import Observations

user = 'Gerard'
user = Observations(user)

num = 12923
user.create_dataset(num)

# Load data
print(f"Loading {num} data...")
darks, flats, data_raw, angles_rad = user.datasets[num].load_data()
print("Done!")

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

# Load frames
meanDarkfield = np.mean(darks, axis=0)

whiteVect = np.zeros((flats.shape[0], flats.shape[1]*flats.shape[2]))
k = 0
for ff in flats:
    whiteVect[k] = ff.flatten() - meanDarkfield.flatten()
    k += 1

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
    # mean + 2 * std
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
EFF = np.zeros((nrEigenflatfields+1, H, W)) #n_EFF + 1 eig0
print("Calculating EFFs:")
EFF[0] = eig0
for i in range(nrEigenflatfields):
    EFF[i+1] = (np.matmul(Data.T, V1[-i]).T).reshape((H,W)) #why the last ones?
print("Done!")

# Filter EFFs (we skip this step since we want to try DL)

# =============================================================================
# Normalize function: intensity values between [0, 1]
# =============================================================================

def normalize(im):
    return (im - im.min()) / (im.max() - im.min())

# =============================================================================
# Function cost: objective function to optimize weights
# =============================================================================

def cost_func(x, *args):
    (projections, meanFF, FF, DF) = args
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

def condTVmean(projection, meanFF, FF, DF, x, DS):
    # Downsample image
    projection = downscale_local_mean(projection, (DS, DS))
    meanFF = downscale_local_mean(meanFF, (DS, DS))
    FF2 = np.zeros((FF.shape[0], meanFF.shape[0], meanFF.shape[1]))
    for i in range(len(FF)):
        FF2[i] = downscale_local_mean(FF[i], (DS, DS))
    FF = FF2
    DF = downscale_local_mean(DF, (DS, DS))
    
    # Optimize weights (x)
    x = scipy.optimize.minimize(cost_func, x, args=(projection, meanFF, FF, DF), method='BFGS')
    
    return x.x

# =============================================================================
# Conventional FFC
# =============================================================================

out_path = "./outDIRCFFC/"
path = os.getcwd() #get the current path folder
out_path = path+'/outDIRDFFC/'
try:
    os.makedirs(out_path)
except OSError:
    print ("Creation of the directory %s failed" % out_path)
else:
    print ("Successfully created the directory %s " % out_path)
n_im = len(data_raw)
meanVector = np.zeros((1, n_im))
for i in range(10):
    projection = data_raw[i]
    tmp = (projection - meanDarkfield)/EFF[0]
    meanVector[:,i] = np.mean(tmp)
    
    tmp[tmp<0] = 0
    tmp = -np.log(tmp)
    tmp[np.isinf(tmp)] = 10^5
    tmp = normalize(tmp)
    tmp = np.uint16((2**16-1)*tmp)
    tiff.imsave(f'{out_path}out_{i}.tiff', tmp)

# =============================================================================
# Dynamic FFC
# =============================================================================

out_path = './outDIRDFFC/'
path = os.getcwd() #get the current path folder
out_path = path+'/outDIRDFFC/'
try:
    os.makedirs(out_path)
except OSError:
    print ("Creation of the directory %s failed" % out_path)
else:
    print ("Successfully created the directory %s " % out_path)
    
n_im = 10
xArray = np.zeros((nrEigenflatfields, n_im))
keepImages = np.zeros((n_im, H, W))
downsample = 20
for i in range(n_im):
    print("Estimation projection:")
    projection = data_raw[i]
    # Estimate weights for a single projection
    meanFF = EFF[0]
    FF = EFF[1:]
    weights = np.zeros(nrEigenflatfields)
    x=condTVmean(projection, meanFF, FF, meanDarkfield, weights, downsample)
    xArray[:,i]=x
    
    # Dynamic FFC
    FFeff = np.zeros(meanDarkfield.shape)
    for j in range(nrEigenflatfields):
        FFeff = FFeff + x[j] * EFF[j+1]
    
    tmp = (projection - meanDarkfield)/(EFF[0] + FFeff)
    keepImages[i] = tmp.copy()
    tmp[tmp<0] = 0
    tmp = -np.log(tmp)
    tmp[np.isinf(tmp)] = 10^5
    tmp = normalize(tmp)
    tmp = np.uint16((2**16-1)*tmp)
    print(f"out_{i}.tiff saved!")
    tiff.imsave(f'{out_path}out_{i}.tiff', tmp)
    
#%% Evaluating results

# =============================================================================
# MSE (Mean Squared Error): mesaures the similarity between projections and corrections
# =============================================================================

def MSE(raw_im, corrected_im):
    return np.square(np.subtract(raw_im, corrected_im)).mean()

for i in range(n_im):
    proj = data_raw[i]
    clean = keepImages[i]
    res = MSE(proj, clean)
    print(f"Image number {i} - MSE: {res}")



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


























































































