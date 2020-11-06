# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:55:28 2020

@author: lqg38422
"""

import numpy as np
from PIL import Image
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
import matplotlib.pyplot as plt
import h5py

PROJECTIONS_PATH = 'C:/Users/lqg38422/Desktop/PSI_Phantom/projections/'
FLATS_PATH = 'C:/Users/lqg38422/Desktop/PSI_Phantom/flat-field/'
DARKS_PATH = 'C:/Users/lqg38422/Desktop/PSI_Phantom/dark-field/'

N_size = 2048
N_cut = 500
n_proj = 10 #943

#%%
print("Loading projections...")
projections = np.float64(np.zeros((N_cut, n_proj, N_size)))
imagename = 'IMAT00006388_PSI_cylinder_Sample_'
for i in range(0,n_proj):
    index_str = str(i).zfill(3)
    tmp = np.array(Image.open(PROJECTIONS_PATH + imagename + index_str + ".tif"))
    projections[:,i,:] = tmp[:N_cut,:]
    
plt.figure()
plt.imshow(projections[:,5,:], vmin=0, vmax= 15000, cmap="gray")
print("Done!")
#%%
print("Loading darks...")
darks = np.float64(np.zeros((N_cut, 10, N_size)))
imagename = 'IMAT00006385_PSI_cylinder_dark_'
for i in range(0,10):
    index_str = str(i).zfill(3)
    tmp = np.array(Image.open(DARKS_PATH + imagename + index_str + ".tif"))
    darks[:,i,:] = tmp[:N_cut,:]
print("Done!")
#%%
"""
flats = np.float32(np.zeros((500, 59, 1548)))
imagename = 'Flat_'
for i in range(0,59):
    if (i < 10):
        index_str = '000' + str(i)
    elif (10 <= i < 100):
        index_str = '00' + str(i)
    elif (100 <= i < 1000):
        index_str = '0' + str(i)
    else:
        index_str = str(i)
    tmp = np.array(Image.open(FLATS_PATH + imagename + index_str + ".tif"))
    flats[:,i,:] = tmp[500:1000,500:]
    
"""
print("Loading flats...")
h5f = h5py.File("C:/Users/lqg38422/Desktop/PSI_Phantom/flats_corrected.h5", "r")
flats_full = np.array(h5f["/flats"], dtype=np.float64)
flats_full = np.swapaxes(flats_full, 1, 0)
flats = flats_full[:N_cut,:,:]
del flats_full
plt.figure()
plt.imshow(flats[:,5,:], vmin=0, vmax= 15000, cmap="gray")
print("Done!")

#%%
# normalising projections:
print("Mean normalisation...")
projections_norm = normaliser(projections, flats, darks, log='true', method='mean')
plt.figure()
plt.axis('off')
plt.imshow(projections_norm[:,5,:], vmin=-0.1, vmax= 0.4, cmap="gray")
print("Done!")

#%%
import numpy as np
import scipy
from skimage.transform import downscale_local_mean
import bm3d

def DFFC(data, flats, darks, downsample=20):
    # Load frames
    meanDarkfield = np.mean(darks, axis=1, dtype=np.float64)
    whiteVect = np.zeros((flats.shape[1], flats.shape[0]*flats.shape[2]), dtype=np.float64)
    for i in range(flats.shape[1]):
        #whiteVect[k] = ff.flatten() - meanDarkfield.flatten()
        whiteVect[i] = flats[:,i,:].flatten()
    mn = np.mean(whiteVect, axis=0)

    # Substract mean flat field
    M, N = whiteVect.shape
    Data = whiteVect - mn
    
    # =============================================================================
    # Parallel Analysis (EEFs selection):
    #      Selection of the number of components for PCA using parallel Analysis.
    #      Each flat field is a single row of the matrix flatFields, different
    #      rows are different observations.
    # =============================================================================
    
    def cov(X):
        one_vector = np.ones((1,X.shape[0]))
        mu = np.dot(one_vector, X) / X.shape[0]
        X_mean_subtract = X - mu
        covA = np.dot(X_mean_subtract.T, X_mean_subtract) / (X.shape[0] - 1);
        return covA

    def parallelAnalysis(flatFields, repetitions):
        stdEFF = np.std(flatFields, axis=0, ddof=1, dtype=np.float64)
        H, W = flatFields.shape
        keepTrack = np.zeros((H, repetitions), dtype=np.float64)
        stdMatrix = np.tile(stdEFF, (H, 1))
        for i in range(repetitions):
            print(f"Parallel Analysis - repetition {i}")
            sample = stdMatrix * np.random.randn(H, W)
            D1, _ = np.linalg.eig(np.cov(sample))
            keepTrack[:,i] = D1.copy()
        mean_flat_fields_EFF = np.mean(flatFields, axis=0)
        F = flatFields - mean_flat_fields_EFF
        # Checkpoint - bug in np.cov - Matlab size is (3170304, 59)
        # F.T is Matlab F
        D1, V1 = np.linalg.eig(np.cov(F))
        selection = np.zeros((1,H))
        # mean + 2 * std
        selection[:,D1>(np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1, ddof=1))] = 1
        numberPC = np.sum(selection)
        return V1, D1, int(numberPC)

    # Parallel Analysis
    nrPArepetions = 10
    nrEigenflatfields = 0
    print("Parallel Analysis:")
    while (nrEigenflatfields <= 0):
        V1, D1, nrEigenflatfields = parallelAnalysis(Data,nrPArepetions)
    print(f"{nrEigenflatfields} eigen flat fields selected!")
    idx = D1.argsort()[::-1]   
    D1 = D1[idx]
    V1 = V1[:,idx]

    # Calculation eigen flat fields
    H, C, W = data.shape
    eig0 = mn.reshape((H,W))
    EFF = np.zeros((nrEigenflatfields+1, H, W)) #n_EFF + 1 eig0
    print("Calculating EFFs:")
    EFF[0] = eig0
    for i in range(nrEigenflatfields):
        EFF[i+1] = (np.matmul(Data.T, V1[i]).T).reshape((H,W))
    print("Done!")
    
    # Denoise eigen flat fields
    print("Denoising EFFs:")
    for i in range(1, len(EFF)):
        print(f"Denoising EFF {i}")
        EFF[i,:,:] = bm3d.bm3d(EFF[i,:,:], sigma_psd=0.1)
        #EFF[i] = bm3d.bm3d(EFF[i], sigma_psd=30/255.0,\
        #                    stage_arg=bm3d.BM3DStages.ALL_STAGES).astype(np.float64)
    print("Done!")
    
    # =============================================================================
    # cost_func: cost funcion used to estimate the weights using TV
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
        x = scipy.optimize.minimize(cost_func, x, args=(projection, meanFF, FF, DF), method='BFGS', tol=1e-6)

        return x.x

    H, C, W = data.shape
    print("DFFC:")
    clean_DFFC = np.zeros((H, C, W), dtype=np.float64)
    for i in range(C):
        if i%25 == 0: print("Iteration", i)
        projection = data[:,i,:]
        # Estimate weights for a single projection
        meanFF = EFF[0]
        FF = EFF[1:]
        weights = np.zeros(nrEigenflatfields)
        x=condTVmean(projection, meanFF, FF, meanDarkfield, weights, downsample)
        # Dynamic FFC
        FFeff = np.zeros(meanDarkfield.shape)
        for j in range(nrEigenflatfields):
            FFeff = FFeff + x[j] * EFF[j+1]
        tmp = np.divide((projection - meanDarkfield),(EFF[0] + FFeff))
        clean_DFFC[:,i,:] = tmp

    return clean_DFFC

#%%
print("Dynamic normalisation...")
projections_dff = DFFC(projections, flats, darks, downsample=2)
projections_dff[projections_dff > 0.0] = -np.log(projections_dff[projections_dff > 0.0])
projections_dff[projections_dff < 0.0] = 0.0 # remove negative values

plt.figure()
plt.axis('off')
plt.imshow(projections_dff[:,5,:], vmin=-0.1, vmax=0.4, cmap="gray")

#%%
# do reconstruction
angles_rad = np.zeros(N_size)
angles_step_rad = (360.0/float(N_size))*np.pi/180.0
for i in range(0,N_size):
    angles_rad[i] = i*angles_step_rad

RectoolsDIR = RecToolsDIR(DetectorsDimH = N_cut,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = -118.0, # Center of Rotation (CoR) scalar 
                    AnglesVec = np.float32(angles_rad), # array of angles in radians
                    ObjSize = 1600, # a scalar to define reconstructed object dimensions
                    device_projector = 'cpu')

print ("Reconstruction using FBP from tomobar")
recFBP= RectoolsDIR.FBP(projections_norm[:,5,:]) # FBP reconstruction

plt.figure()
plt.imshow(recFBP, vmin=0, vmax= 0.00095, cmap="gray")

#%%
# do reconstruction
angles_rad = np.zeros(N_cut)
angles_step_rad = (360.0/float(N_cut))*np.pi/180.0
for i in range(0,N_cut):
    angles_rad[i] = i*angles_step_rad

RectoolsDIR = RecToolsDIR(DetectorsDimH = N_cut,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = -118.0, # Center of Rotation (CoR) scalar 
                    AnglesVec = np.float32(angles_rad), # array of angles in radians
                    ObjSize = 1600, # a scalar to define reconstructed object dimensions
                    device_projector = 'cpu')

print ("Reconstruction using FBP from tomobar")
recFBP= RectoolsDIR.FBP(projections_dff[:,5,:]) # FBP reconstruction

plt.figure()
plt.imshow(recFBP, vmin=0, vmax= 0.00095, cmap="gray")
#%%