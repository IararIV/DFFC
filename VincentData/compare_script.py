# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:30:34 2020

@author: lqg38422
"""

import numpy as np
from PIL import Image
from tomobar.supp.suppTools import normaliser
import os

PROJ_PATH = "C:\\Users\\lqg38422\\Desktop\\VincentTest\\tomo_files_Oak_1\\"
FIELDS_PATH = "C:\\Users\\lqg38422\\Desktop\\VincentTest\\Oak_1_flat_dark_slices\\"

projections_raw = []

for img_name in os.listdir(PROJ_PATH):
    projections_raw.append(np.array(Image.open(PROJ_PATH + img_name)))

tmp = np.array(projections_raw)
projections_raw = tmp[:400]

#%%

DARKS_PATH = ["dark_before_Oak_1\\", "dark_after_Oak_1\\"]
FLATS_PATH = ["flats_before_Oak_1\\", "flats_after_Oak_1\\"]

darks = []

for dark_name in DARKS_PATH:
    for img_name in os.listdir(FIELDS_PATH + dark_name):
        darks.append(np.array(Image.open(FIELDS_PATH + dark_name + img_name)))
    
darks = np.array(darks)

flats = []

for flat_name in FLATS_PATH:
    for img_name in os.listdir(FIELDS_PATH + flat_name):
        flats.append(np.array(Image.open(FIELDS_PATH + flat_name + img_name)))
    
flats = np.array(flats)

#%%

import numpy as np
import scipy
from skimage.transform import downscale_local_mean

def DFFC(data, flats, darks, downsample=20):
    # Load frames
    meanDarkfield = np.mean(darks, axis=0)
    whiteVect = np.zeros((flats.shape[0], flats.shape[1]*flats.shape[2]))
    k = 0
    for ff in flats:
        tmp = ff - meanDarkfield
        whiteVect[k] = tmp.flatten() - meanDarkfield.flatten()
        k += 1  
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

    def parallelAnalysis(flatFields, repetitions):
        stdEFF = np.std(flatFields, axis=0, ddof=1)
        # import matplotlib.pyplot as plt
        # plt.plot(stdEFF) #what is that >:o
        H, W = flatFields.shape
        keepTrack = np.zeros((H, repetitions))
        stdMatrix = np.tile(stdEFF, (H, 1))
        for i in range(repetitions):
            #print(f"Parallel Analysis - repetition {i}")
            sample = stdMatrix * np.random.randn(H, W)
            D1, _ = np.linalg.eig(np.cov(sample))
            keepTrack[:,i] = D1.copy()
        mean_flat_fields_EFF = np.mean(flatFields, axis=0)
        F = flatFields - mean_flat_fields_EFF
        D1, V1 = np.linalg.eig(np.cov(F))
        selection = np.zeros((1,H))
        # mean + 2 * std
        selection[:,D1>(np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1, ddof=1))] = 1
        numberPC = np.sum(selection)
        return V1, D1, int(numberPC)

    # Parallel Analysis
    nrPArepetions = 10
    nrEigenflatfields = 0
    att = 0
    print("Parallel Analysis:")
    while (nrEigenflatfields <= 0):
        print("Attempt number", att)
        att += 1
        V1, D1, nrEigenflatfields = parallelAnalysis(Data,nrPArepetions)
    print(f"{nrEigenflatfields} eigen flat fields selected!")

    # Calculation eigen flat fields
    C, H, W = data.shape
    eig0 = mn.reshape((H,W))
    EFF = np.zeros((nrEigenflatfields+1, H, W)) #n_EFF + 1 eig0
    print("Calculating EFFs:")
    EFF[0] = eig0
    for i in range(nrEigenflatfields):
        EFF[i+1] = (np.matmul(Data.T, V1[-i]).T).reshape((H,W))
    print("Done!")

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

    n_im = len(data)
    print("DFFC:")
    clean_DFFC = np.zeros((n_im, H, W))
    for i in range(n_im):
        if i%100 == 0: print("Iteration", i)
        #print("Estimation projection:")
        projection = data[i]
        # Estimate weights for a single projection
        meanFF = EFF[0]
        FF = EFF[1:]
        weights = np.zeros(nrEigenflatfields)
        x=condTVmean(projection, meanFF, FF, meanDarkfield, weights, downsample)
        # Dynamic FFC
        FFeff = np.zeros(meanDarkfield.shape)
        for j in range(nrEigenflatfields):
            FFeff = FFeff + x[j] * EFF[j+1]
        epsilon = 10e-10
        tmp = np.divide((projection - meanDarkfield),(EFF[0] + FFeff) + epsilon)
        clean_DFFC[i] = tmp

    return clean_DFFC

#%%
projections_normalised = DFFC(projections_raw, flats, darks)

#%%
projections_normalised[projections_normalised > 0.0] = -np.log(projections_normalised[projections_normalised > 0.0])
projections_normalised[projections_normalised < 0.0] = 0.0

import matplotlib.pyplot as plt
plt.figure(), plt.imshow(projections_normalised[0])