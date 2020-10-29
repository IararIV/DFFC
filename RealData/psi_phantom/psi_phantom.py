#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:46:36 2020

@author: algol
"""

import numpy as np
from PIL import Image
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
import matplotlib.pyplot as plt

PROJECTIONS_PATH = '/home/algol/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/sample/'
FLATS_PATH = '/home/algol/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/flat-field/'
DARKS_PATH = '/home/algol/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/dark-field/'
#%%
projections = np.float32(np.zeros((500, 943, 1548)))
imagename = 'IMAT00006388_PSI_cylinder_Sample_'
for i in range(0,943):
    if (i < 10):
        index_str = '00' + str(i)
    elif (10 <= i < 100):
        index_str = '0' + str(i)
    else:
        index_str = str(i)
    tmp = np.array(Image.open(PROJECTIONS_PATH + imagename + index_str + ".tif"))
    projections[:,i,:] = tmp[500:1000,500:]
    
plt.figure()
plt.imshow(projections[:,5,:], vmin=0, vmax= 15000, cmap="gray")
#%%
darks = np.float32(np.zeros((500, 10, 1548)))
imagename = 'IMAT00006385_PSI_cylinder_dark_'
for i in range(0,10):
    if (i < 10):
        index_str = '00' + str(i)
    elif (10 <= i < 100):
        index_str = '0' + str(i)
    else:
        index_str = str(i)
    tmp = np.array(Image.open(DARKS_PATH + imagename + index_str + ".tif"))
    darks[:,i,:] = tmp[500:1000,500:]
#%%
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
    
plt.figure()
plt.imshow(flats[:,5,:], vmin=0, vmax= 15000, cmap="gray")
#%%
# normalising projections:
projections_norm = normaliser(projections[10:20,:,:], flats[10:20,:,:], darks[10:20,:,:], log='true', method='mean')
plt.figure()
plt.imshow(projections_norm[5,:,:], vmin=0, vmax= 2.0, cmap="gray")
#%%
# do reconstruction
angles_rad = np.zeros(943)
angles_step_rad = (360.0/943.0)*np.pi/180.0
for i in range(0,943):
    angles_rad[i] = i*angles_step_rad

RectoolsDIR = RecToolsDIR(DetectorsDimH = 1548,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = -118.0, # Center of Rotation (CoR) scalar 
                    AnglesVec = np.float32(angles_rad), # array of angles in radians
                    ObjSize = 1600, # a scalar to define reconstructed object dimensions
                    device_projector = 'gpu')

print ("Reconstruction using FBP from tomobar")
recFBP= RectoolsDIR.FBP(projections_norm[5,:,:]) # FBP reconstruction

plt.figure()
plt.imshow(recFBP, vmin=0, vmax= 0.00095, cmap="gray")

#%%