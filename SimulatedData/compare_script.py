# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:30:34 2020

@author: lqg38422
"""

import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.flatsgen import synth_flats
import h5py

h5f = h5py.File("synth_data.h5", "r")
projData3D_noisy = np.array(h5f["proj_noisy"], dtype=np.float64)
projData3D_analyt = np.array(h5f["proj_analyt"], dtype=np.float64)
flats = np.array(h5f["flats"], dtype=np.float64)
darks = np.array(h5f["darks"], dtype=np.float64)
phantom_tm = np.array(h5f["phantom"], dtype=np.float64)
h5f.close()

# Projection geometry related parameters:
N_size = 256 # Define phantom dimensions using a scalar value (cubic phantom)
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)
intens_max_clean = projData3D_analyt.max()

#%%
from tomobar.supp.suppTools import normaliser
projData3D_norm = normaliser(projData3D_noisy, flats, darks, log='True', method='mean')

intens_max = np.max(projData3D_norm)
sliceSel = 150
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_norm[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(projData3D_norm[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_norm[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()

#%%
# initialise tomobar DIRECT reconstruction class ONCE
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'cpu')

print ("Reconstruction using FBP from tomobar")
recNumerical= RectoolsDIR.FBP(projData3D_norm) # FBP reconstruction
recNumerical *= intens_max_clean

sliceSel = int(0.5*N_size)
max_val = 1
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(recNumerical[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, axial view')

plt.subplot(132)
plt.imshow(recNumerical[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(recNumerical[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, sagittal view')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_tm, recNumerical)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {}".format(RMSE))
