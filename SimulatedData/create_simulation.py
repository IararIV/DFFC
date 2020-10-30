# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:59:44 2020

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

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N_size = 256 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc=timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5*N_size)
#plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel,:,:],vmin=0, vmax=1)
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(phantom_tm[:,sliceSel,:],vmin=0, vmax=1)
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(phantom_tm[:,:,sliceSel],vmin=0, vmax=1)
plt.title('3D Phantom, sagittal view')
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

#%%
print ("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

intens_max_clean = np.max(projData3D_analyt)
sliceSel = 150
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt[:,sliceSel,:],vmin=0, vmax=intens_max_clean)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel,:,:],vmin=0, vmax=intens_max_clean)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_analyt[:,:,sliceSel],vmin=0, vmax=intens_max_clean)
plt.title('Tangentogram view')
plt.show()

#%%
print ("Simulate synthetic flat fields, add flat field background to the projections and add noise")
I0  = 8000; # Source intensity
flatsnum = 80 # the number of the flat fields required

[projData3D_noisy, flatsSIM] = synth_flats(projData3D_analyt,
                                           source_intensity = I0, source_variation=0.035,\
                                           arguments_Bessel = (1,10,10,12),\
                                           strip_height = 0.15, strip_thickness = 1,\
                                           sigmasmooth = 1, flatsnum=flatsnum)
#del projData3D_analyt
plt.figure()
plt.subplot(121)
plt.imshow(projData3D_noisy[:,0,:])
plt.title('2D Projection (before normalisation)')
plt.subplot(122)
plt.imshow(flatsSIM[:,0,:])
plt.title('A selected simulated flat-field')
plt.show()

darksSIM = np.zeros(flatsSIM.shape)

#%%
import h5py
# save data
h5f = h5py.File('synth_data.h5', 'w')
h5f.create_dataset('proj_noisy', data=projData3D_noisy)
h5f.create_dataset('proj_analyt', data=projData3D_analyt)
h5f.create_dataset('flats', data=flatsSIM)
h5f.create_dataset('darks', data=darksSIM)
h5f.create_dataset('phantom', data=phantom_tm)
h5f.close()
