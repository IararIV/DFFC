#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:41:51 2020

@author: Gerard Jover Pujol

Load datasets from beamline i23
"""

# Obj

class Observations:
    
    def __init__(self, user):
        if user in ['Daniil']: #write your FedID here
            self.path = None
        elif user in ['Gerard', 'lqg38422']:
            self.path = '/dls/science/users/lqg38422/data/'
        
        self.datasets = dict()
        
    def create_dataset(self, num):
        self.datasets[num] = Dataset(num, self.path)

class Dataset:
    
    def __init__(self, num, path):
        self.num = num
        
        if num == 12923:
            self.n_darks = 20
            self.n_flats = 40
            self.n_projections = 1840
            self.n_post_flats = 0
            
        elif num == 13012:
            pass
        
        elif num == 13282:
            self.n_darks = 40
            self.n_flats = 80
            self.n_projections = 1881
            self.n_post_flats = 0
            
        self.path = path
        self.vert_tuple = [i for i in range(500,1300)] # selection of vertical slice

        
    def load_data(self):
        import h5py
        import numpy as np
        
        h5py_list = h5py.File(self.path + f'{self.num}/{self.num}.nxs','r')
        
        darks = h5py_list['/entry1/instrument/flyScanDetector/data'][:self.n_darks,self.vert_tuple,:]
        flats = h5py_list['/entry1/instrument/flyScanDetector/data'][self.n_darks:self.n_flats,self.vert_tuple,:]
        
        data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][self.n_flats:self.n_projections,self.vert_tuple,:]
        angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
        angles_rad = angles[80:1881]*np.pi/180.0
        
        h5py_list.close()
        
        self.vert_tuple = [i for i in range(500,1300)] # at the moment this value can't be changed
       
        return darks, flats, data_raw, angles_rad
    
# =============================================================================
# EXAMPLE
# =============================================================================

# Log in        
user = 'Gerard'
user = Observations(user) 

# Create dataset
num = 12923

user.create_dataset(num)

# Load data
darks, flats, data_raw, angles_rad = user.datasets[num].load_data()







