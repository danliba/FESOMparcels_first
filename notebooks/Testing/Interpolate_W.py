#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:29:07 2024

@author: dlizarbe
"""

## Let's interpolate the real deal 2

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from cmocean import cm
import sys
import pyfesom2 as pf
import numpy as np
from scipy.interpolate import griddata

# Load the mesh

alpha, beta, gamma=[0, 0, 0]
print("mesh will be loaded")
# Insert your custom path
meshpath = '/Users/dlizarbe/Documents/PhD/FESOM'
data_path = '/Users/dlizarbe/Documents/PhD/FESOM/data'
mesh = pf.load_mesh(meshpath, abg=[alpha, beta, gamma], usepickle = False)
#resultpath = f'{meshpath}results/'

print(meshpath)

meshdiag = xr.open_mfdataset(f'{meshpath}/fesom.mesh.diag.nc')

#print(meshdiag)

# Use the 'elem' dimension size directly
elem_n = meshdiag.dims["elem"]
nodes_n =meshdiag.dims["nod2"] 

# Initialize arrays based on the number of elements #array full of zeros
xx2 = np.zeros(shape=(elem_n)) 
yy2 = np.zeros(shape=(elem_n))

nx2 = np.zeros(shape=(nodes_n))
ny2 = np.zeros(shape=((nodes_n)))
#mesh.x2 is the longitude in vector
#mesh.y2 is the latitude in vector

mesh.x2[mesh.elem[1,:]].mean(axis=0)

#lon=mesh.x2; lat=mesh.y2;

X =mesh.x2
Y =mesh.y2
X = X.flatten()
Y = Y.flatten()

## the grid
#Define the target grid with shape (72, 292)
lon_grid, lat_grid = np.meshgrid(
    np.linspace(X.min(), X.max(), 103),
    np.linspace(Y.min(), Y.max(), 412)
)

# Flattened original data arrays (21120,)


# Combine X and Y into a single array of shape (21120, 2) for original coordinates
coords = np.column_stack((X, Y))

# Flatten the target grid for griddata
target_coords = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))

# Data selection
str_id = 'w'
year = 2005
level = 0
time = -1
dat1 = xr.open_dataset(f'{data_path}/{str_id}.fesom.{year}.nc')[str_id]
#time: 365, nz1: 40, elem: 21120

#%% 

W = dat1[0,0,:]
fig, ax = plt.subplots(figsize=(5,20))
#plt.gca().set_aspect('equal')
vmin, vmax = np.round(W.values.min(),8), np.round(W.values.max(),8)
cmap = cm.balance

im = ax.tripcolor(mesh.x2,mesh.y2,dat1[0,0,:], shading='flat', cmap=cm.thermal) 

plt.tick_params(axis='both', labelsize=20)
plt.xlabel('deg', size=20)
plt.ylabel('deg', size=20)

cbar = fig.colorbar(im, orientation='horizontal', pad=.05, extend='both') #  ticks=[v_min, 0, v_max],
im.set_clim(vmin, vmax)
#cbar.set_label(cbartext, size=20)
cbar.ax.tick_params(labelsize=20)

plt.show()
#%% interpolate in loop linearly

#u_grid = np.zeros((292,72,dat1.shape[1],dat1.shape[0])) 
w_grid = np.zeros((412,103,dat1.shape[0])) 

level = 0
#for itime in range(0,dat1.shape[0]):
for itime in range(0,3):

    dat2 = dat1.isel(time=itime, nz=level) # may need to change nz1 to nz, depending on quantity you plot
    dat = dat2.squeeze()
    data_flat = dat.values.flatten()

    # Interpolate the data to the target grid
    data_grid_flat = griddata(
        coords,       # Original element coordinates
        data_flat,    # Original data values
        target_coords,  # Target grid coordinates
        method='linear'  # or 'cubic' for smoother results
    )
    #print(level)
    # Reshape interpolated data to (200, 200)
    w_grids = data_grid_flat.reshape((412,103))
    w_grid[:,:,itime] = w_grids

    #print(itime)
    print(dat1.time.values[itime].astype('datetime64[D]'))
    #u_grid[:,:,level] = data_grid_flat.reshape((292, 72))

#%% 

vmin, vmax = np.round(dat.min().values,6), np.round(dat.max().values,6)
cmap = cm.balance

cbartext, cont	= f'{str_id} / {dat.units}', [vmin, vmax, .001]
bounds=np.linspace(vmin,vmax,100)
fig, ax = plt.subplots(figsize=(5,20))
#plt.gca().set_aspect('equal')

im = ax.pcolormesh(lon_grid, lat_grid, w_grid[:,:,0], shading='auto', cmap=cmap) 
#ax.plot(X,Y, 'o', markersize=0.5, color='grey')

plt.tick_params(axis='both', labelsize=20)
plt.xlabel('deg', size=20)
plt.ylabel('deg', size=20)

cbar = fig.colorbar(im, orientation='horizontal', pad=.05, extend='both') #  ticks=[v_min, 0, v_max],
im.set_clim(vmin, vmax)
cbar.set_label(cbartext, size=20)
#cbar.set_ticks([round(i,8) for i in np.linspace(cont[0], cont[1], 5)], fontsize=20)
#cbar.set_ticklabels([round(i,8) for i in np.linspace(cont[0], cont[1], 5)], fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.title('Linear '+f'{dat.time.values}, (level,nz)=({level},{dat.nz.values})')

#plt.savefig('Results/Channel_u_plot_nearest'+'.png',
#   format='png', dpi=300, transparent=False, bbox_inches='tight')
plt.show(block=True)
