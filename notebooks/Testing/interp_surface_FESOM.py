#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:10:38 2024

@author: dlizarbe
"""

## Let's interpolate the real deal 2

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import cmocean
import sys
import pyfesom2 as pf
import numpy as np
from scipy.interpolate import griddata

# Load the mesh

alpha, beta, gamma=[0, 0, 0]
print("mesh will be loaded")
# Insert your custom path
meshpath = '/Users/dlizarbe/Documents/PhD/FESOM'
mesh = pf.load_mesh(meshpath, abg=[alpha, beta, gamma], usepickle = False)
#resultpath = f'{meshpath}results/'

print(meshpath)

meshdiag = xr.open_mfdataset(f'{meshpath}/fesom.mesh.diag.nc')
#print(meshdiag)

# Use the 'elem' dimension size directly
elem_n = meshdiag.dims["elem"]

# Initialize arrays based on the number of elements #array full of zeros
xx2 = np.zeros(shape=(elem_n)) 
yy2 = np.zeros(shape=(elem_n))

#mesh.x2 is the longitude in vector
#mesh.y2 is the latitude in vector

mesh.x2[mesh.elem[1,:]].mean(axis=0)

#lon=mesh.x2; lat=mesh.y2;

#aa = mesh.elem ---> these are the indices of the elements in the triangle
## the loops find the mean of triangle elements or the centroid
for i in np.arange(0,elem_n):
    xx2[i]=mesh.x2[mesh.elem[i,:]].mean(axis=0)
    yy2[i]=mesh.y2[mesh.elem[i,:]].mean(axis=0)


cyclic_length=4.5
unglue = True
if unglue:
    try:
        tri = np.loadtxt(f'{meshpath}/elem2d.out', skiprows=1, dtype=int)
        nodes = np.loadtxt(f'{meshpath}/nod2d.out', skiprows=1)
        xcoord, ycoord = nodes[:, 1], nodes[:, 2]
        xc, yc = xcoord[tri - 1], ycoord[tri - 1]
        xmin = xc.min(axis=1)
        for i in range(3):
            ai = np.where(xc[:, i] - xmin > cyclic_length / 2)
            xc[ai, i] -= cyclic_length
        X = xc.mean(axis=1)
        Y = yc.mean(axis=1)
    except FileNotFoundError:
        print("Required files for ungluing not found; skipping unglue step.")

## the grid
#Define the target grid with shape (72, 292)
lon_grid, lat_grid = np.meshgrid(
    np.linspace(X.min(), X.max(), 72),
    np.linspace(Y.min(), Y.max(), 292)
)

# Flattened original data arrays (21120,)
X = X.flatten()
Y = Y.flatten()

# Combine X and Y into a single array of shape (21120, 2) for original coordinates
coords = np.column_stack((X, Y))

# Flatten the target grid for griddata
target_coords = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))

# Data selection
str_id = 'u'
year = 1958
level = 0
time = -1
dat1 = xr.open_dataset(f'{meshpath}/{str_id}.fesom.{year}.nc')[str_id]
#time: 365, nz1: 40, elem: 21120

#%% interpolate in loop linearly

#u_grid = np.zeros((292,72,dat1.shape[1],dat1.shape[0])) 
u_grid = np.zeros((292,72,dat1.shape[0])) 

level = 0
#for itime in range(0,dat1.shape[0]):
for itime in range(0,3):

    dat2 = dat1.isel(time=itime, nz1=level) # may need to change nz1 to nz, depending on quantity you plot
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
    u_grids = data_grid_flat.reshape((292, 72))
    u_grid[:,:,itime] = u_grids

    #print(itime)
    print(dat1.time.values[itime].astype('datetime64[D]'))
    #u_grid[:,:,level] = data_grid_flat.reshape((292, 72))

#%% #

vmin, vmax = np.round(dat.min().values), np.round(dat.max().values)
cmap = cmocean.cm.balance

cbartext, cont	= f'{str_id} / {dat.units}', [vmin, vmax, .001]
bounds=np.linspace(vmin,vmax,100)
fig, ax = plt.subplots(figsize=(5,20))
#plt.gca().set_aspect('equal')

im = ax.pcolormesh(lon_grid, lat_grid, u_grid[:,:,0], shading='auto', cmap=cmap) 
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
plt.title('Nearest '+f'{dat.time.values}, (level,nz1)=({level},{dat.nz1.values})')

#plt.savefig('Results/Channel_u_plot_nearest'+'.png',
#   format='png', dpi=300, transparent=False, bbox_inches='tight')
plt.show(block=True)

