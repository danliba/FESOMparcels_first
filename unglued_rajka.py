#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:48:59 2024

@author: dlizarbe
"""

# Import necessary libraries
import datetime
import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import pyfesom2 as pf
from matplotlib.colors import TwoSlopeNorm

def plot(
    #resultpath='/Users/dlizarbe/Documents/PhD/FESOM/',  
    str_id=str, 
    year=int, 
    time=int, 
    level=int, 
    meshpath=str, 
    filter_lat=False, 
    lat_north=12.7525, 
    lat_south=5.0071, 
    unglue=True, 
    cyclic_length=4.5
):
    """
    Plot data of the FESOM model output for the Soufflet channel 
    for a given year, time, and vertical level.
    """
    
    # Set rotation angles
    alpha, beta, gamma = 0, 0, 0

    # Infer meshpath if not provided
    if not meshpath:
        inferred_meshpath = os.path.join(meshpath, '../')
        meshpath = inferred_meshpath

#meshdiag = xr.open_mfdataset(f'{meshpath}/fesom.mesh.diag.nc')

    # Load mesh diagnostics file
    try:
        meshdiag = xr.open_mfdataset(f'{meshpath}/fesom.mesh.diag.nc')
    except FileNotFoundError:
        print(f"Mesh diagnostics file not found in {meshpath}")
        return

    # Load mesh
    print("Loading mesh...")
    mesh = pf.load_mesh(meshpath, abg=[alpha, beta, gamma], usepickle=False)

    # Calculate mean X and Y coordinates for each element
    elem_n = meshdiag.elem.shape[0]
    xx2 = mesh.x2[mesh.elem[:, :elem_n]].mean(axis=1)
    yy2 = mesh.y2[mesh.elem[:, :elem_n]].mean(axis=1)

    # Load dataset and select time
    try:
        dat = xr.open_mfdataset(f'{meshpath}/{str_id}.fesom.{year}.nc')[str_id].isel(time=time)
    except (KeyError, FileNotFoundError) as e:
        print(f"Failed to load data file or variable '{str_id}': {e}")
        return

    # Check if data has a vertical dimension and select level if present
    if 'nz1' in dat.dims or 'nz' in dat.dims:
        dat = dat.isel(nz1=level) if 'nz1' in dat.dims else dat.isel(nz=level, missing_dims="ignore")
    dat = dat.squeeze()  # Remove extra dimensions

    # Determine plot coordinates
    X, Y = (meshdiag.lon, meshdiag.lat) if 'nod2' in dat.dims else (xx2, yy2)

    # Filter data based on latitude limits, if specified
    if filter_lat:
        print("Filtering latitudes...")
        lat_data = meshdiag.lat if 'nod2' in dat.dims else yy2
        lat_mask = (lat_data >= lat_south) & (lat_data <= lat_north)
        
        if 'nod2' in dat.dims:
            lat_mask = xr.DataArray(lat_mask, dims=["nod2"], coords={"nod2": dat["nod2"]})
        else:
            lat_mask = xr.DataArray(lat_mask, dims=["elem"], coords={"elem": dat["elem"]})
        
        dat = dat.where(lat_mask, drop=True)
        X = X[lat_mask.values]
        Y = Y[lat_mask.values]

    # Unglue mesh if required
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

    # Set units and colormap based on units
    units = getattr(dat, 'units', 'none')
    colormap_info = {
        'm/s': (cmocean.cm.balance, 'balance'),
        '1/s': (cmocean.cm.balance, 'balance'),
        'C': (cmocean.cm.thermal, 'thermal'),
        'm': ('Greys', 'greyscale')
    }
    cmap, cmap_type = colormap_info.get(units, (cmocean.cm.balance, 'balance'))

    # Determine color limits
    dat_min, dat_max = dat.min().values, dat.max().values
    if units == 'm':
        vmin, vmax = int(np.round(dat_min)), int(np.round(dat_max))
    elif units in {'m/s', '1/s'}:
        max_val = max(abs(dat_min), abs(dat_max))
        max_val = np.round(max_val, 7) if max_val < 1e-5 else np.round(max_val, 5)
        vmin, vmax = -max_val, max_val
    elif units == 'C':
        vmin, vmax = np.round(dat_min, 1), np.round(dat_max, 1)
    else:
        vmin, vmax = dat_min, dat_max

    # Set up figure
    fig, ax = plt.subplots(figsize=(5, 20))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if units in {'m/s', '1/s', 'none'} else None
    im = ax.tripcolor(X, Y, dat, shading='flat', cmap=cmap, norm=norm, 
                      **({'vmin': vmin, 'vmax': vmax} if norm is None else {}))

    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('lon / deg', size=10)
    ax.set_ylabel('lat / deg', size=10)

    cbar = fig.colorbar(im, orientation='horizontal', pad=0.03, extend='both')
    cbartext = f'{str_id} / {units}'
    cbar.set_label(cbartext, size=10)
    cbar.ax.tick_params(labelsize=10)

    def remove_time(datetime=None) -> np.datetime64:
        return datetime.astype('datetime64[D]')

    try:
        title_text = f'{remove_time(dat.time.values)}, (level,nz1)=({level},{dat.nz1.values}m)'
    except AttributeError:
        title_text = f'{remove_time(dat.time.values)}, (level,nz)=({level},{getattr(dat, "nz", "N/A")}m)'
    plt.title(title_text, loc='center', pad=20, fontsize=14, color='black', y=0.95)

    plt.show(block=False)
