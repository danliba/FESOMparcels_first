#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:45:27 2024

Function to interpolate FESOM model data using various methods.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import pyfesom2 as pf
from scipy.interpolate import griddata

def interpolate_fesom_data(
    meshpath,
    str_id='u',
    year=1958,
    method='linear',
    cyclic_length=4.5,
    unglue=True
):
    """
    Interpolates FESOM model data to a target grid.

    Parameters:
    - meshpath (str): Path to the mesh files.
    - str_id (str): Variable name to plot (e.g., 'u').
    - year (int): Year of the data file to use.
    - method (str): Interpolation method ('linear', 'cubic', 'nearest').
    - cyclic_length (float): Width of the channel in degrees for ungluing.
    - unglue (bool): Whether to unglue the data if mesh is cyclic.

    Returns:
    - u_grid (np.array): Interpolated data array.
    """

    # Load the mesh
    alpha, beta, gamma = [0, 0, 0]
    mesh = pf.load_mesh(meshpath, abg=[alpha, beta, gamma], usepickle=False)
    meshdiag = xr.open_mfdataset(f'{meshpath}/fesom.mesh.diag.nc')

    # Initialize arrays based on the number of elements
    elem_n = meshdiag.dims["elem"]
    xx2 = np.zeros(elem_n)
    yy2 = np.zeros(elem_n)

    # Calculate centroids of triangular elements
    for i in range(elem_n):
        xx2[i] = mesh.x2[mesh.elem[i, :]].mean(axis=0)
        yy2[i] = mesh.y2[mesh.elem[i, :]].mean(axis=0)

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
            X, Y = xx2, yy2  # Fallback if files not found
    else:
        X, Y = xx2, yy2

    # Define the target grid with shape (72, 292)
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(X.min(), X.max(), 72),
        np.linspace(Y.min(), Y.max(), 292)
    )

    # Flatten original and target coordinates for griddata
    coords = np.column_stack((X.flatten(), Y.flatten()))
    target_coords = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))

    # Load dataset and initialize the output array
    dat1 = xr.open_dataset(f'{meshpath}/{str_id}.fesom.{year}.nc')[str_id]
    int_grid = np.zeros((292, 72, dat1.shape[1], dat1.shape[0]))

    # Interpolate data for each time and level
    for itime in range(dat1.shape[0]):
        for level in range(dat1.shape[1]):
            dat2 = dat1.isel(time=itime, nz1=level).squeeze()
            data_flat = dat2.values.flatten()

            # Interpolate the data to the target grid
            data_grid_flat = griddata(
                coords,          # Original element coordinates
                data_flat,       # Original data values
                target_coords,   # Target grid coordinates
                method=method    # Interpolation method
            )

            # Reshape and assign to output array
            int_grid[:, :, level, itime] = data_grid_flat.reshape((292, 72))

        print(f"Processed time index {itime}, date: {dat1.time.values[itime].astype('datetime64[D]')}")

    return int_grid

# Example usage
# meshpath = '/Users/dlizarbe/Documents/PhD/FESOM'
# interpolated_data = interpolate_fesom_data(meshpath, str_id='u', year=1958, method='linear')
