#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:44:04 2024

@author: dlizarbe
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Define grid dimensions
time_steps = 100  # Number of time steps
lat_points = 200  # Number of latitude points
lon_points = 100  # Number of longitude points

# Create lat/lon grid
x = np.linspace(-10, 10, lon_points)  # Longitude-like range
y = np.linspace(-10, 10, lat_points)  # Latitude-like range
X, Y = np.meshgrid(x, y)

# Eddies parameters
vortex_strength = 50  # Maximum velocity magnitude
vortex_radius = 5     # Radius of influence of each vortex
angular_velocity = 0.05  # Angular rotation speed of the eddies

# Cyclonic vortex center
cyclonic_center_x = -5
cyclonic_center_y = 0

# Anticyclonic vortex center
anticyclonic_center_x = 5
anticyclonic_center_y = 0

# Initialize velocity arrays
U = np.zeros((time_steps, lat_points, lon_points))  # Eastward velocity
V = np.zeros((time_steps, lat_points, lon_points))  # Northward velocity

# Create eddies over time
for t in range(time_steps):
    # Time-dependent rotation
    rotation_angle = t * angular_velocity
    
    # Cyclonic vortex (counterclockwise)
    cyclonic_x = cyclonic_center_x
    cyclonic_y = cyclonic_center_y
    
    R_cyclonic = np.sqrt((X - cyclonic_x)**2 + (Y - cyclonic_y)**2)  # Radial distance
    theta_cyclonic = np.arctan2(Y - cyclonic_y, X - cyclonic_x) + rotation_angle
    velocity_cyclonic = vortex_strength * np.exp(-R_cyclonic / vortex_radius)
    U_cyclonic = -velocity_cyclonic * np.sin(theta_cyclonic)
    V_cyclonic = velocity_cyclonic * np.cos(theta_cyclonic)
    
    # Anticyclonic vortex (clockwise)
    anticyclonic_x = anticyclonic_center_x
    anticyclonic_y = anticyclonic_center_y
    
    R_anticyclonic = np.sqrt((X - anticyclonic_x)**2 + (Y - anticyclonic_y)**2)  # Radial distance
    theta_anticyclonic = np.arctan2(Y - anticyclonic_y, X - anticyclonic_x) - rotation_angle
    velocity_anticyclonic = vortex_strength * np.exp(-R_anticyclonic / vortex_radius)
    U_anticyclonic = velocity_anticyclonic * np.sin(theta_anticyclonic)
    V_anticyclonic = -velocity_anticyclonic * np.cos(theta_anticyclonic)
    
    # Combine the velocity fields
    U[t, :, :] = U_cyclonic + U_anticyclonic
    V[t, :, :] = V_cyclonic + V_anticyclonic

# # Create the animation
# fig, ax = plt.subplots(figsize=(8, 6))

# def update(frame):
#     ax.clear()
#     ax.quiver(X, Y, U[frame], V[frame], scale=1000, pivot="middle", color="blue")
#     ax.set_title(f"Eddies at Time Step {frame}")
#     ax.set_xlabel("Longitude-like axis")
#     ax.set_ylabel("Latitude-like axis")
#     ax.set_xlim(-10, 10)
#     ax.set_ylim(-10, 10)
#     ax.grid()

# ani = FuncAnimation(fig, update, frames=time_steps, interval=50)

# # Save as a GIF
# writer = PillowWriter(fps=20)
# ani.save("rotating_eddies.gif", writer=writer)

# print("Animation saved as 'rotating_eddies.gif'")
#%%  now we delete the effect of the V velocity

V0 = np.zeros_like(V)

fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    ax.quiver(X, Y, U[frame], V0[frame], scale=1000, pivot="middle", color="blue")
    ax.set_title(f"Eddies at Time Step {frame}")
    ax.set_xlabel("Longitude-like axis")
    ax.set_ylabel("Latitude-like axis")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid()

ani = FuncAnimation(fig, update, frames=time_steps, interval=50)

# Save as a GIF
writer = PillowWriter(fps=20)
ani.save("rotating_eddies_V0.gif", writer=writer)

print("Animation saved as 'rotating_eddies.gif'")
 
# %% Now we run parcels above it

from parcels import ParticleSet
from parcels import JITParticle
from parcels import AdvectionRK4
from datetime import timedelta
from parcels import FieldSet
import xarray as xr


num_particles = 100
lon_start = np.random.uniform(-5,-2.5,size=(num_particles,)) 
lat_start = np.random.uniform(-2.5, 2.5, size=(num_particles,))


#%create an xarray of the fields
time = np.linspace(0, 99, 100)

ds = xr.Dataset(
    
    data_vars={
        'U': (('time','lat','lon'), U),
        'V': (('time','lat','lon'),V0),
        },
    
    coords= {
        'lat':y,
        'lon':x,
        'time': time
        }
    )
## now the fieldset
fieldset = FieldSet.from_xarray_dataset(
    ds,
    variables={'U':"U", "V":"V"},
    dimensions={'lon':'lon',
                'lat':'lat',
                'time':'time'},
    time_periodic=False,
    allow_time_extrapolation=True,
)
#fieldset

## ad halo
fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])
fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])

fieldset.add_periodic_halo(zonal=True)
## custome kernel for the halo
def periodicBC(particle, fieldset, time):
    if particle.lon < fieldset.halo_west:
        particle_dlon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle_dlon -= fieldset.halo_east - fieldset.halo_west

##
pset = ParticleSet.from_list(
    pclass = JITParticle,
    fieldset = fieldset,
    lon=lon_start,
    lat=lat_start,
)

## halo
output_file = pset.ParticleFile(name='PeriodicParticle', 
                                outputdt=timedelta(hours=1))

pset.execute(
    [AdvectionRK4,periodicBC],
    runtime=timedelta(days=10),
    dt=timedelta(minutes=5),
    output_file=output_file
)


#%%  we plot
plt.plot(lon_start,lat_start,'m.')
plt.plot(pset.lon, pset.lat, 'c.')

plt.show()

ds_part = xr.open_zarr("PeriodicParticle.zarr")

plt.plot(ds_part.lon.T, ds_part.lat.T, "c-")
plt.xlabel("Zonal distance (m)")
plt.ylabel("Meridional distance (m)")
plt.show()




