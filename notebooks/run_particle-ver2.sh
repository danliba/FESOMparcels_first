#!/bin/bash

#SBATCH --job-name=fesom_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH --partition=base
#SBATCH --output=run_particles_output_%j.log
#SBATCH --error=run_particles_error_%j.log

# Load necessary modules
module load gcc12-env/12.3.0 
module load singularity/3.11.5

# Define base parameters
container_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/parcels-container_2024.10.03-921b2b0.sif"  # Path to your container
notebook_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/notebooks/Particle_run-ver2.ipynb"
output_notebook="/gxfs_work/geomar/smomw662/FESOMparcels_first/out_notebooks/one_particle.ipynb"
num_particles=1
yrst=1960
yren=2057
days=35400
lon_start=2
lat_start=10
depth=1500

# Run the Jupyter notebook using Papermill inside the container
srun --ntasks=1 --exclusive singularity exec -B /gxfs_work:/gxfs_work -B $PWD:/work --pwd /work "${container_path}" bash -c \
    ". /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && papermill ${notebook_path} ${output_notebook} \
        -p num_particles ${num_particles} \
        -p yrst ${yrst} \
        -p yren ${yren} \
        -p days ${days} \
        -p lon_start ${lon_start} \
        -p lat_start ${lat_start} \
        -p depth ${depth} \
        "
