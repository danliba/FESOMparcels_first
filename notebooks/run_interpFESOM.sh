#!/bin/bash

#SBATCH --job-name=fesom_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --partition=base
#SBATCH --output=particles_output_%j.log
#SBATCH --error=particles_error_%j.log

# Load necessary modules
module load gcc12-env/12.3.0 
module load singularity/3.11.5

# Define base parameters
container_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/parcels-container_2024.10.03-921b2b0.sif"  # Path to your container
notebook_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/notebooks/FESOM_interpolator.ipynb"
output_notebook="/gxfs_work/geomar/smomw662/FESOMparcels_first/notebooks/FESOM_interpolator_out.ipynb"


# Run the Jupyter notebook using Papermill inside the container
srun --ntasks=1 --exclusive singularity exec -B /gxfs_work:/gxfs_work -B $PWD:/work --pwd /work "${container_path}" bash -c \
    ". /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && papermill ${notebook_path} ${output_notebook}"
