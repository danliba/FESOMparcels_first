#!/bin/bash

#SBATCH --job-name=fesom_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=base
#SBATCH --output=fesom_output_%j.log
#SBATCH --error=fesom_error_%j.log

# Load necessary modules
module load gcc12-env/12.3.0 
module load singularity/3.11.5

# Define base parameters
container_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/parcels-container_2024.10.03-921b2b0.sif"  # Path to your container
notebook_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/notebooks/Parcels_regridded_FESOM.ipynb"
path1="/gxfs_work/geomar/smomw662/FESOM_data/channel/"
mesh_fn="fesom.mesh.diag.nc"
out_path="/gxfs_work/geomar/smomw662/FESOMparcels_first/data/"
out_fn_prefix="UVW_FESOM_parcels"  # Prefix for output files
num_particles=10_000
days=400

# Iterate over years
for year in {2009..2010}; do
    u_path="u.fesom.${year}.nc"
    v_path="v.fesom.${year}.nc"
    w_path="w.fesom.${year}.nc"
    out_fn="${out_fn_prefix}_${year}"  # Output file for each year

    # Run Python script inside the container
    srun singularity exec -B /gxfs_work:/gxfs_work "${container_path}" bash -c \
        "python ${notebook_path} \
        ${num_particles} \
        ${days} \
        ${path1} \
        ${mesh_fn} \
        ${u_path} \
        ${v_path} \
        ${w_path} \
        ${out_path} \
        ${out_fn}\
        ${year}"
done

### create a log file with the deployment details. 

