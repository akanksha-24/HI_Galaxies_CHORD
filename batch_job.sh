#!/bin/bash

#SBATCH --account=def-spekkens-ab
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=750G
#SBATCH --time=00:10:00
#SBATCH --job-name=catalog_gen

## loading the modules we need
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

module --force purge
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06

cd /home/akanksha/projects/def-spekkens-ab/akanksha/chord/HI_Galaxies_CHORD/
mpirun -n 4 python -u Sim_catalog.py
