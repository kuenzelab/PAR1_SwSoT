#!/bin/bash
#SBATCH -J rmsd_PAR1_holo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --array=201-300

module purge
module load PLUMED
module load SciPy-bundle

python3 RMSD_array.py
