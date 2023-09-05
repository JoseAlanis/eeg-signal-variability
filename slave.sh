#!/bin/bash

#SBATCH -A m2_jgu-amd		 # Account
#SBATCH -p parallel		     # Partition: parallel, smp, bigmem
#SBATCH -C skylake 		     # architecture Skylake (64 Cores) or Broadwell (40 Cores)
#SBATCH -N 1                 # number of tasks
#SBATCH -t 00:06:00          # Run time (hh:mm:ss)

module purge

# Run Script
srun /lustre/miifs01/project/m2_jgu-amd/josealanis/envs/mne-1.5/bin/python 201_compute_signal_variability_measures.py --subject=$1 --session=1 --task='oddeven' --stimulus='cue' --window='post' --overwrite=True --jobs=64