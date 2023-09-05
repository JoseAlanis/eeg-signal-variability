#!/bin/bash

#SBATCH -A m2_jgu-amd		 # Account
#SBATCH -p parallel		     # Partition: parallel, smp, bigmem
#SBATCH -C skylake 		     # architecture Skylake (64 Cores) or Broadwell (40 Cores)
#SBATCH -N 1                 # number of tasks
#SBATCH -t 00:02:00          # Run time (hh:mm:ss)

CONDA_ENV = "/lustre/miifs01/project/m2_jgu-amd/josealanis/envs/mne-1.5"

module purge
module add lang/Anaconda3
conda init bash
conda activate $CONDA_ENV

# Run Script
srun python 201_compute_signal_variability_measures.py --subj $i --session 1 --task 'oddeven' --stimulus 'cue' --window 'post' --overwrite True --jobs 64