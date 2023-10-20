#!/bin/bash

#SBATCH -A m2_jgu-amd               # Account
#SBATCH -p parallel                 # Partition: parallel, smp, bigmem
#SBATCH -C skylake                  # architecture Skylake (64 Cores) or Broad$
##SBATCH -N 1                       # number of tasks
#SBATCH -c 14                       # number of cores
#SBATCH -t 00:15:00                 # Run time (hh:mm:ss)

# do not forget to export OMP_NUM_THREADS, if the library you use, supports this
# not scale up to 64 threadsq
#export OMP_NUM_THREADS=64

module purge # ensures vanilla environment
module load lang/R # will load most current version of R

# do not forget to export OMP_NUM_THREADS, if the library you use, supports this
# not scale up to 64 threadsq
#export OMP_NUM_THREADS=64

echo "------child bash arguments------"
echo $1

echo "------end of bash output------"

srun Rscript 302_signal_variability_analysis_single_trial.R \
  --sensor_n $1 --task_i "Odd/Even" --jobs 14

echo "------job is finished------"
