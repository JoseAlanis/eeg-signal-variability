#!/bin/bash

declare -ar sensors=($(seq 0 1))


# submit job for every permutation
# we keep track:
njobs=0

# Guess the account
account=$(sacctmgr -n -s list user $USER format=account%30| grep -v none | head -n1 | tr -d " ")

##Spawn Jobs for each condition combination

    for N in "${sensors[@]}"; do
                    jobname="R_signal_var_subject_N${N}"
                    slurmout="jobs/${jobname}.%j.out"
                    #echo $slurmout
                    if [[ ! -e ${slurmout} ]]; then
                         sbatch -A "$account" -J "$jobname" -o "$slurmout" slaveR.sh "$N"
                         njobs=$((njobs + 1))

                     fi
            done
