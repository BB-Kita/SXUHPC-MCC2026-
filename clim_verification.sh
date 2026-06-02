#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4  
#SBATCH --exclusive 节点独占
#SBATCH --gres=dcu:1  
#SBATCH --time=02:00:00  
export PATH=/public/home/achwjznh4b/install//bin:$PATH
time matlab -nodisplay -nosplash -nodesktop < clim_verification.m
