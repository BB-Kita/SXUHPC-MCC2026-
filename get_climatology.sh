#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4  
#SBATCH --exclusive
#SBATCH --gres=dcu:1  
export PATH=/public/home/achwjznh4b/install//bin:$PATH
time matlab -nodisplay -nosplash -nodesktop < get_climatology.m
