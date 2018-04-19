#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=download_blast_clusters
#SBATCH --mail-type=END
#SBATCH --mail-user=sanyam@nyu.edu
#SBATCH --output=logs/slurm_%j.out

cd $VLG_HOME/arives/folding-rl/neural-md

python download_bc.py
