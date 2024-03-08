#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:RTX_6000:1               # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=12G                       # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (8 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate fowm

echo "Running the following command:"
echo $@

srun $@