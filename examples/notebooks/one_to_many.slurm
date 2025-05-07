#!/bin/bash
# remember to cd to /examples in the command line before running this script
# NOTE: if task dataset is changed, and running in parallel, run twice 
# bc it fails simultaneous writing the first time

#SBATCH --job-name=Mgnode     # create a short name for your job
#SBATCH --output=out/%x_output.txt           # Standard output file
#SBATCH --error=out/%x_error.txt             # Standard error file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1             # Number of gpus per node
#SBATCH --mem-per-cpu=150G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu

module purge
module load anaconda3/2024.6
conda activate ctd

python one_to_many_dsa.py 
