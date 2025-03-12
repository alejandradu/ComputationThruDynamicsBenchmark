#!/bin/bash
# remember to cd to /examples in the command line before running this script
# NOTE: if task dataset is changed, and running in parallel, run twice 
# bc it fails simultaneous writing the first time

#SBATCH --job-name=TEST_D_3bff_node_node     # create a short name for your job
#SBATCH --output=out/output5.txt           # Standard output file
#SBATCH --error=out/error5.txt             # Standard error file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gpus-per-node=1             # Number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu
#SBATCH --partition=MIG

module purge
module load anaconda3/2024.6
conda activate ctd

python /home/ad2002/ComputationThruDynamicsBenchmark/examples/data_training/node_on_3bff_node.py
