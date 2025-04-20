#!/bin/bash
# remember to cd to /examples in the command line before running this script
# NOTE: if task dataset is changed, and running in parallel, run twice 
# bc it fails simultaneous writing the first time

#SBATCH --job-name=dsaALL     # create a short name for your job
#SBATCH --output=out/%x_output.txt           # Standard output file
#SBATCH --error=out/%x_error.txt             # Standard error file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1             # Number of gpus per node
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu
#SBATCH --partition=MIG

module purge
module load anaconda3/2024.6
conda activate ctd

# python get_dsa.py --latent_size 2 --dmd_batch_size 4 --sim_batch_size 32 --num_workers 4 --num_streams 8

# python get_dsa.py --latent_size 2 --num_workers 4

# python get_dsa_sim_mds.py --output_prefix dsa_results --num_gpus 1

# python get_dsa_sim_mds.py --num_workers 15

# python compute_similarity_simple.py --num_workers 8

# python get_dsa_sim_mds.py --output_prefix dsa_results --num_workers 8

# python get_dsa_sim_mds.py --output_prefix dsa_results

python get_dsa_sim_mds.py --max_samples_per_type 50 --reset_device_interval 10 --gpu_memory_fraction 1.0
