from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.analysis.dd.dd import Analysis_DD
from sklearn.manifold import MDS 
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import logging
from datetime import datetime
import gc
from sklearn.decomposition import PCA

def setup_logging(log_file=None):
    """Set up logging to both console and file if specified"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # If no log file specified, create one with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"dsa_analysis_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare consistent latent size matrices for DSA')
    parser.add_argument('--latent_sizes', nargs='+', type=int, default=[2, 3, 5, 10],
                        help='Latent sizes to analyze with PCA reduction')
    parser.add_argument('--percent_data', type=float, default=0.10,
                        help='Percentage of data to use for calculation')
    args = parser.parse_args()
    
    # Extract arguments
    PERCENT_DATA = args.percent_data
    LATENT_SIZES = args.latent_sizes
    OUTPUT_PREFIX = "latents_for_dsa"
    
    # all model paths
    TT_PATHS = ["/scratch/gpfs/ad2002/content/trained_models/task-trained/20250407_PC_NODE_grid_final/max_epochs=1500_weight_decay=1.00E-08_learning_rate=1.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=128_delta_t=1.00E-02_alpha=5.00E-02_leak=True",
           "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250413_PClicks_NoisyGRU_final2/max_epochs=1500_weight_decay=1.00E-04_learning_rate=1.00E-04_noise=5.00E-04_seed=0_latent_size=128_delta_t=1.00E-02_latent_ic_var=5.00E-03_l2_wt=1.00E-05_noise_level=5.00E-03",
           "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250412_PC_gNODE_sweep/max_epochs=1500_weight_decay=1.00E-06_learning_rate=2.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=64_delta_t=1.00E-02_alpha=1.00E-01_leak=True"]

    DD_PATHS = [
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_NODE",
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_GRU",
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_gNODE"
    ]
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_{timestamp}.log")
    logger.info(f"  PERCENT_DATA: {PERCENT_DATA}")
    logger.info(f"  LATENT_SIZES: {LATENT_SIZES}")
    
    # Track success and failures
    success_count = 0
    failure_count = 0
    
    # Lists to collect all latents and their labels
    all_lats = {}   # key: "model_size"
    all_hashes = {}  # 0 hash indicates TT true
    
    logger.info("Collecting TT latents")
    
    # Initialize path_success counter for TT
    tt_path_success = 0
    
    for i, path in enumerate(TT_PATHS):
        model_type = ["node", "gru", "gnode"][i]
        logger.info(f"Processing TT path for {model_type}: {path}")
        
        try:
            analysisTT = Analysis_TT(run_name="tt", filepath=path + "/")
            latents = analysisTT.get_latents(phase="val").detach().cpu().numpy()
            true_dim = latents.shape[-1]
            logger.info(f"Extracted TT latents with shape {latents.shape} (true dim: {true_dim})")
            
            # Process for each requested latent size
            for size in LATENT_SIZES:
                if true_dim >= size:
                    # Ensure latents is properly shaped for PCA
                    orig_shape = latents.shape
                    reshaped_latents = latents.reshape(-1, true_dim)
                    
                    # Apply PCA
                    pca = PCA(n_components=size)
                    reduced_latents = pca.fit_transform(reshaped_latents)
                    
                    # Reshape back maintaining batch structure
                    if len(orig_shape) > 2:
                        reduced_latents = reduced_latents.reshape(orig_shape[0], orig_shape[1], size)
                    else:
                        reduced_latents = reduced_latents.reshape(-1, size)
                    
                    # Take only the requested percentage
                    data_size = int(PERCENT_DATA * reduced_latents.shape[0])
                    sample_latents = reduced_latents[:data_size]
                    
                    # Store the reduced latents and corresponding label
                    key = f"{model_type}{size}"
                    all_lats[key] = [sample_latents]
                    all_hashes[key] = [0]
                    
                    logger.info(f"Added TT latents for {key}")

            # Clean up
            del analysisTT
            del latents
            gc.collect()

            success_count += 1
            tt_path_success += 1
        
        except Exception as e:
            logger.error(f"Failed to process TT path {path}: {str(e)}")
            failure_count += 1
            
    logger.info(f"Completed processing TT paths. Success count: {tt_path_success}") 
    
    # load the hash tables
    hash_node = "hash_tables/final_hash_table_from_node.csv"
    hash_gru = "hash_tables/final_hash_table_from_gru.csv"
    hash_gnode = "hash_tables/final_hash_table_from_gnode.csv"
    
    try:
        hash_table_node = pd.read_csv(hash_node)
        hash_table_gru = pd.read_csv(hash_gru)
        hash_table_gnode = pd.read_csv(hash_gnode)
        logger.info("Successfully loaded hash tables")
    except Exception as e:
        logger.error(f"Error loading hash tables: {str(e)}")
        return
                    
    
    for j, path in enumerate(DD_PATHS):
        
        if j == 0:
            hash_table = hash_table_node
        elif j == 1:
            hash_table = hash_table_gru
        else:
            hash_table = hash_table_gnode

        logger.info(f"Processing DD path: {path}")
        
        # Track successful models per path
        path_success = 0
        
        for run_dir in os.scandir(path):
            if not run_dir.is_dir():
                continue
            
            logger.info(f"Processing run directory: {run_dir.name}")
            
            # For each DT_...
            for tune_dir in os.scandir(os.path.join(path, run_dir)):
                if not tune_dir.is_dir():
                    continue
                
                tune_dir_path = os.path.join(path, run_dir.name, tune_dir.name)
                tune_dir_name = tune_dir.name
                
                # Skip if not a DT_ directory
                if not tune_dir_name.startswith("DT_"):
                    continue
                
                # all subdirs here have the format DT_...
                hashname = tune_dir_name[3:] 
                
                # Check if hashname exists in hash table
                if not any(hash_table["hashname"] == hashname):
                    logger.warning(f"Hashname {hashname} not found in hash table, skipping")
                    continue
                
                # get the value in the df with that hashname and get the model type
                model = hash_table[hash_table["hashname"] == hashname]["model"].values[0]
                
                try:
                    # Create analysis object
                    logger.info(f"Processing tune directory: {tune_dir_name} (model: {model})")
                    analysisDD = Analysis_DD.create(
                        run_name=tune_dir_name,
                        filepath=tune_dir_path + "/",
                        model_type="SAE"
                    )
                    
                    # Get latents
                    latents = analysisDD.get_latents(phase="val").detach().cpu().numpy()
                    true_dim = latents.shape[-1]
                    logger.info(f"Extracted latents with shape {latents.shape} (true dim: {true_dim})")
                    
                    # Do PCA reduction for each target size
                    for size in LATENT_SIZES:
                        if true_dim >= size:
                            # Ensure latents is properly shaped for PCA
                            orig_shape = latents.shape
                            reshaped_latents = latents.reshape(-1, true_dim)
                            
                            # Apply PCA
                            pca = PCA(n_components=size)
                            reduced_latents = pca.fit_transform(reshaped_latents)
                            
                            # Reshape back maintaining batch structure
                            if len(orig_shape) > 2:
                                reduced_latents = reduced_latents.reshape(orig_shape[0], orig_shape[1], size)
                            else:
                                reduced_latents = reduced_latents.reshape(-1, size)
                            
                            # Take only the requested percentage
                            data_size = int(PERCENT_DATA * reduced_latents.shape[0])
                            sample_latents = reduced_latents[:data_size]
                            
                            # Store the reduced latents and corresponding label
                            key = f"{model}{size}"
                            if key not in all_lats:
                                all_lats[key] = [sample_latents]
                                all_hashes[key] = [hashname]
                            else:
                                all_lats[key].append(sample_latents)
                                all_hashes[key].append(hashname)
                            
                            logger.info(f"Added latents for {key} with hashname {hashname}")
                    
                    # Clean up
                    del analysisDD
                    del latents
                    gc.collect()
                    
                    success_count += 1
                    path_success += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {tune_dir_name}: {str(e)}")
                    failure_count += 1
        
        logger.info(f"Completed processing path {path}. Success count: {path_success}")
    
    # save the finalized dict
    logger.info("Saving latents and hash data to disk")
    with open("lats_for_dsa.pkl", "wb") as f:
        pickle.dump(all_lats, f)
        
    with open("hashes_for_dsa.pkl", "wb") as f:
        pickle.dump(all_hashes, f)
        
    logger.info(f"Total success count: {success_count}")
    logger.info(f"Total failure count: {failure_count}")


if __name__ == "__main__":
    main()