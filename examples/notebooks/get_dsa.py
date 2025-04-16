from DSA import DSA
from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
from sklearn.manifold import MDS 
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import logging
from datetime import datetime
import gc

"""
Script to perform DSA computations on pre-processed latent data
"""

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

def fit_dmd(x, n_delays, rank, delay_interval):
    """
    Fit DMD to the provided data
    
    Parameters:
    -----------
    x : ndarray 
        Input data with shape (B, T, N) where B is batch size, T is timesteps, N is dimensions
    n_delays : int
        Number of delays for DMD
    rank : int
        Rank for DMD
    delay_interval : int
        Delay interval for DMD
        
    Returns:
    --------
    ndarray
        DMD matrix A_v
    """
    logger = logging.getLogger()
    
    # Ensure x has shape (B, T, N)
    original_shape = x.shape
    if len(original_shape) != 3:
        logger.warning(f"Reshaping input from {original_shape} to (B, T, N) format")
        if len(original_shape) == 2:
            # If 2D, assume (T, N) and add batch dimension
            x = x.reshape(1, *original_shape)
        else:
            # More complex reshaping needed
            raise ValueError(f"Cannot automatically reshape input with shape {original_shape} to (B, T, N) format")
    
    logger.info(f"Fitting DMD with data shape {x.shape}, n_delays={n_delays}, rank={rank}, delay_interval={delay_interval}")
    
    try:
        dmd = DMD(x, n_delays=n_delays, rank=rank, delay_interval=delay_interval, device='cuda', send_to_cpu=True)
        dmd.fit()
        return dmd.A_v.numpy()
    except Exception as e:
        logger.error(f"Error fitting DMD: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate DSA scores from pre-processed latent data')
    parser.add_argument('--n_delays', type=int, default=20,
                        help='Number of delays for DMD calculation')
    parser.add_argument('--rank', type=int, default=50,
                        help='Rank for DMD calculation')
    parser.add_argument('--delay_interval', type=int, default=1,
                        help='Delay interval for DMD calculation')
    parser.add_argument('--output_prefix', type=str, default="dsa_results",
                        help='Prefix for output files')
    parser.add_argument('--latents_file', type=str, default="lats_for_dsa.pkl",
                        help='Path to pickled latents file')
    parser.add_argument('--hashes_file', type=str, default="hashes_for_dsa.pkl",
                        help='Path to pickled hashes file')
    args = parser.parse_args()
    
    # Extract arguments
    N_DELAYS = args.n_delays
    RANK = args.rank
    DELAY_INTERVAL = args.delay_interval
    OUTPUT_PREFIX = args.output_prefix
    LATENTS_FILE = args.latents_file
    HASHES_FILE = args.hashes_file
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_{timestamp}.log")
    
    logger.info(f"Starting DSA analysis with parameters:")
    logger.info(f"  N_DELAYS: {N_DELAYS}")
    logger.info(f"  RANK: {RANK}")
    logger.info(f"  DELAY_INTERVAL: {DELAY_INTERVAL}")
    logger.info(f"  LATENTS_FILE: {LATENTS_FILE}")
    logger.info(f"  HASHES_FILE: {HASHES_FILE}")
    
    # Load the pre-processed latent data and hash information
    logger.info(f"Loading pre-processed latent data from {LATENTS_FILE}")
    try:
        with open(LATENTS_FILE, "rb") as f:
            all_lats = pickle.load(f)
        
        logger.info(f"Loading hash information from {HASHES_FILE}")
        with open(HASHES_FILE, "rb") as f:
            all_hashes = pickle.load(f)
            
        logger.info(f"Successfully loaded data: {len(all_lats)} model-size combinations")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Prepare data structures for DMD processing
    all_latents = []  # Will store all latent arrays
    all_labels = []   # Will store corresponding labels
    all_model_types = []  # Will store model types (node, gru, gnode)
    all_hash_values = []  # Will store hash values
    
    # Process each model-size combination
    for key, latent_list in all_lats.items():
        # Extract model type and size from key (format: "model_type" + "size")
        model_type = ''.join([c for c in key if not c.isdigit()])
        size = ''.join([c for c in key if c.isdigit()])
        
        # Process each latent in the list
        for i, latent in enumerate(latent_list):
            # Create a unique label
            if key in all_hashes and isinstance(all_hashes[key], list) and len(all_hashes[key]) > i:
                # DD model with hash
                hash_value = all_hashes[key][i]
                label = f"{model_type}_{size}_{hash_value}"
                hash_val = hash_value
            else:
                # TT model (true model)
                label = f"{model_type}_{size}_true"
                hash_val = "true"
            
            all_latents.append(latent)
            all_labels.append(label)
            all_model_types.append(model_type)
            all_hash_values.append(hash_val)
            
            logger.info(f"Added latent with label {label}, shape {latent.shape}")
    
    # Initialize lists to store DMD matrices
    dmds = []
    valid_indices = []  # To keep track of which latents were successfully processed
    valid_labels = []
    
    # Fit DMD to each set of latents
    logger.info(f"Fitting DMD for {len(all_latents)} latent arrays")
    
    for i, latents in enumerate(all_latents):
        logger.info(f"Fitting DMD for model {i+1}/{len(all_latents)}: {all_labels[i]}")
        
        try:
            dmd_matrix = fit_dmd(
                latents, 
                n_delays=N_DELAYS, 
                rank=RANK, 
                delay_interval=DELAY_INTERVAL
            )
            
            if dmd_matrix is not None:
                dmds.append(dmd_matrix)
                valid_indices.append(i)
                valid_labels.append(all_labels[i])
                logger.info(f"Successfully fit DMD for {all_labels[i]}")
            else:
                logger.warning(f"DMD fit returned None for {all_labels[i]}")
        
        except Exception as e:
            logger.error(f"Error fitting DMD for {all_labels[i]}: {str(e)}")
        
        # Force garbage collection
        gc.collect()
    
    # Filter labels to only include successfully processed ones
    valid_model_types = [all_model_types[i] for i in valid_indices]
    valid_hash_values = [all_hash_values[i] for i in valid_indices]
    
    # Calculate similarity matrix
    total_models = len(dmds)
    logger.info(f"Calculating similarity matrix for {total_models} models")
    
    sims_dmd = np.zeros((total_models, total_models))
    sims_model_type = np.zeros((total_models, total_models))
    sims_hash_value = np.zeros((total_models, total_models))
    
    # Initialize similarity transform
    comparison_dmd = SimilarityTransformDist(device='cuda', iters=2000, lr=1e-3)
    
    # Calculate pairwise similarities
    for i in range(total_models):
        for j in range(total_models):
            # Calculate model type similarity (1 if same, 0 if different)
            model_type_i = valid_model_types[i]
            model_type_j = valid_model_types[j]
            smtype = int(model_type_i == model_type_j)
            sims_model_type[i, j] = smtype
            
            # Calculate hash value similarity (1 if same, 0 if different)
            hash_i = valid_hash_values[i]
            hash_j = valid_hash_values[j]
            shash = int(hash_i == hash_j)
            sims_hash_value[i, j] = shash
            
            # Set diagonal to 2 (special value for self-similarity)
            if i == j:
                sims_model_type[i, i] = 2
                sims_hash_value[i, i] = 2
                sims_dmd[i, i] = 1.0  # Self-similarity is 1.0
                continue
            
            # Skip redundant calculations (matrix is symmetric)
            if j < i:
                continue
            
            # Calculate DSA similarity
            logger.info(f"Calculating similarity between {valid_labels[i]} and {valid_labels[j]}")
            try:
                sdmd = comparison_dmd.fit_score(dmds[i], dmds[j])      
                # Fill both entries in the symmetric matrix
                sims_dmd[i, j] = sims_dmd[j, i] = sdmd
            except Exception as e:
                logger.error(f"Error calculating similarity between {valid_labels[i]} and {valid_labels[j]}: {str(e)}")
                sims_dmd[i, j] = sims_dmd[j, i] = np.nan
    
    # Save similarity matrices and labels
    logger.info("Saving outputs")
    
    # Save as numpy arrays for easier loading
    np.save(f'{OUTPUT_PREFIX}_sims_dmd.npy', sims_dmd)
    np.save(f'{OUTPUT_PREFIX}_sims_model_type.npy', sims_model_type)
    np.save(f'{OUTPUT_PREFIX}_sims_hash_value.npy', sims_hash_value)
    np.save(f'{OUTPUT_PREFIX}_labels.npy', np.array(valid_labels))
    
    # Create MDS embedding for visualization
    logger.info("Creating MDS embedding")
    try:
        # Create dissimilarity matrix (1 - similarity)
        # Handle any NaN values by replacing with mean
        dissim_matrix = np.where(np.isnan(sims_dmd), np.nanmean(sims_dmd), sims_dmd)
        # Convert to distances (1 - similarity)
        dissim_matrix = 1 - dissim_matrix
        
        # Perform MDS
        lowd_embedding = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dissim_matrix)
        
        # Create DataFrame for visualization
        df = pd.DataFrame()
        df['Model_Type'] = valid_model_types
        df['Model_Label'] = valid_labels
        df['Hash_Value'] = valid_hash_values
        df['DMD:0'] = lowd_embedding[:, 0]
        df['DMD:1'] = lowd_embedding[:, 1]
        
        # Extract model type and latent size as separate columns
        df['Architecture'] = [label.split('_')[0] for label in valid_labels]
        df['Latent_Size'] = [int(label.split('_')[1]) for label in valid_labels]
        df['Is_True_Model'] = [hash_val == "true" for hash_val in valid_hash_values]
        
        # Save DataFrame
        df.to_csv(f"{OUTPUT_PREFIX}_mds_embedding.csv", index=False)
        logger.info(f"MDS embedding saved to {OUTPUT_PREFIX}_mds_embedding.csv")
        
    except Exception as e:
        logger.error(f"Error creating MDS embedding: {str(e)}")
    
    # Create metadata DataFrame with information about each model
    metadata_df = pd.DataFrame({
        'model_label': valid_labels,
        'model_type': valid_model_types,
        'hash_value': valid_hash_values,
        'reduced_dimension': [int(label.split('_')[1]) for label in valid_labels],
        'is_true_model': [hash_val == "true" for hash_val in valid_hash_values]
    })
    
    metadata_df.to_csv(f"{OUTPUT_PREFIX}_metadata.csv", index=False)
    logger.info(f"Model metadata saved to {OUTPUT_PREFIX}_metadata.csv")
    
    # Final summary
    logger.info(f"Analysis complete. Processed {total_models} models successfully.")
    logger.info(f"Results saved with prefix: {OUTPUT_PREFIX}")

if __name__ == "__main__":
    main()