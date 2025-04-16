from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.analysis.dd.dd import Analysis_DD
from DSA import DSA
from sklearn.manifold import MDS 
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import logging
from datetime import datetime
import gc
from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
from sklearn.decomposition import PCA


"""
Run as an alternative to loading all (many) DD models into the comparison
object and taking long to optimize the whole DSA object
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
        dmd = DMD(x, n_delays=n_delays, rank=rank, delay_interval=delay_interval, device='cuda')
        dmd.fit(send_to_cpu=True)
        return dmd.A_v.numpy()
    except Exception as e:
        logger.error(f"Error fitting DMD: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate DSA scores across model types')
    parser.add_argument('--n_delays', type=int, default=20,
                        help='Number of delays for DMD calculation')
    parser.add_argument('--rank', type=int, default=50,
                        help='Rank for DMD calculation')
    parser.add_argument('--delay_interval', type=int, default=1,
                        help='Delay interval for DMD calculation')
    parser.add_argument('--percent_data', type=float, default=0.10,
                        help='Percentage of data to use for calculation')
    parser.add_argument('--output_prefix', type=str, default="dsa_results",
                        help='Prefix for output files')
    parser.add_argument('--latent_sizes', nargs='+', type=int, default=[2, 3, 5, 10],
                        help='Latent sizes to analyze with PCA reduction')
    args = parser.parse_args()
    
    # Extract arguments
    N_DELAYS = args.n_delays
    RANK = args.rank
    DELAY_INTERVAL = args.delay_interval
    PERCENT_DATA = args.percent_data
    LATENT_SIZES = args.latent_sizes
    OUTPUT_PREFIX = args.output_prefix
    
    # Model paths
    DD_PATHS = [
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_NODE",
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_GRU",
        "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_gNODE"
    ]
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_{timestamp}.log")
    
    logger.info(f"Starting DSA analysis with parameters:")
    logger.info(f"  N_DELAYS: {N_DELAYS}")
    logger.info(f"  RANK: {RANK}")
    logger.info(f"  DELAY_INTERVAL: {DELAY_INTERVAL}")
    logger.info(f"  PERCENT_DATA: {PERCENT_DATA}")
    logger.info(f"  LATENT_SIZES: {LATENT_SIZES}")
    
    # Track success and failures
    success_count = 0
    failure_count = 0
    
    # Lists to collect all latents and their labels
    all_latents = []
    all_labels = []
    all_model_types = []
    all_original_dims = []
    
    # Collect all latents for each model
    model_labels = ['node', 'gru', 'gnode']
    
    logger.info("Collecting latents from all models")
    
    for j, path in enumerate(DD_PATHS):
        model_type = model_labels[j]
        logger.info(f"Processing path for model type {model_type}: {path}")
        
        # Track successful models per path
        path_success = 0
        
        for run_dir in os.scandir(path):
            if not run_dir.is_dir():
                continue
            
            # For each DT_...
            for tune_dir in os.scandir(os.path.join(path, run_dir)):
                if not tune_dir.is_dir():
                    continue
                
                tune_dir_path = os.path.join(path, run_dir.name, tune_dir.name)
                tune_dir_name = tune_dir.name
                
                try:
                    # Create analysis object
                    analysisDD = Analysis_DD.create(
                        run_name=tune_dir_name,
                        filepath=tune_dir_path + "/",
                        model_type="SAE"
                    )
                    
                    # Get latents
                    logger.info(f"Extracting latents from {tune_dir_name}")
                    latents = analysisDD.get_latents(phase="val").detach().cpu().numpy()
                    true_dim = latents.shape[-1]
                    
                    # Track original dimension
                    all_original_dims.append(true_dim)
                    
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
                            all_latents.append(sample_latents)
                            label = f"{model_type}_{size}"
                            all_labels.append(label)
                            all_model_types.append(model_type)
                            
                            logger.info(f"Added latents for {label} with shape {sample_latents.shape}")
                    
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
    
    logger.info(f"Completed collecting latents. Total success: {success_count}, Total failures: {failure_count}")
    logger.info(f"Total models to analyze: {len(all_latents)}")
    
    # Initialize lists to store DMD matrices
    dmds = []
    valid_indices = []  # To keep track of which latents were successfully processed
    valid_labels = []
    
    # Fit DMD to each set of latents
    
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
    
    # Calculate similarity matrix
    total_models = len(dmds)
    logger.info(f"Calculating similarity matrix for {total_models} models")
    
    sims_dmd = np.zeros((total_models, total_models))
    sims_model_type = np.zeros((total_models, total_models))
    
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
            
            # Set diagonal to 2 (special value for self-similarity)
            if i == j:
                sims_model_type[i, i] = 2
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
    
    # as numpy arrays for easier loading
    np.save(f'{OUTPUT_PREFIX}_sims_dmd.npy', sims_dmd)
    np.save(f'{OUTPUT_PREFIX}_sims_model_type.npy', sims_model_type)
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
        df['DMD:0'] = lowd_embedding[:, 0]
        df['DMD:1'] = lowd_embedding[:, 1]
        
        # Extract model type and latent size as separate columns
        df['Architecture'] = [label.split('_')[0] for label in valid_labels]
        df['Latent_Size'] = [int(label.split('_')[1]) for label in valid_labels]
        
        # Save DataFrame
        df.to_csv(f"{OUTPUT_PREFIX}_mds_embedding.csv", index=False)
        logger.info(f"MDS embedding saved to {OUTPUT_PREFIX}_mds_embedding.csv")
        
    except Exception as e:
        logger.error(f"Error creating MDS embedding: {str(e)}")
    
    # Create metadata DataFrame with information about each model
    metadata_df = pd.DataFrame({
        'model_label': valid_labels,
        'model_type': valid_model_types,
        'original_dimension': [all_original_dims[i] for i in valid_indices],
        'reduced_dimension': [int(label.split('_')[1]) for label in valid_labels]
    })
    
    metadata_df.to_csv(f"{OUTPUT_PREFIX}_metadata.csv", index=False)
    logger.info(f"Model metadata saved to {OUTPUT_PREFIX}_metadata.csv")
    
    # Final summary
    logger.info(f"Analysis complete. Processed {total_models} models successfully.")
    logger.info(f"Results saved with prefix: {OUTPUT_PREFIX}")

if __name__ == "__main__":
    main()