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
import torch
import time
import random
from collections import defaultdict

"""
Script to perform DSA computations using cached latents and DMD matrices,
focusing only on similarity calculations that haven't been done yet
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

def clear_gpu_memory():
    """Thoroughly clear GPU memory"""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Additional steps to ensure memory is freed
        # Force garbage collection
        gc.collect()
        
        # Wait a small amount of time to ensure memory is properly released
        time.sleep(0.5)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate DSA similarity scores from cached DMD matrices')
    parser.add_argument('--output_prefix', type=str, default="dsa_results",
                        help='Prefix for output files')
    parser.add_argument('--force_recompute', action='store_true',default=False,
                        help='Force recomputation of similarity values even if cached')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.5,
                        help='Fraction of GPU memory to use (0.0-1.0)')
    parser.add_argument('--max_samples_per_type', type=int, default=50,
                        help='Maximum number of samples per model type to include in analysis')
    parser.add_argument('--reset_device_interval', type=int, default=10,
                        help='Number of similarity calculations after which to reset the GPU device')
    args = parser.parse_args()
    
    # Extract arguments
    OUTPUT_PREFIX = args.output_prefix
    FORCE_RECOMPUTE = args.force_recompute
    GPU_MEMORY_FRACTION = args.gpu_memory_fraction
    MAX_SAMPLES_PER_TYPE = args.max_samples_per_type
    RESET_DEVICE_INTERVAL = args.reset_device_interval
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_sim_{timestamp}.log")
    
    logger.info(f"Starting DSA similarity analysis with parameters:")
    logger.info(f"  OUTPUT_PREFIX: {OUTPUT_PREFIX}")
    logger.info(f"  FORCE_RECOMPUTE: {FORCE_RECOMPUTE}")
    logger.info(f"  GPU_MEMORY_FRACTION: {GPU_MEMORY_FRACTION}")
    logger.info(f"  MAX_SAMPLES_PER_TYPE: {MAX_SAMPLES_PER_TYPE}")
    logger.info(f"  RESET_DEVICE_INTERVAL: {RESET_DEVICE_INTERVAL}")
    
    # Define paths for cache directories
    dmd_cache_dir = f"{OUTPUT_PREFIX}_dmd_cache"
    sim_cache_dir = f"{OUTPUT_PREFIX}_sim_cache"
    
    # Ensure cache directories exist
    os.makedirs(dmd_cache_dir, exist_ok=True)
    os.makedirs(sim_cache_dir, exist_ok=True)
    
    # Load DMD matrices from cache
    logger.info(f"Loading DMD matrices from cache directory: {dmd_cache_dir}")
    
    # Find all DMD cache files
    dmd_files = [f for f in os.listdir(dmd_cache_dir) if f.endswith("_dmd.npy")]
    logger.info(f"Found {len(dmd_files)} DMD cache files")
    
    if len(dmd_files) == 0:
        logger.error("No DMD cache files found. Please run the DMD computation first.")
        return
    
    # Group DMD files by model type
    model_type_files = defaultdict(list)
    for dmd_file in dmd_files:
        label = dmd_file.replace("_dmd.npy", "")
        parts = label.split('_')
        model_type = parts[0]
        if int(parts[1]) == 2:
            model_type_files[model_type].append(dmd_file)
    
    # Limit number of samples per model type
    logger.info(f"Limiting to max {MAX_SAMPLES_PER_TYPE} samples per model type")
    limited_dmd_files = []
    
    for model_type, files in model_type_files.items():
        # If we have more files than the limit, select a random subset
        if len(files) > MAX_SAMPLES_PER_TYPE:
            # Prioritize 'true' models first if they exist
            true_models = [f for f in files if "true" in f]
            other_models = [f for f in files if "true" not in f]
            
            # Take all true models if there are fewer than MAX_SAMPLES_PER_TYPE
            if len(true_models) <= MAX_SAMPLES_PER_TYPE:
                selected_true = true_models
                # Fill remaining slots with randomly selected other models
                remaining_slots = MAX_SAMPLES_PER_TYPE - len(selected_true)
                selected_other = random.sample(other_models, min(remaining_slots, len(other_models)))
                selected_files = selected_true + selected_other
            else:
                # If we have more true models than the limit, randomly select from them
                selected_files = random.sample(true_models, MAX_SAMPLES_PER_TYPE)
        else:
            selected_files = files
            
        limited_dmd_files.extend(selected_files)
        logger.info(f"Model type {model_type}: selected {len(selected_files)} out of {len(files)} files")
    
    # Load the limited set of DMD matrices
    dmds = []
    valid_labels = []
    valid_model_types = []
    valid_hash_values = []
    
    for dmd_file in limited_dmd_files:
        try:
            # Extract label from filename
            label = dmd_file.replace("_dmd.npy", "")
            
            # Extract model type and hash value from label
            parts = label.split('_')
            if len(parts) >= 3:
                model_type = parts[0]
                hash_val = parts[2]
            else:
                model_type = parts[0]
                hash_val = "unknown"
            
            # Load DMD matrix
            dmd_path = os.path.join(dmd_cache_dir, dmd_file)
            dmd_matrix = np.load(dmd_path)
            
            # Convert to float32 if in float64 to save memory
            if dmd_matrix.dtype == np.float64:
                dmd_matrix = dmd_matrix.astype(np.float32)
            
            # Add to collections
            dmds.append(dmd_matrix)
            valid_labels.append(label)
            valid_model_types.append(model_type)
            valid_hash_values.append(hash_val)
            
            logger.info(f"Loaded DMD for {label}, shape {dmd_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error loading DMD file {dmd_file}: {str(e)}")
    
    # Calculate similarity matrix
    total_models = len(dmds)
    logger.info(f"Calculating similarity matrix for {total_models} models")
    
    if total_models == 0:
        logger.error("No valid DMD matrices loaded. Exiting.")
        return
    
    # Save labels
    np.save(f'{OUTPUT_PREFIX}_labels.npy', np.array(valid_labels))
    
    # Define output paths for similarity matrices
    sim_dmd_path = f'{OUTPUT_PREFIX}_sims_dmd.npy'
    sim_model_type_path = f'{OUTPUT_PREFIX}_sims_model_type.npy'
    sim_hash_value_path = f'{OUTPUT_PREFIX}_sims_hash_value.npy'
    
    # Initialize similarity matrices
    sims_dmd = np.zeros((total_models, total_models))
    sims_model_type = np.zeros((total_models, total_models))
    sims_hash_value = np.zeros((total_models, total_models))
    
    # Check if we have CUDA support
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        logger.info("CUDA available, using GPU for computations")
        # Print GPU information
        device_props = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {device_props.name}")
        logger.info(f"Total GPU memory: {device_props.total_memory / 1e9:.2f} GB")
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    else:
        logger.warning("CUDA not available, using CPU (this will be slow)")
    
    # Initialize our device
    device = 'cuda' if has_cuda else 'cpu'
    
    # Track computation count to know when to reset the device
    computation_count = 0
    
    # Calculate pairwise similarities
    total_pairs = (total_models * (total_models - 1)) // 2
    completed_pairs = 0
    start_time = time.time()
    
    comparison_dmd = SimilarityTransformDist(device=device, iters=1000, lr=1e-3)
    
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
            
            # Set diagonal to 1.0 (self-similarity)
            if i == j:
                sims_model_type[i, i] = 1.0
                sims_hash_value[i, i] = 1.0
                sims_dmd[i, i] = 1.0
                continue
            
            # Skip redundant calculations (matrix is symmetric)
            if j < i:
                sims_dmd[i, j] = sims_dmd[j, i]
                continue
            
            # Check if we have a cached similarity result
            sim_cache_file = os.path.join(sim_cache_dir, f"sim_{valid_labels[i]}_{valid_labels[j]}.npy")
            
            if os.path.exists(sim_cache_file) and not FORCE_RECOMPUTE:
                try:
                    sdmd = np.load(sim_cache_file)[0]
                    sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                    logger.info(f"Loaded cached similarity between {valid_labels[i]} and {valid_labels[j]}: {sdmd:.4f}")
                    completed_pairs += 1
                    continue
                except Exception as e:
                    logger.error(f"Error loading cached similarity: {str(e)}. Will recompute.")
            
            # # Check if we need to reset the device
            # if computation_count >= RESET_DEVICE_INTERVAL and has_cuda:
            #     logger.info(f"Reached {computation_count} computations. Resetting device...")
            #     # Delete existing comparison object
            #     if 'comparison_dmd' in locals():
            #         del comparison_dmd
                
            #     # Clear GPU memory
            #     clear_gpu_memory()
                
            #     # Reset computation count
            #     computation_count = 0
                
            #     # Create a new comparison object
            #     logger.info("Creating new SimilarityTransformDist object")
            
            # Initialize a fresh similarity transform object for each calculation
            # This prevents memory accumulation from previous calculations
            # comparison_dmd = SimilarityTransformDist(device=device, iters=2000, lr=1e-3)
            
            # Calculate DSA similarity
            logger.info(f"Calculating similarity between {valid_labels[i]} and {valid_labels[j]}")
            try:
                # # Make sure GPU memory is cleared before calculation
                # if has_cuda:
                #     clear_gpu_memory()
                
                # Calculate similarity
                sdmd = comparison_dmd.fit_score(dmds[i], dmds[j])
                
                # # Explicitly delete the comparison object to free resources
                # del comparison_dmd
                
                # Cache the similarity value
                try:
                    np.save(sim_cache_file, np.array([sdmd]))
                except Exception as e:
                    logger.error(f"Error caching similarity value: {str(e)}")
                
                # Fill both entries in the symmetric matrix
                sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                logger.info(f"Similarity between {valid_labels[i]} and {valid_labels[j]}: {sdmd:.4f}")
                
                # Update progress tracking
                completed_pairs += 1
                # computation_count += 1
                
                # Save intermediate results periodically (every 10 pairs)
                if completed_pairs % 50 == 0:
                    np.save(sim_dmd_path, sims_dmd)
                    np.save(sim_model_type_path, sims_model_type)
                    np.save(sim_hash_value_path, sims_hash_value)
                    logger.info("Saved intermediate similarity matrices")
                    
                    elapsed = time.time() - start_time
                    pairs_per_second = completed_pairs / elapsed if elapsed > 0 else 0
                    remaining_pairs = total_pairs - completed_pairs
                    eta_seconds = remaining_pairs / pairs_per_second if pairs_per_second > 0 else 0

                    logger.info(f"Progress: {completed_pairs}/{total_pairs} pairs ({100*completed_pairs/total_pairs:.1f}%)")
                    logger.info(f"Speed: {pairs_per_second:.2f} pairs/sec, ETA: {eta_seconds/60:.1f} minutes")
                
                # Force garbage collection and clear CUDA cache after each computation
                gc.collect()
                if has_cuda:
                    clear_gpu_memory()
                
            except Exception as e:
                logger.error(f"Error calculating similarity between {valid_labels[i]} and {valid_labels[j]}: {str(e)}")
                # Try with CPU as fallback if GPU fails
                if device == 'cuda':
                    logger.info("Retrying with CPU...")
                    try:
                        cpu_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
                        sdmd = cpu_comparison.fit_score(dmds[i], dmds[j])
                        sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                        np.save(sim_cache_file, np.array([sdmd]))
                        logger.info(f"CPU retry successful: {sdmd:.4f}")
                        completed_pairs += 1
                        del cpu_comparison
                    except Exception as e2:
                        logger.error(f"CPU retry also failed: {str(e2)}")
                        sims_dmd[i, j] = sims_dmd[j, i] = np.nan
                else:
                    sims_dmd[i, j] = sims_dmd[j, i] = np.nan
    
    # Save final similarity matrices and labels
    logger.info("Saving final outputs")
    
    np.save(sim_dmd_path, sims_dmd)
    np.save(sim_model_type_path, sims_model_type)
    np.save(sim_hash_value_path, sims_hash_value)
    
    # Handle any NaN values in the similarity matrix
    if np.any(np.isnan(sims_dmd)):
        logger.warning(f"Found {np.sum(np.isnan(sims_dmd))} NaN values in similarity matrix")
        # Replace NaN with mean of non-NaN elements in each row
        for i in range(sims_dmd.shape[0]):
            row = sims_dmd[i, :]
            if np.any(np.isnan(row)):
                # Get mean of non-NaN elements
                valid_values = row[~np.isnan(row)]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                else:
                    mean_val = 0.5  # Default value if all are NaN
                
                # Replace NaN with mean
                row[np.isnan(row)] = mean_val
                sims_dmd[i, :] = row
                
        # Ensure symmetry after fixing NaN values
        for i in range(sims_dmd.shape[0]):
            for j in range(i+1, sims_dmd.shape[1]):
                avg_val = (sims_dmd[i, j] + sims_dmd[j, i]) / 2
                sims_dmd[i, j] = sims_dmd[j, i] = avg_val
                
        # Save fixed matrix
        np.save(f'{OUTPUT_PREFIX}_sims_dmd_fixed.npy', sims_dmd)
        logger.info("Saved fixed similarity matrix without NaN values")
    
    # Create MDS embedding for visualization
    logger.info("Creating MDS embedding")
    try:
        # Convert to distance matrix (1 - similarity)
        dissim_matrix = 1 - sims_dmd
        
        # Perform MDS
        lowd_embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dissim_matrix)
        
        # Create DataFrame for visualization
        df = pd.DataFrame()
        df['Model_Type'] = valid_model_types
        df['Model_Label'] = valid_labels
        df['Hash_Value'] = valid_hash_values
        df['DMD:0'] = lowd_embedding[:, 0]
        df['DMD:1'] = lowd_embedding[:, 1]
        
        # Extract model type and latent size as separate columns
        df['Architecture'] = [label.split('_')[0] for label in valid_labels]
        df['Latent_Size'] = [int(label.split('_')[1]) if len(label.split('_')) > 1 and label.split('_')[1].isdigit() 
                           else 0 for label in valid_labels]
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
        'is_true_model': [hash_val == "true" for hash_val in valid_hash_values]
    })
    
    # Try to extract latent size from labels
    try:
        metadata_df['reduced_dimension'] = [int(label.split('_')[1]) if len(label.split('_')) > 1 and label.split('_')[1].isdigit() 
                                         else 0 for label in valid_labels]
    except Exception as e:
        logger.warning(f"Could not parse latent size from labels: {str(e)}")
        metadata_df['reduced_dimension'] = 0
    
    metadata_df.to_csv(f"{OUTPUT_PREFIX}_metadata.csv", index=False)
    logger.info(f"Model metadata saved to {OUTPUT_PREFIX}_metadata.csv")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"Analysis complete. Processed {total_models} models with {completed_pairs} similarity computations.")
    logger.info(f"Total computation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per similarity: {total_time/completed_pairs:.4f} seconds")
    logger.info(f"Results saved with prefix: {OUTPUT_PREFIX}")

if __name__ == "__main__":
    main()