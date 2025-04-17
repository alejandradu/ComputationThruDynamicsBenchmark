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
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from functools import partial
from tqdm import tqdm
import time

"""
Optimized script to perform DSA computations on pre-processed latent data with 
incremental saving, batch processing, and parallel computation capabilities
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

def fit_dmd(x, n_delays, rank, delay_interval, device='cuda', sync=True):
    """
    Fit DMD to the provided data with optimized memory management
    
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
    device : str
        Device to use for computation ('cuda' or 'cpu')
    sync : bool
        Whether to synchronize CUDA operations
        
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
        # Clear GPU cache before computation
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Use smaller precision to reduce memory usage if possible
        if x.dtype == np.float64:
            x = x.astype(np.float32)
            
        # Make a copy of the data to avoid modifying the original
        x_copy = x.copy()
        
        # Create and fit DMD
        dmd = DMD(x_copy, n_delays=n_delays, rank=rank, delay_interval=delay_interval, 
                 device=device, send_to_cpu=True)
        dmd.fit()
        
        # Ensure CUDA operations are synchronized if using GPU
        if device == 'cuda' and torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
            
        result = dmd.A_v.numpy()
        
        # Force cleanup
        del dmd
        del x_copy
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()
            
        return result
    except Exception as e:
        logger.error(f"Error fitting DMD: {str(e)}")
        # Try with CPU if GPU fails
        if device == 'cuda':
            logger.info("Retrying with CPU...")
            try:
                dmd = DMD(x, n_delays=n_delays, rank=rank, delay_interval=delay_interval, 
                         device='cpu', send_to_cpu=True)
                dmd.fit()
                return dmd.A_v.numpy()
            except Exception as e2:
                logger.error(f"CPU retry also failed: {str(e2)}")
        return None

def compute_similarity(i, j, dmds, valid_labels, comparison_dmd, dmd_cache_dir, force_recompute):
    """Compute similarity between two DMD matrices with caching"""
    logger = logging.getLogger()
    
    # Set diagonal to 1.0 (self-similarity)
    if i == j:
        return 1.0
    
    # Skip redundant calculations (matrix is symmetric)
    if j < i:
        return None
    
    # Check if we have a cached similarity result
    sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[i]}_{valid_labels[j]}.npy")
    
    if os.path.exists(sim_cache_file) and not force_recompute:
        try:
            sdmd = np.load(sim_cache_file)[0]
            logger.info(f"Loaded cached similarity between {valid_labels[i]} and {valid_labels[j]}: {sdmd:.4f}")
            return sdmd
        except Exception as e:
            logger.error(f"Error loading cached similarity: {str(e)}. Will recompute.")
    
    # Calculate DSA similarity
    logger.info(f"Calculating similarity between {valid_labels[i]} and {valid_labels[j]}")
    try:
        # Clear CUDA cache before computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Wait a bit to ensure resources are freed
        time.sleep(0.1)
            
        # Compute similarity
        sdmd = comparison_dmd.fit_score(dmds[i], dmds[j])
        
        # Ensure CUDA operations are synchronized
        if torch.cuda.is_available():
            torch.cuda.synchronize()      
        
        # Cache the similarity value
        try:
            np.save(sim_cache_file, np.array([sdmd]))
            logger.info(f"Cached similarity value")
        except Exception as e:
            logger.error(f"Error caching similarity value: {str(e)}")
        
        logger.info(f"Similarity between {valid_labels[i]} and {valid_labels[j]}: {sdmd:.4f}")
        return sdmd
        
    except Exception as e:
        logger.error(f"Error calculating similarity between {valid_labels[i]} and {valid_labels[j]}: {str(e)}")
        # Try with safer parameters if possible
        try:
            logger.info("Trying with sync mode...")
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(0.5)  # Wait longer
            
            # Compute with explicit synchronization
            sdmd = comparison_dmd.fit_score(dmds[i].copy(), dmds[j].copy())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            np.save(sim_cache_file, np.array([sdmd]))
            return sdmd
        except Exception as e2:
            logger.error(f"Second attempt also failed: {str(e2)}")
            return np.nan

def compute_batch_similarities(batch_pairs, dmds, valid_labels, dmd_cache_dir, force_recompute):
    """Compute similarities for a batch of pairs using CPU for stability"""
    # Create new SimilarityTransformDist instance for this process - use CPU for stability
    comparison_dmd = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
    
    results = []
    for i, j in batch_pairs:
        sim = compute_similarity(i, j, dmds, valid_labels, comparison_dmd, dmd_cache_dir, force_recompute)
        results.append((i, j, sim))
        # Force garbage collection after each computation
        gc.collect()
        
    # Force cleanup
    del comparison_dmd
    gc.collect()
        
    return results

def main():
    # Set environment variables for better CUDA error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors synchronous for better debugging
    
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
    parser.add_argument('--latent_size', type=int, default=None,
                        help='Only process this specific latent size')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation even if files exist')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force using CPU instead of GPU')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of latents to process in a batch')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for parallel processing')
    parser.add_argument('--sync_cuda', action='store_true',
                        help='Synchronize CUDA operations (slower but safer)')
    args = parser.parse_args()
    
    # Extract arguments
    N_DELAYS = args.n_delays
    RANK = args.rank
    DELAY_INTERVAL = args.delay_interval
    OUTPUT_PREFIX = args.output_prefix
    LATENTS_FILE = args.latents_file
    HASHES_FILE = args.hashes_file
    TARGET_LATENT_SIZE = args.latent_size
    FORCE_RECOMPUTE = args.force_recompute
    USE_CPU = args.use_cpu or not torch.cuda.is_available()
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SYNC_CUDA = args.sync_cuda
    
    # Check for CUDA availability if GPU usage is requested
    if not USE_CPU and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        USE_CPU = True
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_{timestamp}.log")
    
    logger.info(f"Starting DSA analysis with parameters:")
    logger.info(f"  N_DELAYS: {N_DELAYS}")
    logger.info(f"  RANK: {RANK}")
    logger.info(f"  DELAY_INTERVAL: {DELAY_INTERVAL}")
    logger.info(f"  LATENTS_FILE: {LATENTS_FILE}")
    logger.info(f"  HASHES_FILE: {HASHES_FILE}")
    logger.info(f"  Using GPU: {not USE_CPU}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Num Workers: {NUM_WORKERS}")
    logger.info(f"  Sync CUDA: {SYNC_CUDA}")
    if TARGET_LATENT_SIZE is not None:
        logger.info(f"  TARGET_LATENT_SIZE: {TARGET_LATENT_SIZE}")
    logger.info(f"  FORCE_RECOMPUTE: {FORCE_RECOMPUTE}")
    
    # Print GPU information if available
    if not USE_CPU:
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Define paths for intermediate files
    dmd_cache_dir = f"{OUTPUT_PREFIX}_dmd_cache"
    os.makedirs(dmd_cache_dir, exist_ok=True)
    
    # Check if final output files exist and we can skip computation
    final_files_exist = (
        os.path.exists(f'{OUTPUT_PREFIX}_sims_dmd.npy') and
        os.path.exists(f'{OUTPUT_PREFIX}_sims_model_type.npy') and
        os.path.exists(f'{OUTPUT_PREFIX}_sims_hash_value.npy') and
        os.path.exists(f'{OUTPUT_PREFIX}_labels.npy') and
        os.path.exists(f"{OUTPUT_PREFIX}_mds_embedding.csv") and
        os.path.exists(f"{OUTPUT_PREFIX}_metadata.csv")
    )
    
    if final_files_exist and not FORCE_RECOMPUTE:
        logger.info("All output files already exist. Loading them instead of recomputing.")
        # Load existing outputs
        sims_dmd = np.load(f'{OUTPUT_PREFIX}_sims_dmd.npy')
        sims_model_type = np.load(f'{OUTPUT_PREFIX}_sims_model_type.npy')
        sims_hash_value = np.load(f'{OUTPUT_PREFIX}_sims_hash_value.npy')
        valid_labels = np.load(f'{OUTPUT_PREFIX}_labels.npy').tolist()
        df = pd.read_csv(f"{OUTPUT_PREFIX}_mds_embedding.csv")
        metadata_df = pd.read_csv(f"{OUTPUT_PREFIX}_metadata.csv")
        
        logger.info(f"Loaded existing results with {len(valid_labels)} models.")
        logger.info("Analysis complete (loaded from existing files).")
        return
    
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
        
        # Skip if not matching the target latent size (if specified)
        if TARGET_LATENT_SIZE is not None and int(size) != TARGET_LATENT_SIZE:
            logger.info(f"Skipping {key} as it doesn't match target latent size {TARGET_LATENT_SIZE}")
            continue
        
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
            
            # Convert to float32 to save memory if needed
            if isinstance(latent, np.ndarray) and latent.dtype == np.float64:
                latent = latent.astype(np.float32)
                
            all_latents.append(latent)
            all_labels.append(label)
            all_model_types.append(model_type)
            all_hash_values.append(hash_val)
            
            logger.info(f"Added latent with label {label}, shape {latent.shape}")
    
    # Check if we have any latents to process
    if not all_latents:
        logger.warning("No latents to process. Check your latent size filter or input files.")
        return
    
    # Initialize lists to store DMD matrices
    dmds = []
    valid_indices = []  # To keep track of which latents were successfully processed
    valid_labels = []
    
    # Fit DMD to each set of latents - process sequentially for better stability
    logger.info(f"Fitting DMD for {len(all_latents)} latent arrays")
    
    # Process each latent one by one (sequential processing is more stable)
    for idx in tqdm(range(len(all_latents)), desc="Computing DMD matrices"):
        latents = all_latents[idx]
        label = all_labels[idx]
        dmd_cache_file = os.path.join(dmd_cache_dir, f"{label}_dmd.npy")
        
        # Check if DMD for this model is already cached
        if os.path.exists(dmd_cache_file) and not FORCE_RECOMPUTE:
            logger.info(f"Loading cached DMD for {label}")
            try:
                dmd_matrix = np.load(dmd_cache_file)
                dmds.append(dmd_matrix)
                valid_indices.append(idx)
                valid_labels.append(label)
                logger.info(f"Successfully loaded cached DMD for {label}")
                continue
            except Exception as e:
                logger.error(f"Error loading cached DMD for {label}: {str(e)}. Will recompute.")
        
        # If not cached or failed to load, compute DMD
        logger.info(f"Fitting DMD for model {idx+1}/{len(all_latents)}: {label}")
        
        # Clear GPU memory before computation
        if not USE_CPU and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Wait a bit to ensure resources are freed
        time.sleep(0.2)
        
        try:
            device = 'cpu' if USE_CPU else 'cuda'
            dmd_matrix = fit_dmd(
                latents, 
                n_delays=N_DELAYS, 
                rank=RANK, 
                delay_interval=DELAY_INTERVAL,
                device=device,
                sync=SYNC_CUDA
            )
            
            if dmd_matrix is not None:
                # Save DMD to cache as soon as it's computed
                try:
                    np.save(dmd_cache_file, dmd_matrix)
                    logger.info(f"Saved DMD cache for {label}")
                except Exception as e:
                    logger.error(f"Error saving DMD cache for {label}: {str(e)}")
                
                dmds.append(dmd_matrix)
                valid_indices.append(idx)
                valid_labels.append(label)
                logger.info(f"Successfully fit DMD for {label}")
            else:
                logger.warning(f"DMD fit returned None for {label}")
        
        except Exception as e:
            logger.error(f"Error fitting DMD for {label}: {str(e)}")
            # Try with CPU as fallback
            if not USE_CPU:
                logger.info("Retrying with CPU...")
                try:
                    dmd_matrix = fit_dmd(
                        latents, 
                        n_delays=N_DELAYS, 
                        rank=RANK, 
                        delay_interval=DELAY_INTERVAL,
                        device='cpu'
                    )
                    
                    if dmd_matrix is not None:
                        np.save(dmd_cache_file, dmd_matrix)
                        dmds.append(dmd_matrix)
                        valid_indices.append(idx)
                        valid_labels.append(label)
                        logger.info(f"Successfully fit DMD for {label} using CPU")
                except Exception as e2:
                    logger.error(f"CPU retry also failed: {str(e2)}")
        
        # Force garbage collection
        gc.collect()
        if not USE_CPU and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Filter labels to only include successfully processed ones
    valid_model_types = [all_model_types[i] for i in valid_indices]
    valid_hash_values = [all_hash_values[i] for i in valid_indices]
    
    # Save valid labels
    np.save(f'{OUTPUT_PREFIX}_labels.npy', np.array(valid_labels))
    
    # Calculate similarity matrix
    total_models = len(dmds)
    logger.info(f"Calculating similarity matrix for {total_models} models")
    
    # Define intermediate file paths for similarity matrices
    sim_dmd_path = f'{OUTPUT_PREFIX}_sims_dmd.npy'
    sim_model_type_path = f'{OUTPUT_PREFIX}_sims_model_type.npy'
    sim_hash_value_path = f'{OUTPUT_PREFIX}_sims_hash_value.npy'
    
    # Initialize similarity matrices
    sims_dmd = np.zeros((total_models, total_models))
    sims_model_type = np.zeros((total_models, total_models))
    sims_hash_value = np.zeros((total_models, total_models))
    
    # Fill in model type and hash value similarities (simple comparisons)
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
    
    # Save these matrices immediately
    np.save(sim_model_type_path, sims_model_type)
    np.save(sim_hash_value_path, sims_hash_value)
    
    # Generate all pairs to compute
    all_pairs = []
    for i in range(total_models):
        for j in range(i+1, total_models):  # Only upper triangle
            all_pairs.append((i, j))
    
    # Create dictionary to keep track of computed pairs
    completed_pairs_file = os.path.join(dmd_cache_dir, "completed_pairs.pkl")
    completed_pairs = set()
    
    if os.path.exists(completed_pairs_file) and not FORCE_RECOMPUTE:
        try:
            with open(completed_pairs_file, 'rb') as f:
                completed_pairs = pickle.load(f)
            logger.info(f"Loaded {len(completed_pairs)} completed pairs")
        except Exception as e:
            logger.error(f"Error loading completed pairs: {str(e)}")
    
    # Filter out already computed pairs
    remaining_pairs = [(i, j) for i, j in all_pairs if (i, j) not in completed_pairs]
    logger.info(f"Computing {len(remaining_pairs)} remaining similarity pairs")
    
    # Sequential computation is safer for GPU stability
    if not USE_CPU:
        # Use GPU but compute sequentially
        logger.info("Using GPU with sequential computation for better stability")
        device = 'cuda'
        comparison_dmd = SimilarityTransformDist(device=device, iters=2000, lr=1e-3)
        
        for i, j in tqdm(remaining_pairs, desc="Computing similarities"):
            try:
                # Clear GPU memory and synchronize
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.1)  # Brief pause to ensure resources are freed
                
                sdmd = compute_similarity(i, j, dmds, valid_labels, comparison_dmd, dmd_cache_dir, FORCE_RECOMPUTE)
                if sdmd is not None:
                    sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                    completed_pairs.add((i, j))
                    
                    # Save intermediate results periodically
                    if len(completed_pairs) % 5 == 0:
                        np.save(sim_dmd_path, sims_dmd)
                        with open(completed_pairs_file, 'wb') as f:
                            pickle.dump(completed_pairs, f)
                        logger.info(f"Saved intermediate results ({len(completed_pairs)}/{len(all_pairs)} pairs completed)")
            
            except Exception as e:
                logger.error(f"Error in GPU computation for pair ({i}, {j}): {str(e)}")
                logger.info("Switching to CPU for this computation")
                
                # Try with CPU as fallback
                try:
                    # Create a new CPU-based comparison object
                    cpu_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
                    sdmd = compute_similarity(i, j, dmds, valid_labels, cpu_comparison, dmd_cache_dir, FORCE_RECOMPUTE)
                    if sdmd is not None:
                        sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                        completed_pairs.add((i, j))
                    del cpu_comparison
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {str(e2)}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    elif NUM_WORKERS > 1 and len(remaining_pairs) > 1:
        # Use CPU with parallel processing
        logger.info(f"Using CPU parallel computation with {NUM_WORKERS} workers")
        
        # Split pairs into batches for workers
        batch_size = max(1, len(remaining_pairs) // (NUM_WORKERS * 2))
        batched_pairs = [remaining_pairs[i:i+batch_size] for i in range(0, len(remaining_pairs), batch_size)]
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Create partial function with fixed parameters
            compute_func = partial(
                compute_batch_similarities, 
                dmds=dmds, 
                valid_labels=valid_labels,
                dmd_cache_dir=dmd_cache_dir,
                force_recompute=FORCE_RECOMPUTE
            )
            
            # Submit all batches
            future_to_batch = {executor.submit(compute_func, batch): i 
                              for i, batch in enumerate(batched_pairs)}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                             total=len(future_to_batch),
                             desc="Processing similarity batches"):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()
                    
                    # Update similarity matrix with results
                    for i, j, sim in results:
                        if sim is not None:  # Skip redundant computations
                            sims_dmd[i, j] = sims_dmd[j, i] = sim
                            completed_pairs.add((i, j))
                    
                    # Save intermediate similarity matrix
                    np.save(sim_dmd_path, sims_dmd)
                    with open(completed_pairs_file, 'wb') as f:
                        pickle.dump(completed_pairs, f)
                    
                    logger.info(f"Completed batch {batch_idx}, saved intermediate results")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
    else:
        # Sequential CPU computation
        logger.info("Using sequential CPU computation")
        comparison_dmd = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
        
        for i, j in tqdm(remaining_pairs, desc="Computing similarities"):
            sdmd = compute_similarity(i, j, dmds, valid_labels, comparison_dmd, dmd_cache_dir, FORCE_RECOMPUTE)
            if sdmd is not None:
                sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                completed_pairs.add((i, j))
                
            # Save intermediate results periodically
            if len(completed_pairs) % 10 == 0:
                np.save(sim_dmd_path, sims_dmd)
                with open(completed_pairs_file, 'wb') as f:
                    pickle.dump(completed_pairs, f)
                logger.info(f"Saved intermediate results ({len(completed_pairs)}/{len(all_pairs)} pairs completed)")
    
    # Save final similarity matrices
    logger.info("Saving final outputs")
    np.save(sim_dmd_path, sims_dmd)
    
    # Create MDS embedding for visualization
    logger.info("Creating MDS embedding")
    try:
        # Create dissimilarity matrix (1 - similarity)
        # Handle any NaN values by replacing with mean
        dissim_matrix = np.where(np.isnan(sims_dmd), np.nanmean(sims_dmd), sims_dmd)
        # Convert to distances (1 - similarity)
        dissim_matrix = 1 - dissim_matrix
        
        # Perform MDS
        lowd_embedding = MDS(n_components=2, dissimilarity='precomputed', 
                            n_jobs=-1 if NUM_WORKERS > 1 else 1).fit_transform(dissim_matrix)
        
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