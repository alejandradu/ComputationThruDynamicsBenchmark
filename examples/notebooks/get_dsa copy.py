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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
from functools import partial
from tqdm import tqdm
import time
import math

"""
Highly parallelized script to perform DSA computations on pre-processed latent data with
GPU acceleration, dynamic batch processing, and advanced memory management
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

def get_optimal_batch_size(total_memory, sample_size, safety_factor=0.7):
    """Calculate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 1
    
    # Get available GPU memory (in bytes)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    
    # Estimate memory needed per sample (multiply by sample size in bytes)
    # Add safety factor to avoid OOM errors
    return max(1, int((gpu_mem * safety_factor) / (sample_size * 4)))  # Assuming float32 (4 bytes)

def fit_dmd_batch(batch_data, batch_indices, n_delays, rank, delay_interval, device='cuda', 
                 streams=None, max_retries=2):
    """
    Fit DMD to a batch of input data in parallel using CUDA streams
    
    Parameters:
    -----------
    batch_data : list of ndarrays
        List of input data arrays with shape (B, T, N)
    batch_indices : list of int
        Indices of the batch items in the original dataset
    n_delays : int
        Number of delays for DMD
    rank : int
        Rank for DMD
    delay_interval : int
        Delay interval for DMD
    device : str
        Device to use for computation ('cuda' or 'cpu')
    streams : list of torch.cuda.Stream
        CUDA streams for parallel execution
    max_retries : int
        Maximum number of retries with CPU if GPU fails
        
    Returns:
    --------
    list of tuples (index, DMD matrix)
        DMD matrices for each input with their original indices
    """
    logger = logging.getLogger()
    results = []
    
    # If no streams provided and using CUDA, create streams
    if device == 'cuda' and streams is None and torch.cuda.is_available():
        streams = [torch.cuda.Stream() for _ in range(len(batch_data))]
    
    for i, (idx, x) in enumerate(zip(batch_indices, batch_data)):
        # Try with GPU first
        dmd_matrix = None
        retry_count = 0
        current_device = device
        
        while dmd_matrix is None and retry_count <= max_retries:
            try:
                # Ensure input has correct format (B, T, N)
                if len(x.shape) != 3:
                    if len(x.shape) == 2:
                        x = x.reshape(1, *x.shape)
                    else:
                        raise ValueError(f"Cannot reshape input with shape {x.shape} to (B, T, N) format")
                
                # Force data type to float32 to save memory
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
                
                # Make a copy to avoid modifying original data
                x_copy = x.copy()
                
                # Set up stream context if using CUDA
                if current_device == 'cuda' and torch.cuda.is_available() and streams:
                    stream_ctx = torch.cuda.stream(streams[i % len(streams)])
                else:
                    stream_ctx = torch.cuda.stream(torch.cuda.default_stream())
                
                # Fit DMD within stream context for parallelization
                with stream_ctx:
                    dmd = DMD(x_copy, n_delays=n_delays, rank=rank, delay_interval=delay_interval, 
                            device=current_device, send_to_cpu=True)
                    dmd.fit()
                    
                    # Ensure operation is complete if using GPU
                    if current_device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.current_stream().synchronize()
                    
                    dmd_matrix = dmd.A_v.numpy() if torch.is_tensor(dmd.A_v) else dmd.A_v
                    results.append((idx, dmd_matrix))
                    
                    # Clean up
                    del dmd
                    del x_copy
                    
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error fitting DMD (try {retry_count}/{max_retries+1}): {str(e)}")
                
                # Switch to CPU for retry
                if current_device == 'cuda':
                    current_device = 'cpu'
                    logger.info(f"Retrying with CPU for index {idx}")
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                else:
                    # If already using CPU and failed, try one more time with adjusted parameters
                    logger.info(f"Retrying CPU with adjusted parameters for index {idx}")
        
        if dmd_matrix is None:
            logger.error(f"Failed to compute DMD for index {idx} after {max_retries+1} attempts")
    
    # Ensure all streams are synchronized at the end
    if device == 'cuda' and torch.cuda.is_available():
        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize()
    
    return results

def compute_similarity_batch(batch_pairs, dmds, valid_labels, comparison_dmd, dmd_cache_dir, force_recompute, device='cpu'):
    """
    Compute similarities for a batch of pairs
    
    Parameters:
    -----------
    batch_pairs : list of tuples
        List of index pairs (i, j) to compute similarities for
    dmds : list
        List of DMD matrices
    valid_labels : list
        List of model labels
    comparison_dmd : SimilarityTransformDist
        Comparison object to compute similarities
    dmd_cache_dir : str
        Directory to cache computed similarities
    force_recompute : bool
        Whether to force recomputation even if cache exists
    device : str
        Device to use for computation ('cuda' or 'cpu')
        
    Returns:
    --------
    list of tuples
        List of (i, j, similarity) tuples
    """
    results = []
    
    for i, j in batch_pairs:
        # Skip self-comparisons (should be 1.0)
        if i == j:
            results.append((i, j, 1.0))
            continue
            
        # Skip redundant calculations (symmetric matrix)
        if j < i:
            continue
            
        # Check cache first
        sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[i]}_{valid_labels[j]}.npy")
        
        if os.path.exists(sim_cache_file) and not force_recompute:
            try:
                sdmd = np.load(sim_cache_file)[0]
                results.append((i, j, sdmd))
                continue
            except Exception:
                # Cache read failed, will recompute
                pass
        
        # Compute similarity
        try:
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
            
            sdmd = comparison_dmd.fit_score(dmds[i], dmds[j])
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
                
            # Cache result
            np.save(sim_cache_file, np.array([sdmd]))
            results.append((i, j, sdmd))
            
        except Exception as e:
            # If failed, try again with CPU
            if device == 'cuda':
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Create CPU-based comparison
                    cpu_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
                    sdmd = cpu_comparison.fit_score(dmds[i], dmds[j])
                    
                    # Cache result
                    np.save(sim_cache_file, np.array([sdmd]))
                    results.append((i, j, sdmd))
                    
                    # Clean up
                    del cpu_comparison
                    
                except Exception:
                    # Both attempts failed
                    results.append((i, j, np.nan))
            else:
                # Already using CPU and failed
                results.append((i, j, np.nan))
        
        # Force garbage collection after each computation
        gc.collect()
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def main():
    # Set environment variables for better CUDA error handling
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors synchronous for better debugging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate DSA scores with GPU parallelization')
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
    parser.add_argument('--dmd_batch_size', type=int, default=2,
                        help='Number of DMD matrices to compute in parallel')
    parser.add_argument('--sim_batch_size', type=int, default=16,
                        help='Number of similarities to compute in a batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                        help='Fraction of GPU memory to use (0.0-1.0)')
    parser.add_argument('--num_streams', type=int, default=4,
                        help='Number of CUDA streams for parallel GPU computation')
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
    DMD_BATCH_SIZE = args.dmd_batch_size
    SIM_BATCH_SIZE = args.sim_batch_size
    NUM_WORKERS = args.num_workers
    GPU_MEMORY_FRACTION = args.gpu_memory_fraction
    NUM_STREAMS = args.num_streams
    
    # Check for CUDA availability if GPU usage is requested
    if not USE_CPU and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        USE_CPU = True
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f"{OUTPUT_PREFIX}_{timestamp}.log")
    
    logger.info(f"Starting parallel DSA analysis with parameters:")
    logger.info(f"  N_DELAYS: {N_DELAYS}")
    logger.info(f"  RANK: {RANK}")
    logger.info(f"  DELAY_INTERVAL: {DELAY_INTERVAL}")
    logger.info(f"  LATENTS_FILE: {LATENTS_FILE}")
    logger.info(f"  HASHES_FILE: {HASHES_FILE}")
    logger.info(f"  Using GPU: {not USE_CPU}")
    logger.info(f"  DMD Batch Size: {DMD_BATCH_SIZE}")
    logger.info(f"  Similarity Batch Size: {SIM_BATCH_SIZE}")
    logger.info(f"  Num Workers: {NUM_WORKERS}")
    logger.info(f"  GPU Memory Fraction: {GPU_MEMORY_FRACTION}")
    logger.info(f"  Number of CUDA Streams: {NUM_STREAMS}")
    
    if TARGET_LATENT_SIZE is not None:
        logger.info(f"  TARGET_LATENT_SIZE: {TARGET_LATENT_SIZE}")
        
    logger.info(f"  FORCE_RECOMPUTE: {FORCE_RECOMPUTE}")
    
    # Print GPU information if available
    if not USE_CPU:
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"    CUDA Capability: {props.major}.{props.minor}")
            logger.info(f"    Multi-Processor Count: {props.multi_processor_count}")
    
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
            
            # Convert to float32 to save memory
            if isinstance(latent, np.ndarray) and latent.dtype == np.float64:
                latent = latent.astype(np.float32)
                
            all_latents.append(latent)
            all_labels.append(label)
            all_model_types.append(model_type)
            all_hash_values.append(hash_val)
            
            # Estimate memory size for batch optimization
            latent_shape = latent.shape
            latent_size_mb = np.prod(latent_shape) * 4 / (1024 * 1024)  # Size in MB (assuming float32)
            logger.info(f"Added latent with label {label}, shape {latent_shape}, estimated size: {latent_size_mb:.2f} MB")
    
    # Check if we have any latents to process
    if not all_latents:
        logger.warning("No latents to process. Check your latent size filter or input files.")
        return
    
    # Optimize batch size based on GPU memory if using GPU
    if not USE_CPU:
        # Get average latent size in bytes to adjust batch size
        avg_latent_size = sum(np.prod(x.shape) for x in all_latents) / len(all_latents) * 4  # float32 = 4 bytes
        
        # Calculate memory needed for DMD (estimate: ~10x size of input)
        dmd_memory_estimate = avg_latent_size * 10
        
        # Dynamic batch size based on available GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        safe_mem = gpu_mem * GPU_MEMORY_FRACTION  # Use only a portion of GPU memory
        
        # Adjust batch size (at least 1, but limited by memory)
        optimal_batch_size = max(1, int(safe_mem / dmd_memory_estimate))
        DMD_BATCH_SIZE = min(DMD_BATCH_SIZE, optimal_batch_size)
        logger.info(f"Adjusted DMD batch size to {DMD_BATCH_SIZE} based on GPU memory")
    
    # Initialize lists to store DMD matrices
    dmds = []
    valid_indices = []  # To keep track of which latents were successfully processed
    valid_labels = []
    
    # Create CUDA streams for parallel computation
    streams = None
    if not USE_CPU and torch.cuda.is_available():
        streams = [torch.cuda.Stream() for _ in range(min(NUM_STREAMS, DMD_BATCH_SIZE))]
        logger.info(f"Created {len(streams)} CUDA streams for parallel computation")
    
    # Check for cached DMD matrices and prepare computation batches
    computation_needed = []
    cached_dmds = {}
    
    logger.info("Checking for cached DMD matrices...")
    for idx, label in enumerate(all_labels):
        dmd_cache_file = os.path.join(dmd_cache_dir, f"{label}_dmd.npy")
        
        if os.path.exists(dmd_cache_file) and not FORCE_RECOMPUTE:
            try:
                dmd_matrix = np.load(dmd_cache_file)
                cached_dmds[idx] = dmd_matrix
                logger.info(f"Found cached DMD for {label}")
            except Exception as e:
                logger.error(f"Error loading cached DMD for {label}: {str(e)}. Will recompute.")
                computation_needed.append(idx)
        else:
            computation_needed.append(idx)
    
    logger.info(f"Found {len(cached_dmds)} cached DMDs, need to compute {len(computation_needed)}")
    
    # Process DMDs in batches for better memory management
    if computation_needed:
        device = 'cpu' if USE_CPU else 'cuda'
        
        # Process in batches
        for batch_start in range(0, len(computation_needed), DMD_BATCH_SIZE):
            batch_end = min(batch_start + DMD_BATCH_SIZE, len(computation_needed))
            batch_indices = computation_needed[batch_start:batch_end]
            batch_data = [all_latents[idx] for idx in batch_indices]
            batch_labels = [all_labels[idx] for idx in batch_indices]
            
            logger.info(f"Processing DMD batch {batch_start//DMD_BATCH_SIZE + 1}/{math.ceil(len(computation_needed)/DMD_BATCH_SIZE)}")
            logger.info(f"Batch indices: {batch_indices}")
            
            # Clear GPU memory before batch processing
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Process batch
            batch_results = fit_dmd_batch(
                batch_data, 
                batch_indices, 
                n_delays=N_DELAYS, 
                rank=RANK, 
                delay_interval=DELAY_INTERVAL,
                device=device,
                streams=streams
            )
            
            # Save results to cache and add to collection
            for idx, dmd_matrix in batch_results:
                if dmd_matrix is not None:
                    label = all_labels[idx]
                    dmd_cache_file = os.path.join(dmd_cache_dir, f"{label}_dmd.npy")
                    
                    # Save to cache
                    try:
                        np.save(dmd_cache_file, dmd_matrix)
                        logger.info(f"Saved DMD cache for {label}")
                    except Exception as e:
                        logger.error(f"Error saving DMD cache for {label}: {str(e)}")
                    
                    # Add to collection
                    cached_dmds[idx] = dmd_matrix
            
            # Force garbage collection after each batch
            gc.collect()
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Combine all successful DMD computations
    all_successful_indices = sorted(cached_dmds.keys())
    
    if not all_successful_indices:
        logger.error("No DMD matrices were successfully computed or loaded.")
        return
    
    # Filter to only include successful DMDs
    dmds = [cached_dmds[idx] for idx in all_successful_indices]
    valid_indices = all_successful_indices
    valid_labels = [all_labels[idx] for idx in valid_indices]
    valid_model_types = [all_model_types[idx] for idx in valid_indices]
    valid_hash_values = [all_hash_values[idx] for idx in valid_indices]
    
    # Save valid labels
    np.save(f'{OUTPUT_PREFIX}_labels.npy', np.array(valid_labels))
    logger.info(f"Saved {len(valid_labels)} valid labels")
    
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
            smtype = 1.0 if model_type_i == model_type_j else 0.0
            sims_model_type[i, j] = smtype
            
            # Calculate hash value similarity (1 if same, 0 if different)
            hash_i = valid_hash_values[i]
            hash_j = valid_hash_values[j]
            shash = 1.0 if hash_i == hash_j else 0.0
            sims_hash_value[i, j] = shash
            
            # Set diagonal to 1.0 (self-similarity)
            if i == j:
                sims_model_type[i, i] = 1.0
                sims_hash_value[i, i] = 1.0
                sims_dmd[i, i] = 1.0
    
    # Save these matrices immediately
    np.save(sim_model_type_path, sims_model_type)
    np.save(sim_hash_value_path, sims_hash_value)
    
    # Generate all pairs to compute - only upper triangle since matrix is symmetric
    all_pairs = []
    for i in range(total_models):
        for j in range(i+1, total_models):  # Only upper triangle
            all_pairs.append((i, j))
    
    # Load completed pairs cache if exists
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
    
    # Use cached results to fill in similarity matrix
    for i, j in completed_pairs:
        sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[i]}_{valid_labels[j]}.npy")
        if os.path.exists(sim_cache_file):
            try:
                sdmd = np.load(sim_cache_file)[0]
                sims_dmd[i, j] = sims_dmd[j, i] = sdmd
            except Exception as e:
                logger.error(f"Error loading cached similarity for {valid_labels[i]}-{valid_labels[j]}: {str(e)}")
                # Add back to remaining pairs
                remaining_pairs.append((i, j))
    
    # Choose computation strategy based on device and worker count
    device = 'cpu' if USE_CPU else 'cuda'
    
    if device == 'cuda' and NUM_WORKERS == 1:
        # Single GPU with stream parallelism
        logger.info("Using single GPU with CUDA streams for similarity computation")
        
        # Create streams for parallel computation
        sim_streams = [torch.cuda.Stream() for _ in range(min(NUM_STREAMS, SIM_BATCH_SIZE))]
        
        # Create similarity transform object
        comparison_dmd = SimilarityTransformDist(device=device, iters=2000, lr=1e-3)
        
        # Preload DMD matrices to GPU if possible
        try:
            if torch.cuda.is_available():
                # Calculate total memory needed for preloading
                total_elements = sum(np.prod(dmd.shape) for dmd in dmds)
                required_bytes = total_elements * 4  # float32 = 4 bytes
                
                # Check if we have enough GPU memory
                free_memory, total_memory = torch.cuda.mem_get_info()
                if required_bytes < free_memory * 0.9:  # Leave 10% margin
                    gpu_dmds = []
                    for dmd_matrix in dmds:
                        # Convert to PyTorch tensor and move to GPU
                        tensor = torch.tensor(dmd_matrix, device='cuda')
                        gpu_dmds.append(tensor)
                    logger.info(f"Successfully preloaded {len(dmds)} DMD matrices to GPU")
                else:
                    logger.warning(f"Not enough GPU memory for preloading. Need {required_bytes/1e9:.2f} GB, " 
                                  f"have {free_memory/1e9:.2f} GB free out of {total_memory/1e9:.2f} GB")
                    gpu_dmds = None
            else:
                gpu_dmds = None
        except Exception as e:
            logger.warning(f"Could not preload DMD matrices to GPU: {str(e)}")
            gpu_dmds = None
        
        # Process in batches
        for batch_start in range(0, len(remaining_pairs), SIM_BATCH_SIZE):
            batch_end = min(batch_start + SIM_BATCH_SIZE, len(remaining_pairs))
            batch_pairs = remaining_pairs[batch_start:batch_end]
            
            logger.info(f"Processing similarity batch {batch_start//SIM_BATCH_SIZE + 1}/{math.ceil(len(remaining_pairs)/SIM_BATCH_SIZE)}")
            
            # Clear GPU memory before batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Process batch with CUDA streams for parallelism
            batch_results = []
            for i, (idx1, idx2) in enumerate(batch_pairs):
                try:
                    # Use alternating streams for better resource utilization
                    stream = sim_streams[i % len(sim_streams)]
                    
                    # Create cache filename
                    sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[idx1]}_{valid_labels[idx2]}.npy")
                    
                    # Process within stream context
                    with torch.cuda.stream(stream):
                        # Use preloaded GPU tensors if available
                        if gpu_dmds is not None:
                            sdmd = comparison_dmd.fit_score(gpu_dmds[idx1], gpu_dmds[idx2])
                        else:
                            sdmd = comparison_dmd.fit_score(dmds[idx1], dmds[idx2])
                        
                        # Save result to cache
                        np.save(sim_cache_file, np.array([sdmd]))
                        batch_results.append((idx1, idx2, sdmd))
                        
                except Exception as e:
                    logger.error(f"Error in GPU computation for pair ({idx1}, {idx2}): {str(e)}")
                    logger.info("Falling back to CPU for this pair")
                    
                    try:
                        # Create CPU-based comparison as fallback
                        cpu_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
                        sdmd = cpu_comparison.fit_score(dmds[idx1], dmds[idx2])
                        
                        # Save to cache
                        np.save(sim_cache_file, np.array([sdmd]))
                        batch_results.append((idx1, idx2, sdmd))
                        
                        # Clean up
                        del cpu_comparison
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed for pair ({idx1}, {idx2}): {str(e2)}")
                        batch_results.append((idx1, idx2, np.nan))
            
            # Synchronize all streams
            for stream in sim_streams:
                stream.synchronize()
            torch.cuda.synchronize()
            
            # Update similarity matrix with batch results
            for idx1, idx2, sim in batch_results:
                if not np.isnan(sim):
                    sims_dmd[idx1, idx2] = sims_dmd[idx2, idx1] = sim
                    completed_pairs.add((idx1, idx2))
            
            # Save intermediate results
            np.save(sim_dmd_path, sims_dmd)
            with open(completed_pairs_file, 'wb') as f:
                pickle.dump(completed_pairs, f)
            
            logger.info(f"Completed batch, saved intermediate results ({len(completed_pairs)}/{len(all_pairs)} pairs)")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    elif device == 'cuda' and NUM_WORKERS > 1 and torch.cuda.device_count() > 1:
        # Multiple GPUs available - use distributed computation across GPUs
        logger.info(f"Using {min(NUM_WORKERS, torch.cuda.device_count())} GPUs for parallel computation")
        
        # Adjust workers to available GPUs
        effective_workers = min(NUM_WORKERS, torch.cuda.device_count())
        
        # Split pairs by GPU
        gpu_pair_batches = []
        chunk_size = math.ceil(len(remaining_pairs) / effective_workers)
        
        for i in range(0, len(remaining_pairs), chunk_size):
            gpu_pair_batches.append(remaining_pairs[i:i+chunk_size])
        
        # Process with ThreadPoolExecutor (better for GPU parallelism than ProcessPoolExecutor)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Function to process on specific GPU
            def process_gpu_batch(gpu_id, pairs_batch):
                # Set device for this thread
                torch.cuda.set_device(gpu_id)
                logger.info(f"Thread using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                
                # Create comparison object for this GPU
                gpu_comparison = SimilarityTransformDist(device=f'cuda:{gpu_id}', iters=2000, lr=1e-3)
                
                # Process pairs
                results = []
                for idx1, idx2 in tqdm(pairs_batch, desc=f"GPU {gpu_id} processing"):
                    try:
                        # Clear cache periodically
                        if len(results) % 5 == 0:
                            torch.cuda.empty_cache()
                            
                        # Cache filename
                        sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[idx1]}_{valid_labels[idx2]}.npy")
                        
                        # Compute similarity
                        sdmd = gpu_comparison.fit_score(dmds[idx1], dmds[idx2])
                        
                        # Save to cache
                        np.save(sim_cache_file, np.array([sdmd]))
                        results.append((idx1, idx2, sdmd))
                        
                    except Exception as e:
                        logger.error(f"Error on GPU {gpu_id} for pair ({idx1}, {idx2}): {str(e)}")
                        
                        # Try with CPU fallback
                        try:
                            cpu_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
                            sdmd = cpu_comparison.fit_score(dmds[idx1], dmds[idx2])
                            np.save(sim_cache_file, np.array([sdmd]))
                            results.append((idx1, idx2, sdmd))
                            del cpu_comparison
                        except Exception as e2:
                            logger.error(f"CPU fallback also failed: {str(e2)}")
                            results.append((idx1, idx2, np.nan))
                
                # Clean up
                del gpu_comparison
                torch.cuda.empty_cache()
                return results
            
            # Submit jobs to different GPUs
            future_to_gpu = {}
            for gpu_id, batch in enumerate(gpu_pair_batches):
                future = executor.submit(process_gpu_batch, gpu_id, batch)
                future_to_gpu[future] = gpu_id
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    batch_results = future.result()
                    
                    # Update similarity matrix
                    for idx1, idx2, sim in batch_results:
                        if not np.isnan(sim):
                            sims_dmd[idx1, idx2] = sims_dmd[idx2, idx1] = sim
                            completed_pairs.add((idx1, idx2))
                    
                    # Save intermediate results
                    np.save(sim_dmd_path, sims_dmd)
                    with open(completed_pairs_file, 'wb') as f:
                        pickle.dump(completed_pairs, f)
                        
                    logger.info(f"GPU {gpu_id} completed batch, saved intermediate results")
                    
                except Exception as e:
                    logger.error(f"Error processing batch on GPU {gpu_id}: {str(e)}")
    
    elif NUM_WORKERS > 1:
        # Multiple CPU workers - use process pool executor
        logger.info(f"Using CPU parallel processing with {NUM_WORKERS} workers")
        
        # Divide work into batches
        batches = []
        batch_size = max(1, len(remaining_pairs) // (NUM_WORKERS * 2))  # Create at least 2 batches per worker
        
        for i in range(0, len(remaining_pairs), batch_size):
            batches.append(remaining_pairs[i:i+batch_size])
        
        logger.info(f"Split work into {len(batches)} batches of approximately {batch_size} pairs each")
        
        # Create specialized function for process pool
        def process_cpu_batch(batch_id, pairs_batch):
            # Create new comparison object for this process
            process_comparison = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
            
            # Process pairs
            results = []
            for idx1, idx2 in pairs_batch:
                try:
                    # Create cache filename
                    sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[idx1]}_{valid_labels[idx2]}.npy")
                    
                    # Compute similarity
                    sdmd = process_comparison.fit_score(dmds[idx1], dmds[idx2])
                    
                    # Save to cache
                    np.save(sim_cache_file, np.array([sdmd]))
                    results.append((idx1, idx2, sdmd))
                    
                except Exception as e:
                    logger.error(f"Error in CPU process {batch_id} for pair ({idx1}, {idx2}): {str(e)}")
                    # Try again with adjusted parameters
                    try:
                        retry_comparison = SimilarityTransformDist(device='cpu', iters=1000, lr=5e-4)
                        sdmd = retry_comparison.fit_score(dmds[idx1], dmds[idx2])
                        np.save(sim_cache_file, np.array([sdmd]))
                        results.append((idx1, idx2, sdmd))
                        del retry_comparison
                    except Exception as e2:
                        logger.error(f"Retry also failed: {str(e2)}")
                        results.append((idx1, idx2, np.nan))
                
                # Periodic garbage collection
                if len(results) % 10 == 0:
                    gc.collect()
            
            # Clean up
            del process_comparison
            gc.collect()
            return results
        
        # Process batches with process pool
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_batch = {}
            for batch_id, batch in enumerate(batches):
                future = executor.submit(process_cpu_batch, batch_id, batch)
                future_to_batch[future] = batch_id
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                             total=len(future_to_batch),
                             desc="Processing similarity batches"):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Update similarity matrix
                    for idx1, idx2, sim in batch_results:
                        if not np.isnan(sim):
                            sims_dmd[idx1, idx2] = sims_dmd[idx2, idx1] = sim
                            completed_pairs.add((idx1, idx2))
                    
                    # Save intermediate results
                    np.save(sim_dmd_path, sims_dmd)
                    with open(completed_pairs_file, 'wb') as f:
                        pickle.dump(completed_pairs, f)
                    
                    logger.info(f"Completed batch {batch_id}, saved intermediate results ({len(completed_pairs)}/{len(all_pairs)} pairs)")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {str(e)}")
    
    else:
        # Single CPU - process sequentially
        logger.info("Using sequential CPU computation")
        comparison_dmd = SimilarityTransformDist(device='cpu', iters=2000, lr=1e-3)
        
        for i, j in tqdm(remaining_pairs, desc="Computing similarities"):
            sim_cache_file = os.path.join(dmd_cache_dir, f"sim_{valid_labels[i]}_{valid_labels[j]}.npy")
            
            try:
                sdmd = comparison_dmd.fit_score(dmds[i], dmds[j])
                np.save(sim_cache_file, np.array([sdmd]))
                sims_dmd[i, j] = sims_dmd[j, i] = sdmd
                completed_pairs.add((i, j))
            except Exception as e:
                logger.error(f"Error calculating similarity between {valid_labels[i]} and {valid_labels[j]}: {str(e)}")
                sims_dmd[i, j] = sims_dmd[j, i] = np.nan
            
            # Save intermediate results periodically
            if len(completed_pairs) % 10 == 0:
                np.save(sim_dmd_path, sims_dmd)
                with open(completed_pairs_file, 'wb') as f:
                    pickle.dump(completed_pairs, f)
                logger.info(f"Saved intermediate results ({len(completed_pairs)}/{len(all_pairs)} pairs completed)")
    
    # Save final similarity matrices
    logger.info("Saving final outputs")
    np.save(sim_dmd_path, sims_dmd)
    
    # Handle NaN values in the similarity matrix
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
        
    # Create MDS embedding for visualization
    logger.info("Creating MDS embedding")
    try:
        # Convert to distance matrix (1 - similarity)
        dissim_matrix = 1 - sims_dmd
        
        # Ensure all values are in valid range [0, 1]
        dissim_matrix = np.clip(dissim_matrix, 0, 1)
        
        # Perform MDS with parallel computation
        logger.info("Running MDS algorithm...")
        start_time = time.time()
        
        # Try non-metric MDS first (better for preserving local structure)
        try:
            lowd_embedding = MDS(
                n_components=2, 
                dissimilarity='precomputed',
                metric=False,  # Non-metric MDS
                n_jobs=NUM_WORKERS if NUM_WORKERS > 1 else None,
                n_init=4,  # Try multiple initializations
                max_iter=500,
                verbose=1
            ).fit_transform(dissim_matrix)
        except Exception as e:
            logger.warning(f"Non-metric MDS failed: {str(e)}. Falling back to metric MDS.")
            # Fallback to metric MDS
            lowd_embedding = MDS(
                n_components=2, 
                dissimilarity='precomputed',
                n_jobs=NUM_WORKERS if NUM_WORKERS > 1 else None,
                n_init=1
            ).fit_transform(dissim_matrix)
        
        elapsed = time.time() - start_time
        logger.info(f"MDS completed in {elapsed:.2f} seconds")
        
        # Create DataFrame for visualization
        df = pd.DataFrame()
        df['Model_Type'] = valid_model_types
        df['Model_Label'] = valid_labels
        df['Hash_Value'] = valid_hash_values
        df['DMD:0'] = lowd_embedding[:, 0]
        df['DMD:1'] = lowd_embedding[:, 1]
        
        # Extract model type and latent size as separate columns
        df['Architecture'] = [label.split('_')[0] for label in valid_labels]
        df['Latent_Size'] = [int(label.split('_')[1]) if label.split('_')[1].isdigit() 
                            else 0 for label in valid_labels]
        df['Is_True_Model'] = [hash_val == "true" for hash_val in valid_hash_values]
        
        # Save DataFrame
        df.to_csv(f"{OUTPUT_PREFIX}_mds_embedding.csv", index=False)
        logger.info(f"MDS embedding saved to {OUTPUT_PREFIX}_mds_embedding.csv")
        
        # Try to create a 3D embedding for additional visualization if requested
        try:
            logger.info("Creating 3D MDS embedding for advanced visualization")
            lowd_embedding_3d = MDS(
                n_components=3, 
                dissimilarity='precomputed',
                n_jobs=NUM_WORKERS if NUM_WORKERS > 1 else None,
                n_init=1
            ).fit_transform(dissim_matrix)
            
            # Add 3D coordinates to DataFrame
            df['DMD:0_3D'] = lowd_embedding_3d[:, 0]
            df['DMD:1_3D'] = lowd_embedding_3d[:, 1]
            df['DMD:2_3D'] = lowd_embedding_3d[:, 2]
            
            # Save extended DataFrame
            df.to_csv(f"{OUTPUT_PREFIX}_mds_embedding_3d.csv", index=False)
            logger.info(f"3D MDS embedding saved to {OUTPUT_PREFIX}_mds_embedding_3d.csv")
        except Exception as e:
            logger.warning(f"3D MDS embedding failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error creating MDS embedding: {str(e)}")
    
    # Create metadata DataFrame with information about each model
    metadata_df = pd.DataFrame({
        'model_label': valid_labels,
        'model_type': valid_model_types,
        'hash_value': valid_hash_values,
        'reduced_dimension': [label.split('_')[1] if len(label.split('_')) > 1 else "unknown" 
                             for label in valid_labels],
        'is_true_model': [hash_val == "true" for hash_val in valid_hash_values]
    })
    
    metadata_df.to_csv(f"{OUTPUT_PREFIX}_metadata.csv", index=False)
    logger.info(f"Model metadata saved to {OUTPUT_PREFIX}_metadata.csv")
    
    # Final summary and performance statistics
    logger.info(f"Analysis complete. Processed {total_models} models with {len(completed_pairs)} similarity computations.")
    logger.info(f"Results saved with prefix: {OUTPUT_PREFIX}")
    
    # Print success rate statistics
    similarity_entries = total_models * (total_models - 1) // 2
    completed_entries = len(completed_pairs)
    success_rate = completed_entries / similarity_entries * 100
    logger.info(f"Similarity computation success rate: {success_rate:.2f}% ({completed_entries}/{similarity_entries})")
    
    # Print timing information
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    if completed_entries > 0:
        time_per_similarity = total_time / completed_entries
        logger.info(f"Average time per similarity computation: {time_per_similarity:.4f} seconds")

if __name__ == "__main__":
    # Record start time for performance measurement
    start_time = time.time()
    main()