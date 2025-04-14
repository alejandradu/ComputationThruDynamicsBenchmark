from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.comparison import Comparison
import os
import pandas as pd
import pickle
import argparse
import time
import gc
import logging
from datetime import datetime

# Configure logging
def setup_logging(log_file=None):
    """Set up logging to both console and file if specified"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # If no log file specified, create one with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"sequential_processing_{timestamp}.log"
    
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
    parser = argparse.ArgumentParser(description='Process directories sequentially and save metrics to CSV')
    parser.add_argument('--model_type', type=str, default='NODE', choices=['NODE', 'nGRU', 'gNODE'], 
                        help='Model type to process (NODE, nGRU, or gNODE)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index for processing (to resume from a previous run)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='How often to save progress to CSV (in number of directories)')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info(f"Starting sequential processing for model type: {args.model_type}")
    logger.info(f"Starting from index: {args.start_idx}")
    
    # Initialize paths and settings based on model type
    if args.model_type == 'NODE':
        TT_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250407_PC_NODE_grid_final/max_epochs=1500_weight_decay=1.00E-08_learning_rate=1.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=128_delta_t=1.00E-02_alpha=5.00E-02_leak=True"
        DD_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_NODE"
        comparison_tag = "from_NODE"
        run_name = "20250407_PC_NODE_grid_final"
        hash_table_file = "hash_table_from_node.csv"
    elif args.model_type == 'nGRU':
        TT_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250413_PClicks_NoisyGRU_final2/max_epochs=1500_weight_decay=1.00E-04_learning_rate=1.00E-04_noise=5.00E-04_seed=0_latent_size=128_delta_t=1.00E-02_latent_ic_var=5.00E-03_l2_wt=1.00E-05_noise_level=5.00E-03"
        DD_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_GRU" 
        comparison_tag = "from_nGRU"
        run_name = "20250413_PClicks_NoisyGRU_final2"
        hash_table_file = "hash_table_from_noisy_gru.csv"
    elif args.model_type == 'gNODE':
        TT_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250412_PC_gNODE_sweep/max_epochs=1500_weight_decay=1.00E-06_learning_rate=2.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=64_delta_t=1.00E-02_alpha=1.00E-01_leak=True"
        DD_PATH = "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_gNODE"
        comparison_tag = "from_gNODE"
        run_name = "20250412_PC_gNODE_sweep"
        hash_table_file = "hash_table_from_gnode.csv"
    
    start_time = time.time()
    
    # Load hash table
    logger.info(f"Loading hash table from {hash_table_file}")
    df = pd.read_csv(hash_table_file)
    
    # Create reference comparator with just the TT model
    logger.info("Creating reference comparator")
    comparator = Comparison(comparison_tag=comparison_tag)
    analysis_tt = Analysis_TT(run_name=run_name, filepath=TT_PATH + "/")
    comparator.load_analysis(analysis_tt, group=comparison_tag, reference_analysis=True)
    
    # Define metrics to compute
    metric_list = ["state_r2", "rate_r2", 'co-bps', 'cycle_con']
    
    # Add metric columns to dataframe if they don't exist
    for metric in metric_list:
        if metric not in df.columns:
            df[metric] = None
    
    # Create output dataframe for tracking processed directories
    output_df_file = f"processed_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Collect all tune directories
    tune_dirs = []
    for run_dir in os.scandir(DD_PATH):
        if not run_dir.is_dir():
            continue
            
        for tune_dir in os.scandir(os.path.join(DD_PATH, run_dir)):
            if not tune_dir.is_dir():
                continue
                
            tune_dirs.append((os.path.join(DD_PATH, run_dir.name, tune_dir.name), tune_dir.name))
    
    logger.info(f"Found {len(tune_dirs)} directories to process")
    
    # Process each directory sequentially
    success_count = 0
    failure_count = 0
    
    for i, (tune_dir_path, tune_dir_name) in enumerate(tune_dirs[args.start_idx:], start=args.start_idx):
        logger.info(f"Processing directory {i+1}/{len(tune_dirs)}: {tune_dir_name}")
        
        try:
            # Create analysis object
            analysisDD = Analysis_DD.create(
                run_name=tune_dir_name,
                filepath=tune_dir_path + "/",
                model_type="SAE"
            )
            
            # Get hashname from directory name (assuming format like DT_[hashname])
            hashname = tune_dir_name[3:] if tune_dir_name.startswith("DT_") else tune_dir_name
            
            # Find row in dataframe
            row_idx = df[df['hashname'] == hashname].index
            
            if len(row_idx) == 0:
                logger.warning(f"No matching row found for hashname: {hashname}")
                failure_count += 1
                continue
            
            # Add to comparator temporarily
            comparator.load_analysis(analysisDD, group=hashname)
            
            # Compute metrics
            metrics = comparator.compute_metrics(metric_list=metric_list, cycle_con_var=0.01)
            
            # Update dataframe with metrics
            for metric in metric_list:
                if hashname in metrics and metric in metrics[hashname]:
                    df.loc[row_idx, metric] = metrics[hashname][metric]
            
            # Remove analysis from comparator to free memory
            del comparator.analyses[-1]
            del comparator.groups[-1]
            comparator.num_analyses -= 1
            
            success_count += 1
            logger.info(f"Successfully processed {tune_dir_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {tune_dir_name}: {str(e)}")
            failure_count += 1
        
        # Save progress at checkpoint intervals
        if (i + 1) % args.checkpoint_interval == 0 or i + 1 == len(tune_dirs):
            logger.info(f"Saving progress to {hash_table_file}")
            df.to_csv(hash_table_file, index=False)
            logger.info(f"Progress: {i+1}/{len(tune_dirs)} directories processed ({success_count} success, {failure_count} failure)")
        
        # Force garbage collection
        gc.collect()
    
    # Save final results
    logger.info(f"Processing completed: {success_count} successes, {failure_count} failures")
    logger.info(f"Saving final results to {hash_table_file}")
    df.to_csv(hash_table_file, index=False)
    
    # Also save a backup with timestamp
    backup_file = f"{hash_table_file.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False)
    logger.info(f"Saved backup to {backup_file}")
    
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()