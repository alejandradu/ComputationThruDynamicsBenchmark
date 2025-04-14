from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.comparison import Comparison
import pandas as pd
import os
import re
import pickle

def extract_params_from_prefix(value):
    params = {}
    if "prefix=" in value:
        prefix_str = value.split("prefix=")[1]

        # Match all key=value pairs (keys can contain underscores, values can include scientific notation or strings)
        for match in re.finditer(r'([\w_]+)=([^_]+)', prefix_str):
            key = match.group(1)
            val = match.group(2)
            params[key] = val
    return params

def parse(DD_paths):
    """Parse hash_keys document for all DD models for a task and
    return a dataframe with the key and extracted results as columns, 
    and each model as a row
    
    Exports the dataframe to a csv file
    
    DD_paths: list of paths to DD models. Its subfolders should be hashed filenames
    """
    
    # create dataframe
    all_hashes = pd.DataFrame()
    
    for path in DD_paths:
        
        with os.scandir(path) as subdirs:
            
            # each of the e.g. ngru_200 ...
            for subdir in subdirs:
        
        hash_table = path + "/hash_keys.txt"
        
        current_key = None
        current_value_lines = []

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
    
                # Check for a new key line
                match = re.match(r'^([a-f0-9]{8}):\s*(.*)', line)
                if match:
                    # If there's an existing block, process it
                    if current_key is not None and current_value_lines:
                        full_value = ' '.join(current_value_lines)
                        parsed_data[current_key] = extract_params_from_prefix(full_value)
    
                    # Start a new block
                    current_key = match.group(1)
                    current_value_lines = [match.group(2)]
                else:
                    # Continuation of the previous key's value
                    current_value_lines.append(line)
    
            # Don't forget the last block
            if current_key is not None and current_value_lines:
                full_value = ' '.join(current_value_lines)
                parsed_data[current_key] = extract_params_from_prefix(full_value)

    return parsed_data



# from NODE

TT_NODE = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250407_PC_NODE_grid_final/max_epochs=1500_weight_decay=1.00E-08_learning_rate=1.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=128_delta_t=1.00E-02_alpha=5.00E-02_leak=True"
comparison_tag = "from_NODE"
run_name = "20250407_PC_NODE_grid_final"

tt_node = Analysis_TT(run_name=run_name, filepath= TT_NODE + "/") 

# create comparator and save TT model as reference
comparator = Comparison(comparison_tag=comparison_tag)
analysis = Analysis_TT(run_name=run_name, filepath= TT_NODE + "/") 
comparator.load_analysis(analysis, group = "NODE-PC-2", reference_analysis=True)

# group by model+linearity
# from noisy GRU

TT_nGRU = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250413_PClicks_NoisyGRU_final2/max_epochs=1500_weight_decay=1.00E-04_learning_rate=1.00E-04_noise=5.00E-04_seed=0_latent_size=128_delta_t=1.00E-02_latent_ic_var=5.00E-03_l2_wt=1.00E-05_noise_level=5.00E-03"


# from gNODE

TT_gNODE = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250412_PC_gNODE_sweep/max_epochs=1500_weight_decay=1.00E-06_learning_rate=2.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=64_delta_t=1.00E-02_alpha=1.00E-01_leak=True"
