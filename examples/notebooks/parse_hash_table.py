import re
import os
import pandas as pd
from collections import defaultdict

def parse(DD_path):
    """Parse hash_keys document for all DD models for a task and
    return a dataframe with the key and extracted results as columns, 
    and each model as a row
    
    Exports the dataframe to a csv file
    
    DD_paths: path to DD models all from one TT sim data. Its subfolders
              have other subfolders where the hash_keys.txt files are
    """
    with os.scandir(DD_path) as subdirs:
        # Create a list to store all parameter dictionaries
        all_params = []
            
        # each of the e.g. ngru_200 ...
        for subdir in subdirs:
            if not subdir.is_dir():
                continue
            
            hash_table = os.path.join(subdir.path, "hash_keys.txt")

            if not os.path.exists(hash_table):
                print(f"Warning: hash_keys.txt not found in {subdir.name}")
                continue

            # Determine model type from subdirectory name
            model_type = "unknown"
            subdir_name_lower = subdir.name.lower()
            if "node" in subdir_name_lower:
                model_type = "node"
            elif "gnode" in subdir_name_lower:
                model_type = "gnode"
            elif "fr" in subdir_name_lower:
                model_type = "fr"
            elif "gru" in subdir_name_lower:
                model_type = "gru"

            current_key = None
            current_value_lines = []

            with open(hash_table, 'r') as file:
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
                            params = extract_params_from_prefix(full_value)
                            # Add the hashname as a parameter
                            params['hashname'] = current_key
                            # Add the model type as a parameter
                            params['model'] = model_type
                            all_params.append(params)

                        # Start a new block
                        current_key = match.group(1)
                        current_value_lines = [match.group(2)]
                    else:
                        # Continuation of the previous key's value
                        current_value_lines.append(line)

                # Don't forget the last block
                if current_key is not None and current_value_lines:
                    full_value = ' '.join(current_value_lines)
                    params = extract_params_from_prefix(full_value)
                    params['hashname'] = current_key
                    params['model'] = model_type
                    all_params.append(params)
    
    # Convert the list of dictionaries to a pandas DataFrame
    if all_params:
        # Find all unique keys across all dictionaries
        all_keys = set()
        for params in all_params:
            all_keys.update(params.keys())
        
        # Fill in missing keys with None
        for params in all_params:
            for key in all_keys:
                if key not in params:
                    params[key] = None
        
        # Create DataFrame with hashname as the first column
        df = pd.DataFrame(all_params)
        
        # Reorder columns to put hashname first, then model
        if 'hashname' in df.columns:
            cols = ['hashname', 'model'] + [col for col in df.columns if col not in ['hashname', 'model']]
            df = df[cols]
            
        return df
    else:
        # Return empty DataFrame with hashname and model columns
        return pd.DataFrame(columns=['hashname', 'model'])
    
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