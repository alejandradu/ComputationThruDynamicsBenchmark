# Import necessary libraries
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.comparison import Comparison
import dotenv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import csv
from ctd.task_modeling.task_env.task_env import PClicks
from sklearn.decomposition import PCA
from DSA import DSA
from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist

# Set up task environment
pc1e2 = PClicks(
    n_timesteps=2000, 
    noise=0.00, 
    rateL=39,               # Hz
    delta_t=1e-2,           # seconds
    allRates=True, 
    fixation_period=1.5,    # seconds
    response_period=0.8,    # seconds 
    delay_min=0.5,          # seconds
    delay_max=1.3,          # seconds
) 
task_dataset_dict, _ = pc1e2.generate_dataset(2000)  # Generate dataset with same size as wrapper saved input

# Function to flatten latent states for DSA/DMD
def flatten_x(x):
    # Ensure proper shape for DSA/DMD
    if len(x.shape) == 3:  # [B, T, D]
        return x
    elif len(x.shape) == 4:  # [Conditions, Trials, Time, Dimension]
        # Reshape to [B, T, D] where B = Conditions*Trials
        return x.reshape(-1, x.shape[2], x.shape[3])
    else:
        raise ValueError(f"Unexpected shape: {x.shape}")

def fit_dmd(x,n_delays,rank,delay_interval):
    x = flatten_x(x)
    #notice how we initialize the dmd separately here, rather than the DSA object itself
    dmd = DMD(x,n_delays=n_delays,rank=rank,delay_interval=delay_interval,device='cuda', send_to_cpu=True)
    dmd.fit()
    return dmd.A_v.numpy()

# Get ground truth latents
ground_truth_latents = "/scratch/gpfs/ad2002/content/trained_models/task-trained/20250412_PC_gNODE_sweep/max_epochs=1500_weight_decay=1.00E-06_learning_rate=2.00E-03_seed=0_noise=1.70E-04_latent_size=2_layer_hidden_size=64_delta_t=1.00E-02_alpha=1.00E-01_leak=True"
ana = Analysis_TT(run_name="gnode_pc", filepath=ground_truth_latents + "/")
latents_truth = ana.get_latents().detach().numpy()

# Apply PCA to ground truth latents
n_components = 2
B, T, N = latents_truth.shape
latents_flat = latents_truth.reshape(-1, N).squeeze()
pca_model_truth = PCA(n_components=n_components)
pca_model_truth.fit(latents_flat)
latents_truth_pca = pca_model_truth.transform(latents_flat)
latents_truth_pca = latents_truth_pca.reshape(B, T, n_components)
latents_truth_pca = fit_dmd(latents_truth_pca,30,75,1)

# Directory for data-driven models
dd_dir = "/scratch/gpfs/ad2002/content/trained_models/task-trained/tt_PClicks/from_gNODE/"

# List to store all processed latents (excluding ground truth)
D = []  # This will hold all models except the ground truth
hashnames = []  # This will hold the corresponding hashnames


# Prepare CSV file for storing results
csv_file = "/home/ad2002/ComputationThruDynamicsBenchmark/examples/notebooks/hash_tables_final/final_hash_table_from_gnode.csv"
# Check if CSV exists, if not create it with headers
if not os.path.exists(csv_file):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Create directory if it doesn't exist
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["hashname", "similarity_to_ground_truth"])

# Function to update CSV with new similarity score
def update_csv_row(hashname, similarity):
    # Read existing CSV
    rows = []
    hashname_exists = False
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header
            for row in reader:
                if row[0] == hashname:  # If hashname matches
                    rows.append([hashname, similarity])
                    hashname_exists = True
                else:
                    rows.append(row)
    
    # If hashname doesn't exist, add new row
    if not hashname_exists:
        rows.append([hashname, similarity])
    
    # Write back to CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["hashname", "similarity_to_ground_truth"])  # Write header
        writer.writerows(rows)

# Function to get latent size from model
def get_latent_size(model):
    # Try to get latent size directly
    try:
        return model.latent_size
    except AttributeError:
        pass
    
    # Try to get from ic_linear attribute (output size)
    try:
        return model.ic_linear.out_features
    except AttributeError:
        pass
    
    # If all fails, raise error
    raise AttributeError("Could not determine latent size")

# Find model.pkl at any nesting level
for root, dirs, files in os.walk(dd_dir):
    if "model.pkl" in files:
        # Get the immediate parent directory path
        parent_dir_path = root
        
        # Extract the directory name (not the full path)
        parent_dir_name = os.path.basename(parent_dir_path)
        
        # Extract hashname from the parent directory name
        # Looking for directories like "DT_xxxxxxxx" where "xxxxxxxx" is the hashname
        if parent_dir_name.startswith("DT_") and len(parent_dir_name) > 3:
            hashname = parent_dir_name[3:]
        else:
            # If the directory doesn't match expected pattern, try looking at grandparent
            grandparent_dir_path = os.path.dirname(parent_dir_path)
            grandparent_dir_name = os.path.basename(grandparent_dir_path)
            
            if grandparent_dir_name.startswith("DT_") and len(grandparent_dir_name) > 3:
                hashname = grandparent_dir_name[3:]
            else:
                # If no hash pattern found, use the parent directory name as is
                hashname = parent_dir_name
        
        print(f"Processing model in {root} with hashname {hashname}")
        
        try:
            # Load the model and datamodule
            model_path = os.path.join(root, "model.pkl")
            datamodule_path = os.path.join(root, "datamodule.pkl")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(datamodule_path, "rb") as f:
                datamodule = pickle.load(f)
            
            # Get latent size using our helper function
            try:
                latent_size = get_latent_size(model)
                print(f"Found latent size: {latent_size}")
            except AttributeError:
                print(f"Could not determine latent size, skipping {root}")
                continue
            
            # Prepare initial conditions
            ic_point = np.zeros((2000, latent_size))
            inputs_latents = task_dataset_dict["inputs"]
            input_latents = torch.tensor(inputs_latents, dtype=torch.float32)
            
            if isinstance(ic_point, np.ndarray):
                ic_point = torch.tensor(ic_point, dtype=torch.float32)
                
            # Add batch dimension if not present
            if len(ic_point.shape) == 1:
                ic_point = ic_point.unsqueeze(0)  # [1, latent_size]
                
            dyn_model = model.decoder.cell
            
            # Generate trajectory
            try:
                with torch.no_grad():
                    hidden = ic_point
                    states = []
                    
                    # Step through input sequence
                    for input_step in input_latents.transpose(1, 0):
                        hidden = dyn_model(input_step, hidden)
                        states.append(hidden.clone())
                    
                    # Stack states into trajectory
                    latents = torch.stack(states, dim=1)  # [B, T, latent_size]
            except Exception as e:
                print(f"Error generating trajectory: {e}, skipping {root}")
                continue
            
            # Skip if latent dimension is too small
            if latents.shape[-1] < n_components:
                print(f"Latent dimension too small in {root}, skipping")
                continue
            
            # Apply PCA to current model's latents
            B, T, N = latents.shape
            latents_flat = latents.reshape(-1, N).squeeze()
            pca_model = PCA(n_components=n_components)
            pca_model.fit(latents_flat)
            latents_pca = pca_model.transform(latents_flat)
            latents_pca = latents_pca.reshape(B, T, n_components)
            
            dmd_here = fit_dmd(latents_pca,30,75,1)
            
            # Add to collection
            D.append(dmd_here)
            hashnames.append(hashname)
            print(f"Successfully processed model with latent size {latent_size}")
            
        except Exception as e:
            print(f"Error processing {root}: {e}")
            continue

# Create a SimilarityTransformDist object for comparison
comparison = SimilarityTransformDist(device='cuda', iters=2000, lr=1e-3)

# Initialize array to store similarity scores
similarities = np.zeros(len(D))

# Compare ground truth with each model in D and update CSV
for i, model_latents in enumerate(D):
    try:
        # Compute similarity score
        similarity = comparison.fit_score(latents_truth_pca, model_latents)
        similarities[i] = similarity
        
        # Update CSV
        update_csv_row(hashnames[i], similarity)
        
        print(f"Model {hashnames[i]}: Similarity with ground truth = {similarity}")
    except Exception as e:
        print(f"Error computing similarity for model {hashnames[i]}: {e}")
        similarities[i] = float('nan')

# Save all similarities as a pickle file
print("Saving similarities")
similarity_dict = {hashnames[i]: similarities[i] for i in range(len(hashnames))}
with open("similarities_one_to_many.pkl", "wb") as f:
    pickle.dump({
        "similarities": similarities,
        "hashnames": hashnames,
        "similarity_dict": similarity_dict
    }, f)

print("Done!")