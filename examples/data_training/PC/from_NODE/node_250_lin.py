# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import dotenv
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from ctd.data_modeling.train_PTL import train_PTL

dotenv.load_dotenv(override=True)
HOME_DIR = Path(os.environ.get("HOME_DIR"))

log = logging.getLogger(__name__)
# ---------------Options---------------
LOCAL_MODE = False
OVERWRITE = True
WANDB_LOGGING = False  # If users have a WandB account

RUN_DESC = "lin_250:50_1"  # Description of the run
MODEL_CLASS = "SAE"  # "LFADS" or "SAE" MAYBE ALSO HAS LDS
MODEL = "NODE"  # see /ctd/data_modeling/configs/models/{MODEL_CLASS}/ for options
DATA = "PClicks"  # "NBFF", "RandomTarget" or "MultiTask
INFER_INPUTS = False  # Whether external inputs are inferred or supplied

# default datasets
if DATA == "NBFF":
    prefix = "tt_3bff"   
elif DATA == "MultiTask":
    prefix = "tt_MultiTask"
elif DATA == "RandomTarget":
    prefix = "tt_RandomTarget"
elif DATA == "PClicks":
    prefix = "tt_PClicks"
    
## CHANGE ME
NUM_SAMPLES = 1
CPU_PER_SAMPLE = 1       # this is usually just 1 
GPU_PER_SAMPLE = 0.2     # this def varies (0.125 - 0.5)

# -------------------------------------
# Hyperparameter sweeping:
# Default parameters chosen to replicate Fig. 5
# -------------------------------------

# HYDRA WILL SCREAM IF ANY OF THE PARAMETERS HAVE '=' INSIDE A STRING so you need the index. 
# 1. run this in shell to get the file index for the desired run names
# 2. the name has to be under datasets / dt /
# 3. write down the maping name -> file index

# import os
# def get_file_index(directory, filenames):
#     ind = []
#     files = os.listdir(directory)
#     for index, filename in enumerate(files):
#         for f in filenames:
#             if f == filename:
#                 ind.append(index)
#     return ind 

# rateL=39
# [13, 52, 66]
# ["max_epochs=1000_weight_decay=1.00E-09_learning_rate=1.00E-03_seed=0_noise=0_rateL=39_latent_size=2_layer_hidden_size=128_latent_l2_wt=1.00E-08", 
#  "max_epochs=1000_weight_decay=1.00E-09_learning_rate=1.00E-03_seed=0_noise=0_rateL=39_latent_size=3_layer_hidden_size=128_latent_l2_wt=1.00E-08", 
#  "max_epochs=1000_weight_decay=1.00E-09_learning_rate=1.00E-03_seed=0_noise=0_rateL=39_latent_size=5_layer_hidden_size=128_latent_l2_wt=1.00E-08"]),

# rateL=26
# [40, 50]
# ["max_epochs=1000_weight_decay=1.00E-09_learning_rate=1.00E-03_seed=0_noise=0_rateL=26_latent_size=2_layer_hidden_size=128_latent_l2_wt=1.00E-08",
# "max_epochs=1000_weight_decay=1.00E-09_learning_rate=1.00E-03_seed=0_noise=0_rateL=26_latent_size=3_layer_hidden_size=128_latent_l2_wt=1.00E-08"]
    

SEARCH_SPACE = {
    "datamodule.prefix": "20250407_PC_NODE_grid_final", # prefix for TT sweep
    "model.latent_size": tune.grid_search([2,3,5,10,32,64]), 
    "trainer.max_epochs": tune.grid_search([500, 501]), 
    "params.seed": 0,
    "model.lr": 2e-3,   # prob too high
    "model.weight_decay": 0,
    "model.encoder_size": 128,
    "datamodule.file_index": 21,         # CHANGE ME
    "model.vf_hidden_size": 128, 
    "model.vf_num_layers": 3,
    "model.output_nonlinearity": None,   # else is tanh
    "model.heldin_size": 250,
    "model.heldout_size": 300,   # SUM held_in + held_out
    "model.alpha": 0.05, # 0.05,
    
    # COPY THE TARGET DATA AND FILL IN IN ORDER: 
    # heldin_280_heldout_20_fr_scaling_1.0_rect_func_softplus_seed_0.h5
    
    "datamodule.neuron_dict.n_heldin": 250,
    "datamodule.neuron_dict.n_heldout": 50,
    "datamodule.embed_dict.fr_scaling": 1.0,
    "datamodule.embed_dict.rect_func": "softplus",
    "datamodule.seed": 0,
    "datamodule.noise_dict.obs_noise": "poisson",
    "datamodule.noise_dict.dispersion": 1.0
    # "model.alpha": 0.05,   # delta_t(model)/tau
    # "model.leak": True,
    # "model.encoder_size": 128,  # GRU makes the embeddings
}

# -----------------Default Parameter Sets -----------------------------------
cpath = "../data_modeling/configs"

model_path = Path(
    (
        f"{cpath}/models/{MODEL_CLASS}/{DATA}/{DATA}_{MODEL}"
        f"{'_infer' if INFER_INPUTS else ''}.yaml"
    )
)

datamodule_path = Path(
    (
        f"{cpath}/datamodules/{MODEL_CLASS}/data_{DATA}"
        f"{'_infer' if INFER_INPUTS else ''}.yaml"
    )
)

callbacks_path = Path(f"{cpath}/callbacks/{MODEL_CLASS}/default_{DATA}.yaml")
loggers_path = Path(f"{cpath}/loggers/{MODEL_CLASS}/default.yaml")
trainer_path = Path(f"{cpath}/trainers/{MODEL_CLASS}/trainer_{DATA}.yaml")

if not WANDB_LOGGING:
    loggers_path = Path(f"{cpath}/loggers/{MODEL_CLASS}/default_no_wandb.yaml")
    callbacks_path = Path(f"{cpath}/callbacks/{MODEL_CLASS}/default_no_wandb.yaml")

if MODEL_CLASS not in ["LDS"]:
    config_dict = dict(
        model=model_path,
        datamodule=datamodule_path,
        callbacks=callbacks_path,
        loggers=loggers_path,
        trainer=trainer_path,
    )
    train = train_PTL
else:
    config_dict = dict(
        model=model_path,
        datamodule=datamodule_path,
        trainer=trainer_path,
    )
    # train = train_JAX

# ------------------Data Management Variables --------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")
RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
RUNS_HOME = Path(HOME_DIR)
RUN_DIR = HOME_DIR / "content" / "runs" / "data-trained" / RUN_TAG
path_dict = dict(
    dd_datasets=HOME_DIR / "content" / "datasets" / "dd",   # will this sweep over all dt?
    trained_models=HOME_DIR / "content" / "trained_models" / "task-trained" / prefix,
)


def trial_function(trial):
    return trial.experiment_tag


# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: dict,
    config_dict: dict,
):
    if LOCAL_MODE:
        ray.init(local_mode=True)
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True)
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    run_dir = str(RUN_DIR)
    tune.run(
        tune.with_parameters(
            train, run_tag=run_tag_in, config_dict=config_dict, path_dict=path_dict
        ),
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=CPU_PER_SAMPLE, gpu=GPU_PER_SAMPLE),
        num_samples=NUM_SAMPLES,
        storage_path=run_dir,
        search_alg=BasicVariantGenerator(),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["loss", "training_iteration"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=trial_function,
    )


if __name__ == "__main__":
    main(
        run_tag_in=RUN_TAG,
        config_dict=config_dict,
        path_dict=path_dict,
    )
