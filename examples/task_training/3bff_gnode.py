import os

import ray

LOCAL_MODE = False  # Set to True to run locally (for debugging or RandomTarget)
if LOCAL_MODE:
    ray.init(local_mode=True, num_gpus=0)  # Ensure no GPUs are requested
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
from pathlib import Path

import dotenv
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from ctd.task_modeling.task_training import train
from utils import generate_paths, trial_function

dotenv.load_dotenv(override=True)


# ---------------Options---------------
OVERWRITE = True  # Set to True to overwrite existing run

RUN_DESC = "3BFF_gNODE"
TASK = "NBFF"  # Task to train on (see configs/task_env for options)
MODEL = "gNODE"  # Model to train (see configs/model for options)

# ----------------- Parameter Selection -----------------------------------
CPU_PER_SAMPLE = 1
GPU_PER_SAMPLE = 0.25
TOTAL_SAMPLES = 16

SEARCH_SPACE = {
    "trainer.max_epochs": 1000,
    # 'datamodule_train.batch_size': tune.choice([1000]),
    "task_wrapper.weight_decay": 1e-10,
    "task_wrapper.learning_rate": tune.choice([1e-2, 1e-3]),
    "params.seed": 0,
    "env_params.noise": tune.choice([0.0, 0.1]),  # sets both env_sim and env_task
    "model.gating_num_layers": tune.choice([2,3]),
    "model.gating_hidden_size": tune.choice([32,64])
}

# careful not to put too many params or the filename is too long

# node for 3bff already ahs parameters shown on notebook, already in config files

# ------------------Data Management --------------------------------
combo_dict = generate_paths(RUN_DESC, TASK, MODEL)
path_dict = combo_dict["path_dict"]
RUN_TAG = combo_dict["RUN_TAG"]
RUN_DIR = combo_dict["RUN_DIR"]
config_dict = combo_dict["config_dict"]


# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: str,
    config_dict: dict,
):

    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True)
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    tune.run(
        tune.with_parameters(
            train,
            run_tag=run_tag_in,
            path_dict=path_dict,
            config_dict=config_dict,
        ),
        metric="loss",
        mode="min",
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=CPU_PER_SAMPLE, gpu=GPU_PER_SAMPLE),
        num_samples=TOTAL_SAMPLES,
        storage_path=str(RUN_DIR),
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
        path_dict=path_dict,
        config_dict=config_dict,
    )
