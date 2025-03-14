import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
import random

from ctd.task_modeling.datamodule.samplers import GroupedSampler, RandomSampler
from ctd.task_modeling.task_env.loss_func import MultiTaskLoss, SimpleMultiTaskLoss
from ctd.task_modeling.task_env.task_env import DecoupledEnvironment


class MultiTaskWrapper(DecoupledEnvironment):
    """
    An environment for the MultiTask dataset (Driscol et al. 2021).
    15 tasks are available: Select which ones to include in the
    task_env config file.

    """

    def __init__(
        self,
        task_list: list,
        bin_size: int,
        n_timesteps: int,
        num_targets: int,
        noise: float,
        dynamic_noise: float,
        grouped_sampler: bool = False,
        dataset_name="MultiTask",
        loss_func="MultiTaskLoss",
        *args,
        **kwargs,
    ):
        """
        Args:
            task_list: List of task names to include in the dataset
            bin_size: Bin size of the dataset (default = 20 ms)
            n_timesteps: Number of timesteps in the dataset (Default = 640 maximum)
            num_targets: Number of targets in each task (Default = 32)
            noise: Noise level of the dataset (Default = 0.31)
            grouped_sampler: Whether to use a grouped sampler or a random sampler
            - Grouped sampler: Samples tasks in groups by minibatch
                May be necessary for shared motifs?
            - Random sampler: Samples tasks randomly by minibatch
            dataset_name: Name of the dataset
            - Default = "MultiTask"
            
            NOTE: would be good to carefully check for magic numbers
            
        """
        self.n_timesteps = n_timesteps
        self.dataset_name = dataset_name
        self.bin_size = bin_size

        # Iterate through the task list and create a list of MultiTask objects
        self.task_list = [
            MultiTask(
                task, bin_size=self.bin_size, num_targets=num_targets, noise=noise
            )
            for task in task_list
        ]

        # Labels for the inputs and outputs
        self.input_labels = [
            "Fixation",
            "StimMod1Cos",
            "StimMod1Sin",
            "StimMod2Cos",
            "StimMod2Sin",
            "DelayPro",
            "DelayAnti",
            "MemoryPro",
            "MemoryAnti",
            "ReactPro",
            "ReactAnti",
            "IntMod1",
            "IntMod2",
            "ContextIntMod1",
            "ContextIntMod2",
            "ContextIntMultimodal",
            "ReactMatch2Sample",
            "ReactNonMatch2Sample",
            "ReactCatPro",
            "ReactCatAnti",
            "DM1",
            "DM2",
            "CDM1",
            "CDM2",
        ]
        self.output_labels = [
            "Fixation",
            "ResponseCos",
            "ResponseSin",
        ]
        
        # avoid magic numbers
        self.INPUT_SIZE = len(self.input_labels)
        self.OUTPUT_SIZE = len(self.output_labels)
        
        # Create a string of the task list for the dataset name
        self.task_list_str = task_list

        # Create the action, observation, and goal spaces
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(self.INPUT_SIZE,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )

        latent_l2_wt = kwargs.get("latent_l2_wt", 1.0)
        self.noise = noise
        self.dynamic_noise = dynamic_noise
        self.coupled_env = False
        self.extra = "phase_dict"
        self.latent_l2_wt = latent_l2_wt
        if grouped_sampler:
            self.sampler = GroupedSampler
        else:
            self.sampler = RandomSampler
        
        if loss_func=="MultiTaskLoss":
            self.loss_func = MultiTaskLoss(lat_loss_weight=self.latent_l2_wt)
        elif loss_func=="SimpleMultiTaskLoss":
            self.loss_func = SimpleMultiTaskLoss(lat_loss_weight=self.latent_l2_wt)
        else:
            raise ValueError("Invalid loss function: try MultiTaskLoss or SimpleMultiTaskLoss")
            

    # NOTE: check if the seeding is correct
    def set_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """
        No step is implemented for the MultiTask dataset

        TODO: REVISE
        """
        pass

    def reset(self):
        """
        No reset is implemented for the MuliTask dataset

        TODO: REVISE
        """
        pass

    def generate_dataset(self, n_samples):
        """
        Generates a dataset for the MultiTask dataset

        TODO: REVISE

        Args:
            n_samples (int) : Number of samples to generate (from each task)

        Returns:
            dataset_dict (dict) :
            extra_dict (dict) :
        """
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples * len(self.task_list), 3))
        outputs_ds = []
        inputs_ds = []
        true_inputs_ds = []
        phase_list = []
        task_names = []
        conds_ds = []
        extra_ds = []

        # Iterate through the task list and generate trials
        for task_num, task in enumerate(self.task_list):
            inputs_task = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))
            true_inputs_task = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))
            outputs_task = np.zeros(shape=(n_samples, n_timesteps, self.OUTPUT_SIZE))
            extra_task = np.zeros(shape=(n_samples, 2))
            for i in range(n_samples):
                (
                    inputs_noise,
                    outputs,
                    phase_dict,
                    task_name,
                    inputs,
                ) = task.generate_trial()
                trial_len = inputs_noise.shape[0]
                outputs_task[i, :trial_len, :] = outputs
                inputs_task[i, :trial_len, :] = inputs_noise
                true_inputs_task[i, :trial_len, :] = inputs
                phase_list.append(phase_dict)
                task_names.append(task_name)
                extra_task[i, :] = phase_dict["response"]
            conds_ds.append(task_num * np.ones(shape=(n_samples, 1)))
            inputs_ds.append(inputs_task)
            outputs_ds.append(outputs_task)
            true_inputs_ds.append(true_inputs_task)
            extra_ds.append(extra_task)

        inputs_ds = np.concatenate(inputs_ds, axis=0)
        true_inputs_ds = np.concatenate(true_inputs_ds, axis=0)
        outputs_ds = np.concatenate(outputs_ds, axis=0)
        conds_ds = np.concatenate(conds_ds, axis=0)
        extra_ds = np.concatenate(extra_ds, axis=0)

        # Return dictionary of necessary variables for task-trained models
        dataset_dict = {
            # ----------Mandatory------------------
            "inputs": inputs_ds,
            "targets": outputs_ds,
            "ics": ics_ds,
            "true_inputs": true_inputs_ds,
            "conds": conds_ds,
            # Extra is anything that is needed for the training
            # that isn't an input, target, or ic
            #   E.g. in Multi-task, it is the start and end index of
            #   the response phase (for loss weighting)
            "inputs_to_env": np.zeros((n_samples * len(self.task_list), 0)),
            "extra": extra_ds,
        }
        extra_dict = {
            "phase_dict": phase_list,
            "task_names": task_names,
        }
        return dataset_dict, extra_dict

    def plot_tasks(self):
        for task in self.task_list:
            task.plot_trial()


class MultiTask:
    def __init__(self, task_name: str, bin_size: int, num_targets: int, noise: float):
        """
        Args:

            task_name: Name of the task to generate
            bin_size: Bin size of the dataset (default = 20 ms)
            num_targets: Number of targets in each task (Default = 32)
            noise: Noise level of the dataset (Default = 0.31)
        """
        self.task_name = task_name
        self.noise = noise
        self.bin_size = bin_size
        self.num_targets = num_targets

        # Check if dataset name is in the list of available datasets
        if self.task_name not in [
            "DelayPro",
            "DelayAnti",
            "MemoryPro",
            "MemoryAnti",
            "ReactPro",
            "ReactAnti",
            "IntMod1",
            "IntMod2",
            "ContextIntMod1",
            "ContextIntMod2",
            "ContextIntMultimodal",
            "Match2Sample",
            "NonMatch2Sample",
            "MatchCatPro",
            "MatchCatAnti",
            "DM1",
            "DM2",
            "CDM1",
            "CDM2",
        ]:
            raise ValueError("Dataset name not in available datasets.")
        self.input_labels = [
            "Fixation",
            "StimMod1Cos",
            "StimMod1Sin",
            "StimMod2Cos",
            "StimMod2Sin",
            "DelayPro",
            "DelayAnti",
            "MemoryPro",
            "MemoryAnti",
            "ReactPro",
            "ReactAnti",
            "IntMod1",
            "IntMod2",
            "ContextIntMod1",
            "ContextIntMod2",
            "ContextIntMultimodal",
            "ReactMatch2Sample",
            "ReactNonMatch2Sample",
            "ReactCatPro",
            "ReactCatAnti",
            "DM1",
            "DM2",
            "CDM1",
            "CDM2",
        ]

        self.output_labels = [
            "Fixation",
            "ResponseCos",
            "ResponseSin",
        ]
        # changing if clause to avoid mismatch
        if "DM" in self.task_name:
            self.task_type = "Simple"
        elif "Delay" in self.task_name:
            self.task_type = "Delay"
        elif "Memory" in self.task_name:
            self.task_type = "Memory"
        elif "React" in self.task_name:
            self.task_type = "React"
        elif "Int" in self.task_name:
            self.task_type = "Decision"
        elif "Match" in self.task_name:
            self.task_type = "Match"

    def generate_trial(self):
        """
        Generates a trial for the MultiTask dataset
        Implements trial logic for each task type
        """

        # Generate the trial parameters and phase lengths
        bs = self.bin_size
        if self.task_type == "Delay":
            context_len = np.random.randint(300 / bs, 700 / bs)
            stim1_len = np.random.randint(200 / bs, 1500 / bs)
            mem1_len = 0
            stim2_len = 0
            mem2_len = 0
            response_len = np.random.randint(300 / bs, 700 / bs)

        elif self.task_type == "Memory":
            context_len = np.random.randint(300 / bs, 700 / bs)
            stim1_len = np.random.randint(200 / bs, 1600 / bs)
            mem1_len = np.random.randint(200 / bs, 1600 / bs)
            stim2_len = 0
            mem2_len = 0
            response_len = np.random.randint(300 / bs, 700 / bs)

        elif self.task_type == "React":
            context_len = np.random.randint(500 / bs, 2500 / bs)
            stim1_len = 0
            mem1_len = 0
            stim2_len = 0
            mem2_len = 0
            response_len = np.random.randint(300 / bs, 1700 / bs)
            
        elif self.task_type == "Simple":
            context_len = int(350 / bs)
            stim1_len = int(800 / bs)
            mem1_len = int(100 / bs)
            stim2_len = 0
            mem2_len = 0
            response_len = int(20 / bs)

        elif self.task_type == "Decision":
            context_len = np.random.randint(200 / bs, 600 / bs)
            stim1_len = np.random.randint(200 / bs, 1600 / bs)
            mem1_len = np.random.randint(200 / bs, 1600 / bs)
            stim2_len = np.random.randint(200 / bs, 1600 / bs)
            mem2_len = np.random.randint(100 / bs, 300 / bs)
            response_len = np.random.randint(300 / bs, 700 / bs)

        elif self.task_type == "Match":
            context_len = np.random.randint(200 / bs, 600 / bs)
            stim1_len = np.random.randint(200 / bs, 1600 / bs)
            mem1_len = np.random.randint(200 / bs, 1600 / bs)
            stim2_len = 0
            mem2_len = 0
            response_len = np.random.randint(300 / bs, 700 / bs)

        # Set the indices for each phase
        stim1_ind = context_len
        mem1_ind = context_len + stim1_len
        stim2_ind = context_len + stim1_len + mem1_len
        mem2_ind = context_len + stim1_len + mem1_len + stim2_len
        response_ind = context_len + stim1_len + mem1_len + stim2_len + mem2_len
        total_len = (
            context_len + stim1_len + mem1_len + stim2_len + mem2_len + response_len
        )

        # Initialize the inputs and outputs and set the target angles
        inputs = np.zeros((total_len, len(self.input_labels)))
        outputs = np.zeros((total_len, len(self.output_labels)))
        targ_ang_list = np.linspace(-np.pi, np.pi, self.num_targets, endpoint=False)

        match self.task_type:
            case "Delay":
                # Fixation
                inputs[:response_ind, 0] = 1

                # Target
                targ_num_1 = np.random.randint(0, self.num_targets)
                targ_ang_1 = targ_ang_list[targ_num_1]

                inputs[stim1_ind:, 1] = np.cos(targ_ang_1)
                inputs[stim1_ind:, 2] = np.sin(targ_ang_1)
                if "Pro" in self.task_name:
                    inputs[:, 5] = 1
                    pro_anti = 1
                else:
                    inputs[:, 6] = 1
                    pro_anti = -1

                # Build the output
                outputs[:response_ind, 0] = 1
                outputs[response_ind:total_len, 1] = pro_anti * np.cos(targ_ang_1)
                outputs[response_ind:total_len, 2] = pro_anti * np.sin(targ_ang_1)
                phase_dict = {
                    "context": [0, stim1_ind],
                    "stim1": [stim1_ind, response_ind],
                    "response": [response_ind, total_len],
                }

            case "Memory":
                # Fixation
                inputs[:response_ind, 0] = 1

                # Target
                targ_num_1 = np.random.randint(0, self.num_targets)
                targ_ang_1 = targ_ang_list[targ_num_1]

                inputs[stim1_ind:mem1_ind, 1] = np.cos(targ_ang_1)
                inputs[stim1_ind:mem1_ind, 2] = np.sin(targ_ang_1)
                if "Pro" in self.task_name:
                    inputs[:, 7] = 1
                    pro_anti = 1
                else:
                    inputs[:, 8] = 1
                    pro_anti = -1

                # Build the output
                outputs[:response_ind, 0] = 1
                outputs[response_ind:total_len, 1] = pro_anti * np.cos(targ_ang_1)
                outputs[response_ind:total_len, 2] = pro_anti * np.sin(targ_ang_1)
                phase_dict = {
                    "context": [0, stim1_ind],
                    "stim1": [stim1_ind, mem1_ind],
                    "mem1": [mem1_ind, response_ind],
                    "response": [response_ind, total_len],
                }
            case "React":
                # Fixation
                inputs[:, 0] = 1

                # Target
                targ_num_1 = np.random.randint(0, self.num_targets)
                targ_ang_1 = targ_ang_list[targ_num_1]

                inputs[response_ind:, 1] = np.cos(targ_ang_1)
                inputs[response_ind:, 2] = np.sin(targ_ang_1)
                if "Pro" in self.task_name:
                    inputs[:, 9] = 1
                    pro_anti = 1
                else:
                    inputs[:, 10] = 1
                    pro_anti = -1

                # Build the output
                outputs[:response_ind, 0] = 1
                outputs[response_ind:total_len, 1] = pro_anti * np.cos(targ_ang_1)
                outputs[response_ind:total_len, 2] = pro_anti * np.sin(targ_ang_1)
                phase_dict = {
                    "context": [0, response_ind],
                    "response": [response_ind, total_len],
                }

            case "Decision":
                # Fixation
                inputs[:response_ind, 0] = 1

                # Target
                stim_meanA = np.random.uniform(0.8, 1.2)
                stim_meanB = np.random.uniform(0.8, 1.2)

                stim_coh_range = np.array([0.08, 0.16, 0.32])
                stim_coh_signA = np.random.choice([-1, 1])
                stim_coh_signB = np.random.choice([-1, 1])

                stim_cohA = np.random.choice(stim_coh_range)
                stim_cohB = np.random.choice(stim_coh_range)

                targ_num_1 = np.random.randint(0, self.num_targets)
                targ_ang_1 = targ_ang_list[targ_num_1]

                angle_diff = np.deg2rad(np.random.uniform(90, 270))

                targ_ang_2 = np.mod(targ_ang_1 + angle_diff, 2 * np.pi)

                targ_mag_1 = stim_meanA + stim_cohA * stim_coh_signA
                targ_mag_2 = stim_meanA - stim_cohA * stim_coh_signA

                targ_mag_1B = stim_meanB + stim_cohB * stim_coh_signB
                targ_mag_2B = stim_meanB - stim_cohB * stim_coh_signB

                outputs[:response_ind, 0] = 1
                if "Context" in self.task_name:
                    inputs[stim1_ind:mem1_ind, 1] = targ_mag_1 * np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = targ_mag_1 * np.sin(targ_ang_1)
                    inputs[stim2_ind:mem2_ind, 1] = targ_mag_2 * np.cos(targ_ang_2)
                    inputs[stim2_ind:mem2_ind, 2] = targ_mag_2 * np.sin(targ_ang_2)

                    inputs[stim1_ind:mem1_ind, 3] = targ_mag_1B * np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 4] = targ_mag_1B * np.sin(targ_ang_1)
                    inputs[stim2_ind:mem2_ind, 3] = targ_mag_2B * np.cos(targ_ang_2)
                    inputs[stim2_ind:mem2_ind, 4] = targ_mag_2B * np.sin(targ_ang_2)

                    if "1" in self.task_name:
                        inputs[:, 13] = 1
                        largest_mag = np.argmax([targ_mag_1, targ_mag_2])
                        ang_array = [targ_ang_1, targ_ang_2]
                        max_ang = ang_array[largest_mag]
                    elif "2" in self.task_name:
                        inputs[:, 14] = 1
                        largest_mag = np.argmax([targ_mag_1B, targ_mag_2B])
                        ang_array = [targ_ang_1, targ_ang_2]
                        max_ang = ang_array[largest_mag]
                    else:
                        inputs[:, 15] = 1
                        largest_mag = np.argmax(
                            [targ_mag_1 + targ_mag_1B, targ_mag_2 + targ_mag_2B]
                        )
                        ang_array = [targ_ang_1, targ_ang_2]
                        max_ang = ang_array[largest_mag]

                    outputs[response_ind:total_len, 1] = np.cos(max_ang)
                    outputs[response_ind:total_len, 2] = np.sin(max_ang)

                elif "1" in self.task_name:
                    inputs[stim1_ind:mem1_ind, 1] = targ_mag_1 * np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = targ_mag_1 * np.sin(targ_ang_1)
                    inputs[stim2_ind:mem2_ind, 1] = targ_mag_2 * np.cos(targ_ang_2)
                    inputs[stim2_ind:mem2_ind, 2] = targ_mag_2 * np.sin(targ_ang_2)
                    inputs[:, 11] = 1
                    if targ_mag_1 > targ_mag_2:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_1)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_1)
                    else:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)
                elif "2" in self.task_name:
                    inputs[stim1_ind:mem1_ind, 3] = targ_mag_1 * np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 4] = targ_mag_1 * np.sin(targ_ang_1)
                    inputs[stim2_ind:mem2_ind, 3] = targ_mag_2 * np.cos(targ_ang_2)
                    inputs[stim2_ind:mem2_ind, 4] = targ_mag_2 * np.sin(targ_ang_2)
                    inputs[:, 12] = 1
                    if targ_mag_1 > targ_mag_2:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_1)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_1)
                    else:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)
                phase_dict = {
                    "context": [0, stim1_ind],
                    "stim1": [stim1_ind, mem1_ind],
                    "mem1": [mem1_ind, stim2_ind],
                    "stim2": [stim2_ind, mem2_ind],
                    "mem2": [mem2_ind, response_ind],
                    "response": [response_ind, total_len],
                }
                
            case "Simple":
                # Fixation
                inputs[:response_ind, 0] = 1
                
                # noise added to inputs after
                coherences = [-4, -2, -1, 1, 2, 4]
                # fixation output
                outputs[:response_ind, 0] = 1
                
                mod1 = coherences[random.randint(0, len(coherences)-1)]
                mod2 = coherences[random.randint(0, len(coherences)-1)]
                
                # leaving inputs and outputs corresponding to second coordinate = 0
                if "C" in self.task_name:
                    # input coherences for one stimulus period, mod1 (say color)
                    inputs[stim1_ind:mem1_ind, 1] = mod1
                    # inputs[stim1_ind:mem1_ind, 2] = targ_mag_1 * np.sin(targ_ang_1)
                    
                    # input coherences for one stimulus period, mod2 (say motion)
                    inputs[stim1_ind:mem1_ind, 3] = mod2
                    # inputs[stim1_ind:mem1_ind, 4] = targ_mag_1B * np.sin(targ_ang_1)

                    if "1" in self.task_name:
                        inputs[:, 22] = 1
                        target = np.sign(mod1)
                    else:
                        inputs[:, 23] = 1
                        target = np.sign(mod2)

                    outputs[response_ind:total_len, 1] = target

                elif "1" in self.task_name:
                    inputs[:, 20] = 1
                    inputs[stim1_ind:mem1_ind, 1] = mod1
                    outputs[response_ind:total_len, 1] = np.sign(mod1)
                    
                elif "2" in self.task_name:
                    inputs[:, 21] = 1
                    inputs[stim1_ind:mem1_ind, 1] = mod2
                    outputs[response_ind:total_len, 1] = np.sign(mod2)
                
                phase_dict = {
                    "context": [0, stim1_ind],
                    "stim1": [stim1_ind, mem1_ind],
                    "mem1": [mem1_ind, response_ind], 
                    "response": [response_ind, total_len],
                }
                
                
                
            case "Match":
                # Fixation
                inputs[:, 0] = 1
                if self.task_name == "Match2Sample":
                    inputs[:, 16] = 1
                    matched = np.random.choice([True, False])
                    # Target
                    if matched:
                        targ_num_1 = np.random.randint(0, self.num_targets)
                        targ_ang_1 = targ_ang_list[targ_num_1]

                        targ_num_2 = targ_num_1
                        targ_ang_2 = targ_ang_1
                    else:
                        targ_num_1 = np.random.randint(0, self.num_targets)
                        targ_ang_1 = targ_ang_list[targ_num_1]

                        angle_diff = np.deg2rad(np.random.randint(10, 350))

                        targ_ang_2 = np.mod(
                            targ_ang_1 + angle_diff,
                            2 * np.pi,
                        )

                    inputs[stim1_ind:mem1_ind, 1] = np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = np.sin(targ_ang_1)

                    inputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                    inputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                    outputs[:response_ind, 0] = 1
                    if matched:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                elif self.task_name == "NonMatch2Sample":
                    inputs[:, 17] = 1
                    isOpposite = np.random.choice([True, False])
                    if isOpposite:
                        targ_num_1 = np.random.randint(0, self.num_targets)
                        targ_ang_1 = targ_ang_list[targ_num_1]

                        targ_ang_2 = np.mod(targ_ang_1 + np.pi, 2 * np.pi)
                    else:
                        targ_num_1 = np.random.randint(0, self.num_targets)
                        targ_ang_1 = targ_ang_list[targ_num_1]

                        targ_ang_2 = np.mod(
                            targ_ang_1 + np.pi + np.deg2rad(np.random.randint(10, 350)),
                            2 * np.pi,
                        )

                    inputs[stim1_ind:mem1_ind, 1] = np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = np.sin(targ_ang_1)

                    inputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                    inputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                    if isOpposite:
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                elif self.task_name == "MatchCatPro":
                    inputs[:, 18] = 1
                    targ_num_1 = np.random.randint(0, self.num_targets)
                    targ_ang_1 = targ_ang_list[targ_num_1]
                    targ_num_2 = np.random.randint(0, self.num_targets)
                    targ_ang_2 = targ_ang_list[targ_num_2]

                    inputs[stim1_ind:mem1_ind, 1] = np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = np.sin(targ_ang_1)

                    inputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                    inputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                    if (targ_ang_1 < 0 and targ_ang_2 < 0) or (
                        targ_ang_1 >= 0 and targ_ang_2 >= 0
                    ):
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                elif self.task_name == "MatchCatAnti":
                    inputs[:, 19] = 1
                    targ_num_1 = np.random.randint(0, self.num_targets)
                    targ_ang_1 = targ_ang_list[targ_num_1]
                    targ_num_2 = np.random.randint(0, self.num_targets)
                    targ_ang_2 = targ_ang_list[targ_num_2]

                    inputs[stim1_ind:mem1_ind, 1] = np.cos(targ_ang_1)
                    inputs[stim1_ind:mem1_ind, 2] = np.sin(targ_ang_1)

                    inputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                    inputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                    if (targ_ang_1 < 0 and targ_ang_2 >= 0) or (
                        targ_ang_1 >= 0 and targ_ang_2 < 0
                    ):
                        outputs[response_ind:total_len, 1] = np.cos(targ_ang_2)
                        outputs[response_ind:total_len, 2] = np.sin(targ_ang_2)

                phase_dict = {
                    "context": [0, stim1_ind],
                    "stim1": [stim1_ind, mem1_ind],
                    "mem1": [mem1_ind, response_ind],
                    "response": [response_ind, total_len],
                }
        inputs_noise = inputs + self.noise * np.random.randn(*inputs.shape)
        return inputs_noise, outputs, phase_dict, self.task_name, inputs

    def plot_trial(self):
        inputs, outputs, phase_dict, task_name, true_inputs = self.generate_trial()
        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(7, 1, 1)      
        for phase in phase_dict:
            ax1.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax1.axvline(phase_dict[phase][1], color="k", linestyle="--")
            ax1.text(
                phase_dict[phase][0]
                + (phase_dict[phase][1] - phase_dict[phase][0]) / 2,
                0.5,
                phase,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="top",
            )
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ax_small_limits = [-1.5, 1.5]
        if self.task_type == "Simple":
            ax_limits = [-5, 5]
        else:
            ax_limits = ax_small_limits

        ax1 = fig.add_subplot(7, 1, 2)
        ax1.plot(inputs[:, 0])
        ax1.set_ylabel("Fixation")
        for phase in phase_dict:
            ax1.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax1.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax1.set_xticks([])
        ax1.set_ylim(ax_small_limits)

        ax1 = fig.add_subplot(7, 1, 3)
        ax1.plot(inputs[:, 1])
        ax1.plot(inputs[:, 2])
        ax1.set_ylabel("StimMod1")
        for phase in phase_dict:
            ax1.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax1.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax1.set_xticks([])
        ax1.set_ylim(ax_limits)

        ax1 = fig.add_subplot(7, 1, 4)
        ax1.plot(inputs[:, 3])
        ax1.plot(inputs[:, 4])
        ax1.set_ylabel("StimMod2")
        for phase in phase_dict:
            ax1.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax1.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax1.set_xticks([])
        ax1.set_ylim(ax_limits)

        ax1 = fig.add_subplot(7, 1, 5)
        for i in range(15):
            ax1.plot(inputs[:, 5 + i])
        ax1.set_ylabel("Task")
        for phase in phase_dict:
            ax1.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax1.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax1.set_xticks([])
        ax1.set_ylim(ax_limits)

        ax2 = fig.add_subplot(7, 1, 6)
        ax2.plot(outputs[:, 0])
        ax2.set_ylabel("Fixation")
        for phase in phase_dict:
            ax2.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax2.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax2.set_xticks([])
        ax2.set_ylim(ax_small_limits)

        ax2 = fig.add_subplot(7, 1, 7)
        ax2.plot(outputs[:, 1])
        ax2.plot(outputs[:, 2])
        ax2.set_ylabel("Response")
        for phase in phase_dict:
            ax2.axvline(phase_dict[phase][0], color="k", linestyle="--")
            ax2.axvline(phase_dict[phase][1], color="k", linestyle="--")
        ax2.set_ylim(ax_small_limits)

        plt.suptitle(f"Trial: {self.task_name}")
        # plt.savefig(f"{self.task_name}_state_diag.png")

        fig1 = plt.figure()
        # Plot stim1, stim2 and response as a scatter plot
        ax_input1 = fig1.add_subplot(1, 4, 1)
        ax_input2 = fig1.add_subplot(1, 4, 2)
        ax_output = fig1.add_subplot(1, 4, 3)
        ax_labels = fig1.add_subplot(1, 4, 4)

        color_dict = {
            "context": "k",
            "stim1": "r",
            "mem1": "orange",
            "stim2": "b",
            "mem2": "purple",
            "response": "g",
        }
        for phase1 in phase_dict.keys():
            ax_input1.plot(
                inputs[phase_dict[phase1][0] : phase_dict[phase1][1], 1],
                inputs[phase_dict[phase1][0] : phase_dict[phase1][1], 2],
                c=color_dict[phase1],
            )

            ax_input2.plot(
                inputs[phase_dict[phase1][0] : phase_dict[phase1][1], 3],
                inputs[phase_dict[phase1][0] : phase_dict[phase1][1], 4],
                c=color_dict[phase1],
            )

            ax_output.scatter(
                outputs[phase_dict[phase1][0] : phase_dict[phase1][1], 1],
                outputs[phase_dict[phase1][0] : phase_dict[phase1][1], 2],
                c=color_dict[phase1],
            )
            ax_labels.scatter([0, 0], [0, 0], c=color_dict[phase1], label=phase1)
        ax_labels.legend()

        ax_input1.set_title("Input 1")
        ax_input2.set_title("Input 2")
        ax_output.set_title("Output")

        ax_input1.set_xlim(ax_limits)
        ax_input1.set_ylim(ax_limits)

        ax_input2.set_xlim(ax_limits)
        ax_input2.set_ylim(ax_limits)

        ax_output.set_xlim(ax_limits)
        ax_output.set_ylim(ax_limits)            

        ax_input1.set_xticklabels([])
        ax_input1.set_yticklabels([])
        ax_input2.set_xticklabels([])
        ax_input2.set_yticklabels([])
        ax_output.set_xticklabels([])
        ax_output.set_yticklabels([])
        ax_labels.set_xticklabels([])
        ax_labels.set_yticklabels([])

        ax_input1.set_aspect("equal", adjustable="box")
        ax_input2.set_aspect("equal", adjustable="box")
        ax_output.set_aspect("equal", adjustable="box")
        ax_labels.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.suptitle(f"Trial: {self.task_name}")
        # plt.savefig(f"{self.task_name}_radial.png")

        plt.show()

    def reset(self):
        return super().reset()

    def step(self, action):
        return super().step(action)
