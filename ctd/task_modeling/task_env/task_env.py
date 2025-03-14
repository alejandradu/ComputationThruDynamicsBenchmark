# Class to generate training data for task-trained RNN that does 3 bit memory task
from abc import ABC, abstractmethod

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from ctd.task_modeling.task_env.loss_func import NBFFLoss, ClicksLoss


class DecoupledEnvironment(gym.Env, ABC):
    """
    Abstract class representing a decoupled environment.
    This class is abstract and cannot be instantiated.

    """

    # All decoupled environments should have
    # a number of timesteps and a noise parameter

    @abstractmethod
    def __init__(self, n_timesteps: int, noise: float):
        super().__init__()
        self.dataset_name = "DecoupledEnvironment"
        self.n_timesteps = n_timesteps
        self.noise = noise

    # All decoupled environments should have
    # functions to reset, step, and generate trials
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_dataset(self, n_samples):
        """Must return a dictionary with the following keys:
        #----------Mandatory keys----------
        ics: initial conditions
        inputs: inputs to the environment
        targets: targets of the environment
        conds: conditions information (if applicable)
        extra: extra information
        #----------Optional keys----------
        true_inputs: true inputs to the environment (if applicable)
        true_targets: true targets of the environment (if applicable)
        phase_dict: phase information (if applicable)
        """

        pass


class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy environment where the goal is to flip the required bit.
    """

    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        n=1,
        switch_prob=0.01,
        transition_blind=4,
        dynamic_noise=0,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = f"{n}BFF"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(n,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )
        self.n = n
        self.state = np.zeros(n)
        self.input_labels = [f"Input {i}" for i in range(n)]
        self.output_labels = [f"Output {i}" for i in range(n)]
        self.noise = noise
        self.dynamic_noise = dynamic_noise
        self.coupled_env = False
        self.switch_prob = switch_prob
        self.transition_blind = transition_blind
        self.loss_func = NBFFLoss(transition_blind=transition_blind)

    def set_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """
        Generates a state update given an input to the flip-flop

        TODO: Revise

        Args:
            action (TODO: dtype) :

        Returns:
            None
        """
        # Generates state update given an input to the flip-flop
        for i in range(self.n):
            if action[i] == 1:
                self.state[i] = 1
            elif action[i] == -1:
                self.state[i] = 0

    def generate_trial(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand = np.random.random(size=(self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :] = 0

        # Generate the desired outputs given the inputs
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :] = self.state

        # Add noise to the inputs for the trial
        true_inputs = inputs
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def generate_trial_with_stim(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand = np.random.random(size=(self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :] = 0

        bit = np.random.randint(0, self.n)
        ind_25 = int(self.n_timesteps / 4)
        ind_75 = int(3 * self.n_timesteps / 4)
        stim_mag = np.random.uniform(0.5, 1.5)
        inputs[ind_25:ind_75, bit] += stim_mag
        # pick a random bit to add a step function to

        # Generate the desired outputs given the inputs
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :] = self.state

        # Add noise to the inputs for the trial
        true_inputs = inputs
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def reset(self):
        """
        Resets the state of the flip-flop

        TODO: Revise

        Args:
            None

        Returns:
            state (TODO: dtype) :
        """
        self.state = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples, stim=False):
        """
        Generates a dataset for the NBFF task

        TODO: Revise

        Args:
            n_samples (int) :

        Returns:
            dataset_dict (dict) :
            extra_dict (dict) :
        """
        # Generates a dataset for the NBFF task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.n))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            if stim:
                inputs, outputs, true_inputs = self.generate_trial_with_stim()
            else:
                inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.zeros(shape=(n_samples, 1)),
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def render(self):
        inputs, states, _ = self.generate_trial()
        fig1, axes = plt.subplots(nrows=self.n + 1, ncols=1, sharex=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n))
        for i in range(self.n):
            axes[i].plot(states[:, i], color=colors[i])
            axes[i].set_ylabel(f"State {i}")
            axes[i].set_ylim(-0.2, 1.2)
        ax2 = axes[-1]
        for i in range(self.n):
            ax2.plot(inputs[:, i], color=colors[i])
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Inputs")
        plt.tight_layout()
        plt.show()
        fig1.savefig("nbitflipflop.pdf")

    def render_3d(self, n_trials=10):
        if self.n > 2:
            fig = plt.figure(figsize=(5 * n_trials, 5))
            # Make colormap for the timestep in a trial
            for i in range(n_trials):

                ax = fig.add_subplot(1, n_trials, i + 1, projection="3d")
                inputs, states, _ = self.generate_trial()
                ax.plot(states[:, 0], states[:, 1], states[:, 2])
                ax.set_xlabel("Bit 1")
                ax.set_ylabel("Bit 2")
                ax.set_zlabel("Bit 3")
                ax.set_title(f"Trial {i+1}")
            plt.tight_layout()
            plt.show()


class PClicks(DecoupledEnvironment):
    """
    Simulate the auditory clicks task from the paper:
    https://www.biorxiv.org/content/10.1101/2023.10.15.562427v3.full.pdf
    easiest: 39:1, hardest: 26:14
    """
    
    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        rateL=30,  # Hz,
        **kwargs   # latent_l2_wt
    ):
        self.dataset_name = "PClicks"
        self.n_timesteps = n_timesteps
        self.noise = noise
        self.rateL = rateL
        self.rateR = 40 - rateL
        self.fixation_period = 1.5  # seconds
        self.response_period = 0.1  # seconds - this is me guessing
        self.delay_min = 0.5  # seconds
        self.delay_max = 1.3  # seconds
        self.memory = np.zeros(2)  # memory[0] = left, memory[1] = right
        self.state = 0 # 0 = head fixed no response
        self.LEFT = -1
        self.RIGHT = 1
        self.FIX = 0
        self.INPUT_SIZE = 3
        self.OUTPUT_SIZE = 1
        self.stim_end = None
        
        if int(self.fixation_period / self.response_period) > self.n_timesteps:
            raise ValueError("Increase n_timesteps. Response period must be 1 at least")
        
        if noise > 0.5:
            raise ValueError("Noise 0.5 is too high be fr")
        
        # NOTE: adding 0.5 to make room for added noise (and too much noise will make it fail)
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(self.OUTPUT_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(self.INPUT_SIZE,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )
        
        self.latent_l2_wt = kwargs.get("latent_l2_wt", 1.0)
        self.loss_func = ClicksLoss(lat_loss_weight=self.latent_l2_wt)
        
    def step(self, action):
        fix = action[0]  # 1 for fix (don't respond), 0 for not fix (respond)
        left = action[1]
        right = action[2]
        
        if fix == 1:
            # still register the stimuli
            self.memory += action[1:]
            self.state = self.FIX
        else:
            # if tie, just stays at the last state (unlikely)
            if self.memory[0] > self.memory[1]:
                self.state = self.LEFT
            elif self.memory[1] > self.memory[0]:
                self.state = self.RIGHT
        
    def reset(self):
        # erase memory and set back to on fixation
        self.memory = np.zeros(2)
        self.state = 0
        return self.state
    
    def make_pclicks(self, rate, n_timesteps):
        # sample from a Bernoulli dist, rate has to be in timestep units
        p = 1 - np.exp(-rate)  
        return np.random.rand(n_timesteps) < p 

    def generate_trial(self):
        # start from no bias always
        self.reset()
        
        # random delay to start the clicks for each trial
        delay = np.random.uniform(self.delay_min, self.delay_max)
        trial_duration = self.fixation_period + self.response_period
        dt = trial_duration / self.n_timesteps   # sec / step 
        
        stim_start = int(delay / dt)  # index to start the clicks
        stim_end = int(self.fixation_period / dt)  # index to end the clicks
        
        # store this marker
        if self.stim_end is None:
            self.stim_end = stim_end
        
        left_clicks = self.make_pclicks(self.rateL * dt, self.n_timesteps)
        right_clicks = self.make_pclicks(self.rateR * dt, self.n_timesteps)
        # adjust for the delay 
        left_clicks[:stim_start] = 0
        right_clicks[:stim_start] = 0
        # null in reponse period
        left_clicks[stim_end:] = 0
        right_clicks[stim_end:] = 0
        
        # add stereoclick to the very first click after the delay
        left_clicks[stim_start] = 1
        right_clicks[stim_start] = 1
        
        # fixation signal
        fixation = np.zeros(self.n_timesteps)
        fixation[:stim_end] = 1
        inputs = np.stack([fixation, left_clicks, right_clicks], axis=1)
        
        # generate desired outputs
        outputs = np.zeros(self.n_timesteps)
        for i in range(self.n_timesteps):
            self.step(inputs[i,:])
            outputs[i] = self.state
            
        # Add noise (not to the fixation cue)
        true_inputs = inputs
        inputs[:, 1:] = inputs[:, 1:] + np.random.normal(0, self.noise, size=(self.n_timesteps, self.INPUT_SIZE-1))
        
        return inputs, outputs, true_inputs


    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.INPUT_SIZE))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.OUTPUT_SIZE))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))
        for i in range(n_samples):
            inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # stim_end marker is the same for all trials
            "extra": np.ones(shape=(n_samples, 1))*self.stim_end,
        }
        extra_dict = {}
        return dataset_dict, extra_dict


    def render(self):
        inputs, outputs, _ = self.generate_trial()
        fig1, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        colors = plt.cm.viridis(np.linspace(0, 1, 3))
        # first row is the fixation signal
        axes[0].plot(inputs[:, 0], color=colors[0])
        axes[0].set_ylabel("fixation cue")
        # second row is left and right clicks
        axes[1].plot(self.LEFT*inputs[:, 1], color=colors[1]) #left
        axes[1].plot(self.RIGHT*inputs[:, 2], color=colors[2]) # right
        axes[1].set_ylabel("clicks")
        # third row is the expected output
        axes[2].plot(outputs, color=colors[0])
        axes[2].set_ylabel("target output")
        plt.tight_layout()
        plt.show()
        
        
        
class MarinoPagan(DecoupledEnvironment):
    """
    Simulate the context-dependent decision making + evidence accumulation
    task from the paper:
    https://www.nature.com/articles/s41586-024-08433-6
    """
    
    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        rateL=30,  # Hz
        **kwargs   # latent_l2_wt
    ):
        self.dataset_name = "MarinoPagan"
        self.n_timesteps = n_timesteps
        self.noise = noise
        self.rateL = rateL
        self.rateR = 40 - rateL
        self.ctx_period = 1.0  # seconds (context cue) 
        self.fixation_period = 1.3  # seconds (all stimuli no delay)
        self.response_period = 0.1  # this is me guessing
        self.memory = np.zeros(4)  # memory[1-2] = loc, memory[3-4] = freq
        self.state = 0  # 0 = head fixed no response
        self.ctx = None
        self.LEFT = -1
        self.RIGHT = 1
        self.HI = 1
        self.LO = -1
        self.FIX = 0
        self.LOC = -1  # loc context
        self.FREQ = 1  # freq context
        # fixation, context, 2 loc pulses, 2 freq pulses
        self.INPUT_SIZE = 6
        self.OUTPUT_SIZE = 1
        self.stim_end = None
        
        if int(self.fixation_period / self.response_period) > self.n_timesteps:
            raise ValueError("Increase n_timesteps. Response period must be 1 at least")
        
        if noise > 0.5:
            raise ValueError("Noise 0.5 is too high be fr")
        
        # NOTE: adding 0.5 to make room for added noise (and too much noise will make it fail)
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(self.OUTPUT_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(self.INPUT_SIZE-1,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(1,), dtype=np.float32
        )
        
        self.latent_l2_wt = kwargs.get("latent_l2_wt", 1.0)
        self.loss_func = ClicksLoss(lat_loss_weight=self.latent_l2_wt)
        
    def step(self, action):
        # action[2:] = left, right, hi, lo
        fix = action[0]  # 1 for fix (don't respond), 0 for not fix (respond)
        ctx = action[1]  # -1 for loc, 1 for freq
        
        # set the context - state stays at 0 (irl rat could move head)
        if ctx != 0:
            self.ctx = ctx
        # don't respond if fixation is on, still register stimuli
        if fix == 1:
            self.memory += action[2:]
            self.state = self.FIX
        else:
            # if tie, just stays at the last state (unlikely)
            # loc context
            if self.ctx == self.LOC:
                if self.memory[0] > self.memory[1]:
                    self.state = self.LEFT
                elif self.memory[1] > self.memory[0]:
                    self.state = self.RIGHT
            elif self.ctx == self.FREQ:
                # turn right if more hi pulses
                if self.memory[3] > self.memory[2]:
                    self.state = self.RIGHT
                elif self.memory[2] > self.memory[3]:
                    self.state = self.LEFT
            else:
                raise ValueError("ctx not set properly")
            
    def reset(self):
        # erase memory and set back to on fixation
        self.memory = np.zeros(4)
        self.state = 0
        self.ctx = None
        return self.state
    
    def make_pclicks_labeled(self, rate, n_timesteps):
        # sample from a Bernoulli dist, rate has to be in timestep units
        p = 1 - np.exp(-rate)  
        clicks = np.random.rand(n_timesteps) < p 
        labels = np.random.rand(n_timesteps) < 0.5 
        clicks_hi = np.logical_and(clicks, labels)  # high frequency
        clicks_lo = np.logical_and(clicks, np.logical_not(labels))  # low frequency
        return clicks, clicks_hi, clicks_lo
    
    def generate_trial(self):
        # start from no bias always
        self.reset()
        
        trial_duration = self.ctx_period + self.fixation_period + self.response_period
        dt = trial_duration / self.n_timesteps   # sec / step 
        
        stim_start = int(self.ctx_period / dt)
        stim_end = int((self.ctx_period + self.fixation_period) / dt)
        
        # store this marker
        if self.stim_end is None:
            self.stim_end = stim_end
        
        # initial context cue
        ctx = np.zeros(self.n_timesteps)
        ctx[0:stim_start] = np.random.choice([-1,1])
        
        left, left_hi, left_lo = self.make_pclicks_labeled(self.rateL * dt, self.n_timesteps)
        right, right_hi, right_lo = self.make_pclicks_labeled(self.rateR * dt, self.n_timesteps)
        hi = left_hi + right_hi
        lo = left_lo + right_lo
        
        # fixation signal
        fixation = np.zeros(self.n_timesteps)
        fixation[stim_start:stim_end] = 1
        
        inputs_to_plot = np.stack([left_hi, left_lo, right_hi, right_lo], axis=1)
        inputs = np.stack([fixation, ctx, left, right, hi, lo], axis=1)
        
        # null in ctx
        inputs_to_plot[0:stim_start, :] = 0
        inputs[0:stim_start, 2:] = 0
        # null in response period
        inputs_to_plot[stim_end:, :] = 0
        inputs[stim_end:, 2:] = 0
        
        # generate desired outputs
        outputs = np.zeros(self.n_timesteps)
        for i in range(self.n_timesteps):
            self.step(inputs[i,:])
            outputs[i] = self.state
            
        # Add noise (not to the fixation or ctx cue)
        true_inputs = inputs
        inputs[:, 2:] = inputs[:, 2:] + np.random.normal(0, self.noise, size=(self.n_timesteps, self.INPUT_SIZE-2))
        
        return inputs, outputs, true_inputs, inputs_to_plot
    
    def generate_dataset(self, n_samples):
        # Generates a dataset for the MarinoPagan task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.INPUT_SIZE))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.OUTPUT_SIZE))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.INPUT_SIZE))

        for i in range(n_samples):
            inputs, outputs, true_inputs, _ = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs
            
            
        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.ones(shape=(n_samples, 1))*self.stim_end,
        }
        
        extra_dict = {}
        
        return dataset_dict, extra_dict
    
    def render(self):
        inputs, outputs, _ , inputs_to_plot = self.generate_trial()
        fig1, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(7, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        # first row is the fixation and context signal
        axes[0].plot(inputs[:, 0], color=colors[0], label='fixation')
        axes[0].plot(inputs[:, 1], color=colors[5], label='context')
        axes[0].set_yticks([self.LOC, self.FREQ])
        axes[0].set_yticklabels(['(ctx) loc', '(ctx) freq'])
        axes[0].set_ylabel("cues")
        axes[0].legend()
        axes[0].set_ylim([-1.1, 1.1])
        # second row is left and right clicks
        axes[1].plot(self.LEFT*inputs[:, 2], color=colors[1])
        axes[1].plot(self.RIGHT*inputs[:, 3], color=colors[2])
        axes[1].set_ylabel("loc")
        axes[1].set_yticks([self.LEFT, self.RIGHT])
        axes[1].set_yticklabels(['L', 'R'])
        # third row is hi and lo clicks
        axes[2].plot(self.HI*inputs[:, 4], color=colors[3])
        axes[2].plot(self.LO*inputs[:, 5], color=colors[4])
        axes[2].set_ylabel("freq")
        axes[2].set_yticks([self.HI, self.LO])
        axes[2].set_yticklabels(['Hi', 'Lo'])
        # fourth row is combined clicks
        axes[3].plot(self.RIGHT*inputs_to_plot[:, 2], color='tomato', label='Hi')
        axes[3].plot(self.RIGHT*inputs_to_plot[:, 3], color='maroon', label='Lo')
        axes[3].plot(self.LEFT*inputs_to_plot[:, 0], color='royalblue', label='Hi')
        axes[3].plot(self.LEFT*inputs_to_plot[:, 1], color='navy', label='Lo')
        axes[3].legend()
        axes[3].set_ylabel("combined by loc")
        axes[3].set_yticks([self.LEFT, self.RIGHT])
        axes[3].set_yticklabels(['L', 'R'])
        # fourth row is the expected output
        axes[4].plot(outputs, color='k')
        axes[4].set_ylabel("target")
        axes[4].set_yticks([self.HI, self.LO])
        axes[4].set_yticklabels(['R/Hi', 'L/Lo'])
        
        plt.tight_layout()
        plt.show()
        