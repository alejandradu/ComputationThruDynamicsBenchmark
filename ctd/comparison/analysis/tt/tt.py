import os
import pickle
from pathlib import Path
from matplotlib import cm

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
# from DSA.stats import dsa_bw_data_splits, dsa_to_id
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ctd.comparison.analysis.analysis import Analysis
from ctd.task_modeling.model.rnn import FullRankRNNCell, LowRankRNNCell
from ctd.comparison.fixedpoints import find_fixed_points
from ctd.task_modeling.task_env.task_env import DecoupledEnvironment

dotenv.load_dotenv(override=True)
HOME_DIR = os.getenv("HOME_DIR")


class Analysis_TT(Analysis):
    def __init__(self, run_name, filepath, use_train_dm=False):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath, use_train_dm)
        self.run_hps = None

    def load_wrapper(self, filepath, use_train_dm=False):

        with open(filepath + "model.pkl", "rb") as f:
            self.wrapper = pickle.load(f)
        self.env = self.wrapper.task_env
        self.model = self.wrapper.model
        if use_train_dm:
            with open(filepath + "datamodule_train.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        else:
            with open(filepath + "datamodule_sim.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        # self.env = self.datamodule.data_env.dataset_name
        # if the simulator exists
        if Path(filepath + "simulator.pkl").exists():
            with open(filepath + "simulator.pkl", "rb") as f:
                self.simulator = pickle.load(f)
        n_train = len(self.datamodule.train_ds)
        n_val = len(self.datamodule.valid_ds)
        self.n_trials = n_train + n_val
        self.train_inds = range(0, int(0.8 * self.n_trials))
        self.valid_inds = range(int(0.8 * self.n_trials), self.n_trials)
        self.trim_inds = self.simulator.trim_inds

    def get_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[1], valid_ds.tensors[1]], dim=0)
        if phase == "all":
            return tt_inputs
        elif phase == "train":
            return tt_inputs[self.train_inds]
        elif phase == "val":
            return tt_inputs[self.valid_inds]

    def get_true_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[7], valid_ds.tensors[7]], dim=0)
        if phase == "all":
            return tt_inputs
        elif phase == "train":
            return tt_inputs[self.train_inds]
        elif phase == "val":
            return tt_inputs[self.valid_inds]

    def get_inputs_to_env(self, phase="all"):
        if phase == "all":
            train_inputs_to_env = self.datamodule.train_ds.tensors[6]
            valid_inputs_to_env = self.datamodule.valid_ds.tensors[6]
            return torch.cat([train_inputs_to_env, valid_inputs_to_env], dim=0)
        elif phase == "train":
            return self.datamodule.train_ds.tensors[6]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[6]

    def get_model_inputs(self, phase="all"):

        if phase == "all":
            train_ics = self.datamodule.train_ds.tensors[0]
            train_inputs = self.datamodule.train_ds.tensors[1]
            train_targets = self.datamodule.train_ds.tensors[2]
            valid_ics = self.datamodule.valid_ds.tensors[0]
            valid_inputs = self.datamodule.valid_ds.tensors[1]
            valid_targets = self.datamodule.valid_ds.tensors[2]
            tt_ics = torch.cat([train_ics, valid_ics], dim=0)
            tt_inputs = torch.cat([train_inputs, valid_inputs], dim=0)
            tt_targets = torch.cat([train_targets, valid_targets], dim=0)
            return tt_ics, tt_inputs, tt_targets
        elif phase == "train":
            return (
                self.datamodule.train_ds.tensors[0],
                self.datamodule.train_ds.tensors[1],
                self.datamodule.train_ds.tensors[2],
            )
        elif phase == "val":
            return (
                self.datamodule.valid_ds.tensors[0],
                self.datamodule.valid_ds.tensors[1],
                self.datamodule.valid_ds.tensors[2],
            )

    def get_extra_inputs(self, phase="all"):
        if phase == "all":
            train_extra = self.datamodule.train_ds.tensors[5]
            valid_extra = self.datamodule.valid_ds.tensors[5]
            tt_extra = torch.cat([train_extra, valid_extra], dim=0)
            return tt_extra
        elif phase == "train":
            return self.datamodule.train_ds.tensors[5]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[5]

    def get_model_inputs_noiseless(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)

        train_noiseless_inputs = self.datamodule.train_ds.tensors[7]
        valid_noiseless_inputs = self.datamodule.valid_ds.tensors[7]
        tt_noiseless_inputs = torch.cat(
            [train_noiseless_inputs, valid_noiseless_inputs], dim=0
        )

        if phase == "all":
            return tt_ics, tt_noiseless_inputs, tt_targets
        elif phase == "train":
            return tt_ics, train_noiseless_inputs, tt_targets
        elif phase == "val":
            return tt_ics, valid_noiseless_inputs, tt_targets

    def get_model_outputs(self, phase="all"):
        inputs_to_env = self.get_inputs_to_env(phase=phase)
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env)
        return out_dict

    def get_model_outputs_noiseless(self, phase="all"):
        inputs_to_env = self.get_inputs_to_env(phase=phase)
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs_noiseless(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env)
        return out_dict

    def get_latents(self, phase="all"):
        out_dict = self.get_model_outputs(phase=phase)
        if self.trim_inds is not None:
            return out_dict["latents"][:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
        else:
            return out_dict["latents"]

    def get_latents_noiseless(self, phase="all"):
        out_dict = self.get_model_outputs_noiseless(phase=phase)
        if self.trim_inds is not None:
            return out_dict["latents"][:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
        else:
            return out_dict["latents"]

    def get_latents_pca(self, num_PCs=3):
        latents = self.get_latents()
        B, T, N = latents.shape
        latents = latents.reshape(-1, N)
        pca = PCA(n_components=num_PCs)
        latents_pca = pca.fit_transform(latents)
        latents_pca = latents.reshape(B, T, num_PCs)
        return latents_pca, pca
    
    def plot_trial_latents(self, num_trials=10, common_basis=True, pca=False, n_components=3, avg_per_rate=True):
        """
        Plot latent trajectories for trials ran during training, with
        predetermined train/val inputs.
        """
        out_dict = self.get_model_outputs()
        _, inputs_latents, _ = self.get_model_inputs()
        latents = out_dict["latents"].detach().numpy()
        labels = self.get_extra_inputs().detach().numpy()

        # Check if the latent dimension is at least equal to n_components
        if latents.shape[-1] < n_components:
            raise ValueError(f"Latent dimension ({latents.shape[-1]}) must be at least equal to n_components ({n_components}).")

        # Apply PCA if specified
        if pca:
            pca_model = PCA(n_components=n_components)
            B, T, N = latents.shape
            latents = pca_model.fit_transform(latents.reshape(-1, N))
            latents = latents.reshape(B, T, n_components)

        # Average latents if specified
        if avg_per_rate:
            fig, ax = plt.subplots()
            unique_labels = np.unique(labels[:, 1])
            for i, label in enumerate(unique_labels):
                # Find the indices of latents with the current label
                indices = np.where(labels[:, 1] == label)[0]
                # Get the aligned average of the latents by phase
                delay, stim, resp = self.average_latents_by_phase(latents[indices], inputs_latents[indices])
                
                # if all inputs were zero (no stim - don't need phases)
                if delay == 0 and stim == 0 and resp == 0:
                    print("Plotting for all zero inputs - no phase averaging")
                    cat = np.mean(latents[indices], axis=0)
                else:  
                    print("Plotting for non-trivial inputs with phase averaging")
                    #cat = np.concatenate((delay, stim, resp), axis=1)
                    
                norm_labels = plt.Normalize(1,39)
                # concatenate to make a continuous time series and color by label
                ax.plot(*cat.T, linewidth=1.5, color=cm.coolwarm(norm_labels(label)))
                
            return

        # Determine plot type (2D or 3D)
        is_3d = n_components == 3 or (not pca and latents.shape[-1] == 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)

        # Use a colormap to plot the trials
        colors = cm.viridis(np.linspace(0, 1, num_trials))

        # Plot latents
        for i in range(latents.shape[0]):
            if is_3d:
                ax.plot(latents[i, :, 0], latents[i, :, 1], latents[i, :, 2], color=colors[i])
            else:
                ax.plot(latents[i, :, 0], latents[i, :, 1], color=colors[i])

        # Set grid color to white and adjust axis labels
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.show()
        
    def plot_flow_fieldLR(self, latents_range: list, num_points:int, input_field: np.array=None,
                          vec1: np.array=None, vec2: np.array=None, orth=False, sizes=1.):
        """
        Plot 2d flow field and eventually fixed points for a rank 2 LowRankRNN. Can plot the affine flow field in presence of a
        constant input with argument input.
        
        :param vec1: None or a numpy array of shape (hidden_size). If None, will be taken as vector m1 of the network
        :param vec2: same with m2
        :param input: None or torch tensor of shape (n_inputs), provides constant input for plotting affine flow field
        :param orth: bool, if True, start by orthogonalizing (vec1, vec2)
        :param sizes: float, general scaling factor for arrows
        
        """
        
        model = self.wrapper.model
        lr_cell = model.cell
        m = lr_cell.recW2.weight
        n = lr_cell.recW1.weight.t()
        
        if vec1 is None:
            vec1 = m[:, 0].squeeze().detach().numpy()
        if vec2 is None:
            vec2 = m[:, 1].squeeze().detach().numpy()
            
        m = m.detach().numpy()
        n = n.detach().numpy()
        
        # Orthogonalization of the basis vec1, vec2, I - gram schmidt
        if orth:
            vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
        if input_field is not None:
            # inpW.weight has dimension (input x hidden)
            I = (input_field @ lr_cell.inpW.weight).detach().numpy()
            I_orth = I - (I @ vec1) * vec1 / (vec1 @ vec1) - (I @ vec2) * vec2 / (vec2 @ vec2)
        else:
            I = np.zeros(model.latent_size)
            I_orth = np.zeros(model.latent_size)
            
        # Rescaling the space
        r1 = model.latent_size / (vec1 @ vec1)
        r2 = model.latent_size / (vec2 @ vec2)
            
        # Define the grid
        x = np.linspace(latents_range[0][0], latents_range[0][1], num_points+1)
        y = np.linspace(latents_range[1][0], latents_range[1][1], num_points+1)
        x_mpts = (x[1:] + x[:-1]) / 2
        y_mpts = (y[1:] + y[:-1]) / 2
        field = np.zeros((num_points, num_points, 2))
        U, V = np.meshgrid(x_mpts, y_mpts)
        
        fig, ax = plt.subplots()
        
        # BUG: potentially don't think i need it...
        # adjust shape of input field for my model
        # input_field = torch.unsqueeze(I, 0)
        
        # velocity field from the defining ODE
        def dh_dt(I, hidden):
            # using all in numpy for speed
            return -hidden + m @ (n.T @ np.tanh(hidden)) / model.latent_size + I + lr_cell.bias.detach().numpy()
        
        # Compute flow in each point of the grid
        for i, x in enumerate(x_mpts):
            for j, y in enumerate(y_mpts):
                hidden = r1 * x * vec1 + r2 * y * vec2 + I_orth
                # NOTE: velocity specific to a lrRNN
                delta = dh_dt(I, hidden)
                field[j, i, 0] = delta @ vec1
                field[j, i, 1] = delta @ vec2
        ax.streamplot(x_mpts, y_mpts, field[:, :, 0], field[:, :, 1], color='white', density=0.5, arrowsize=sizes,
                      linewidth=sizes*.8)
        norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)
        mappable = ax.pcolor(U, V, norm_field)
        
        return ax, mappable
        
    
    def average_latents_by_phase(self, inputs, latents):
        """Align time series and take an average over each phase
        Separates latents during fixation, stimulus, and response
        
        Returns: aligned average for delay, stimulus, and response
                 0,0,0 if inputs are all 0 and no phases
        """
        trials, timesteps, input_dim = inputs.shape
        trials, timesteps, latent_dim = latents.shape
        stim_onset = []
        
        # get the onset of stimulus (the stereoclick)
        for i, trial in enumerate(inputs):
            index = np.where(trial[:, 1] == 1)[0]  # first left stereoclick
            # return if index is empty
            if len(index) == 0:
                return 0, 0, 0
            stim_onset.append(index[0])
            fix_off = np.where(trial[:, 1] == 0)[0][0] # fix off - same for all
            
        delay = np.empty((trials, max(stim_onset), latent_dim))
        stim = np.empty((trials, fix_off - min(stim_onset), latent_dim))
        resp = np.empty((trials, timesteps - fix_off, latent_dim))
        
        for i, trial in enumerate(latents):
            delay[i, :stim_onset[i], :] = latents[i, :stim_onset[i], :]
            stim[i, stim_onset[i]:fix_off, :] = latents[i, stim_onset[i]:fix_off, :]
            resp[i, fix_off:, :] = latents[i, fix_off:, :]
            
        # take average over trials ignoring NaN
        return np.nanmean(delay, axis=0), np.nanmean(stim, axis=0), np.nanmean(resp, axis=0)
        
        
    def plot_flow_field(self, latents_range: list, num_points: int, inputs_latents: np.array, input_field: np.array,  
                    input_latents_extra: np.array=None, custom_n_timesteps: int=None, n_trials=10, 
                    scatter_trajectories=False, xstar=None, q_flag=None, colors_fps=None,  cmap=plt.cm.pink, 
                    plot_wrapper_trajs=False, filter_pc_rate:int =None, avg_per_rate=False, lint_plot_style=False,
                    cmap_field=plt.get_cmap('pink'), cmap_time=plt.get_cmap('copper'), cmap_rate=plt.get_cmap('coolwarm'),
                    ics_noise=None, **kwargs):
        """
        Plot the velocity flow field for a previously trained NODE model. 
        Args:
            latents_range (list): range of each axis on the grid
            num_points (int): to set the grid
            inputs_latents (np.array):(n_trials, n_timesteps, input_dim) array to draw trajectories 
            inputs_latents_extra (np.array): (n_trials, n_timesteps, label) array to draw trajectories, get from input_dataset_dict
            input_field (np.array): flat array (input_dim) - fixed inputs to get the velocities
            custom_n_timesteps: number of timestep to simulate if different from original task_env of training
            n_trials (int, optional): Number of trials to plot. Defaults to 10.
            scatter_trajectories (bool, optional): True to plot the trajectories with a colormap indicating time evolution. Defaults to False.
            xstar (None, optional): Fixed points to plot. Defaults to None.
            q_flag (None, optional): Flag to indicate which fixed points to plot. Defaults to None.
            colors_fps (None, optional): Colors for the fixed points. Defaults to None.
            cmap (colormap, optional): Colormap for the flow field plot. Defaults to plt.cm.pink.
            plot_saved_trajs (bool, optional): True to plot the trajectories saved during training. Defaults to False.
            filter_pc_rate (int): set to a rate to filter when plotting trajectories
            avg_per_rate (bool): set to True to average trajectories per rate
        """
        
        if hasattr(self.wrapper.model, "generator"):
            model = self.wrapper.model.generator
        elif hasattr(self.wrapper.model, "cell"):
            model = self.wrapper.model.cell
        else:
            raise ValueError("No generator or cell found in model")
        
        # input shape should match n_dimension
        tt_ics, correct_inputs, _ = self.get_model_inputs(phase='all')
        if inputs_latents.shape[-1] != correct_inputs.shape[-1]: 
            raise ValueError("inputs_latents should have last dimension: ", correct_inputs.shape[-1])
        elif input_field.shape[0] != correct_inputs.shape[-1]:
            raise ValueError("input_field should have last dimension: ", correct_inputs.shape[-1])
        else:
            inputs_latents = torch.tensor(inputs_latents, dtype=torch.float32)  #from numpy to tensor
            input_field = torch.tensor(input_field, dtype=torch.float32)  #from numpy to tensor
        # get the latents for as many trials as in inputs_latents
        inputs_to_env = self.get_inputs_to_env(phase="all")  # TODO: is inputs_to_env, tt_ics an issue?
        # same number of trials
        n, t, _ = inputs_latents.shape
        tt_ics = tt_ics[:n]
        inputs_to_env = inputs_to_env[:n]
        
        if plot_wrapper_trajs:
            latents = self.get_latents().detach().numpy()
            labels = self.get_extra_inputs().detach().numpy()
        else:
            if custom_n_timesteps is not None and custom_n_timesteps > t:
                raise ValueError("got more timesteps than in inputs_latents. Reduce custom_n_timesteps")
            # run inference with custom value
            out_dict = self.wrapper(tt_ics, inputs_latents, inputs_to_env, custom_n_timesteps=custom_n_timesteps)
            latents = out_dict["latents"].detach().numpy()
            if input_latents_extra is not None:
                labels = input_latents_extra
            
        if latents.shape[-1] > 3:
            raise ValueError("Latents have more than 3 dimensions. Not supported now")
        elif latents.shape[-1] != len(latents_range):
            raise ValueError("Adjust latents_range to dimension ", latents.shape[-1])
        
        input_field = torch.unsqueeze(input_field, 0)
        
        fig, ax = plt.subplots()
        
        if lint_plot_style and latents.shape[-1] == 2:
            x = np.linspace(latents_range[0][0], latents_range[0][1], num_points+1)
            y = np.linspace(latents_range[1][0], latents_range[1][1], num_points+1)
            x_mpts = (x[1:] + x[:-1]) / 2
            y_mpts = (y[1:] + y[:-1]) / 2
            field = np.zeros((num_points, num_points, 2))
            X, Y = np.meshgrid(x_mpts, y_mpts)
            for i, x in enumerate(x_mpts):
                for j, y in enumerate(y_mpts):
                    state = torch.tensor([[x, y]], dtype=torch.float)
                    # NOTE: keep the indexing like ji
                    field[j,i,:] = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy()
            ax.streamplot(x_mpts, y_mpts, field[:, :, 0], field[:, :, 1], color='white', density=1., arrowsize=1.,
                          linewidth=1.*.8)
            norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)        
            mappable = ax.pcolor(X, Y, norm_field, cmap=cmap_field)
            
        else:
            num_points = int(num_points / 3)
            # Calculate velocities over a grid using a double for loop implementation
            x = np.linspace(latents_range[0][0], latents_range[0][1], num_points)
            y = np.linspace(latents_range[1][0], latents_range[1][1], num_points)
            if len(latents_range) == 3:
                z = np.linspace(latents_range[2][0], latents_range[2][1], num_points)
            if len(latents_range) == 2:
                U = np.zeros([num_points, num_points])
                V = np.zeros([num_points, num_points])
            else:
                U = np.zeros([num_points, num_points, num_points])
                V = np.zeros([num_points, num_points, num_points])
                W = np.zeros([num_points, num_points, num_points])
            for i in range(num_points):
                for j in range(num_points):
                    state = torch.tensor([[x[i], y[j]]], dtype=torch.float)
                    if len(latents_range) == 2:
                        # NOTE: this is only applicable to a NODE w/ leak term
                        U[i, j], V[i, j] = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy().flatten()
                    else:
                        for k in range(num_points):
                            state = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float)
                            U[i, j, k], V[i, j, k], W[i, j, k] = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy().flatten()
            # Create a colormap based on the normalized magnitude
            if len(latents_range) == 2:
                magnitude = np.sqrt(U**2 + V**2)
            else:
                magnitude = np.sqrt(U**2 + V**2 + W**2)
            normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            colors_map = cmap_field(normalized_magnitude.flatten())
            # Plot the velocity field
            if len(latents_range) == 2:
                ax.quiver(*np.meshgrid(x, y, indexing='ij'), U, V, color=colors_map)
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(*np.meshgrid(x, y, z), U, V, W, color=colors_map)
            
        if n_trials > latents.shape[0]:
            n_trials = latents.shape[0]
            
        # get an average latent per rate
        if avg_per_rate:
            unique_labels = np.unique(labels[:, 1])
            
            for i, label in enumerate(unique_labels):
                if filter_pc_rate is not None and label != filter_pc_rate:
                    continue
                
                # Find the indices of latents with the current label
                indices = np.where(labels[:, 1] == label)[0]
                # Get the aligned average of the latents by phase
                delay, stim, resp = self.average_latents_by_phase(latents[indices], inputs_latents[indices])
                
                # if all inputs were zero (no stim - don't need phases)
                if delay == 0 and stim == 0 and resp == 0:
                    print("Plotting for all zero inputs - no phase averaging")
                    cat = np.mean(latents[indices], axis=0)
                else:  
                    print("Plotting for non-trivial inputs with phase averaging")
                    #cat = np.concatenate((delay, stim, resp), axis=1)
                    
                # plot all the latents for this label
                if scatter_trajectories:
                    # can color each phase different
                    c = np.linspace(0, 1, cat.shape[0])
                    ax.scatter(*cat.T, s=6, color=cmap_time(c))
                else: 
                    norm_labels = plt.Normalize(1,39)
                    # concatenate to make a continuous time series and color by label
                    ax.plot(*cat.T, linewidth=1.5, color=cmap_rate(norm_labels(label)))
                        
        else:
            # plot all trials separately
            for i in range(latents.shape[0]):
                if scatter_trajectories:
                    ax.scatter(*latents[i].T, s=6, color=cmap_time(np.linspace(0, 1, latents.shape[1])))
                else: 
                    norm_labels = plt.Normalize(1,39)
                    ax.plot(*latents[i].T, linewidth=1.5, color=cmap_rate(norm_labels(labels[i,1])))
            
        ax.set_xlim(latents_range[0])
        ax.set_ylim(latents_range[1])
            
        # plot fixed points
        if xstar is not None and q_flag is not None and colors_fps is not None:
            ax.scatter(*xstar[q_flag].T, c=colors_fps[q_flag, :])
            
        ax.set_ylabel("$lat_2$", fontsize=20)
        ax.set_xlabel("$lat_1$", fontsize=20)
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.show()
        

    def plot_trial_io(self, num_trials):
        ics, inputs, targets = self.get_model_inputs()
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        controlled = out_dict["controlled"].detach().numpy()
        if latents.shape[-1] <= 3:
            lats_pca = latents
        else:
            pca = PCA(n_components=3)
            lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
            lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(3 * num_trials, 6))
        for i in range(num_trials):
            ax1 = fig.add_subplot(4, num_trials, i + 1)
            ax1.plot(lats_pca[i, :, 0])
            if lats_pca.shape[-1] >= 2:
                ax1.plot(lats_pca[i, :, 1])
            if lats_pca.shape[-1] == 3:
                ax1.plot(lats_pca[i, :, 2])
            ax1.set_title(f"Trial {i}")
            ax2 = fig.add_subplot(4, num_trials, i + num_trials + 1)
            for j in range(controlled.shape[-1]):
                ax2.plot(controlled[i, :, j])

            ax3 = fig.add_subplot(4, num_trials, i + 2 * num_trials + 1)
            for j in range(targets.shape[-1]):
                ax3.plot(targets[i, :, j])

            ax4 = fig.add_subplot(4, num_trials, i + 3 * num_trials + 1)
            for j in range(inputs.shape[-1]):
                ax4.plot(inputs[i, :, j])
            if i == 0:
                ax1.set_ylabel("Latent Activity")
                ax2.set_ylabel("Controlled")
                ax3.set_ylabel("Targets")
                ax4.set_ylabel("Inputs")
            if i == 4:
                ax1.set_xlabel("Time")
                ax2.set_xlabel("Time")
                ax3.set_xlabel("Time")
                ax4.set_xlabel("Time")
            else:
                ax1.set_xlabel("")
                ax2.set_xlabel("")
                ax3.set_xlabel("")
                ax4.set_xlabel("")
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])

        plt.suptitle("Task-trained Latent Activity")
        plt.show()

    def compute_FPs(
        self,
        noiseless=True,
        inputs=None,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
    ):
        # Compute latent activity from task trained model
        if inputs is None and noiseless:
            _, inputs, _ = self.get_model_inputs_noiseless()
            latents = self.get_latents_noiseless()
        elif inputs is None and not noiseless:
            _, inputs, _ = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()
        if hasattr(self.wrapper.model, "generator"):
            cell = self.wrapper.model.generator
        elif hasattr(self.wrapper.model, "cell"):
            cell = self.wrapper.model.cell
        else:
            raise ValueError("Implement cell/generator in the model")
        fps = find_fixed_points(
            model=cell,
            state_trajs=latents,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        return fps
    
    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
        n_pca_components=3,
        return_pca_model = False,
        do_pca=True,
        plot_only_points=False,
        return_points = False,
    ):

        latents = self.get_latents(phase="val").detach().numpy()
        fps = self.compute_FPs(
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        
        if not do_pca and xstar.shape[1] > 3:
            do_pca = True
            print("Using PCA for latent dimension > 3. Set do_pca = True to use PCA.")
        
        xstar = fps.xstar
        q_vals = fps.qstar  
        is_stable = fps.is_stable
        figQs = plt.figure()
        axQs = figQs.add_subplot(111)
        q_flag_temp = q_vals < 1e-15
        q_vals[q_flag_temp] = 1e-15
        axQs.hist(np.log10(q_vals), bins=100)
        axQs.set_title("Q* Histogram")
        axQs.set_xlabel("log10(Q*)")

        colors = np.zeros((xstar.shape[0], 3))
        colors[is_stable, :] = np.array([0, 0, 1])
        colors[~is_stable, 0] = 0  # black

        q_flag = q_vals < q_thresh
        if do_pca:
            pca = PCA(n_components=n_pca_components)
            xstar_pca = pca.fit_transform(xstar)
            lats_flat = latents.reshape(-1, latents.shape[-1])
            lats_pca = pca.transform(lats_flat)

            if n_pca_components == 3:
                lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                   xstar_pca[q_flag, 0],
                   xstar_pca[q_flag, 1],
                   xstar_pca[q_flag, 2],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            lats_pca[i, :, 0],
                            lats_pca[i, :, 1],
                            lats_pca[i, :, 2], linewidth=0.5,
                        )
            elif n_pca_components == 2:
                lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 2)
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(
                   xstar_pca[q_flag, 0],
                   xstar_pca[q_flag, 1],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            lats_pca[i, :, 0],
                            lats_pca[i, :, 1],
                        )
                    
        else:
            if xstar.shape[1] == 3:
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                   xstar[q_flag, 0],
                   xstar[q_flag, 1],
                   xstar[q_flag, 2],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            latents[i, :, 0],
                            latents[i, :, 1],
                            latents[i, :, 2],linewidth=0.5,
                        )
            elif xstar.shape[1] == 2:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(
                   xstar[q_flag, 0],
                   xstar[q_flag, 1],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            latents[i, :, 0],
                            latents[i, :, 1],
                        )
        
        # Add legend for stability
        ax.plot([], [], "o", color="black", label="Unstable")
        ax.plot([], [], "o", color="blue", label="Stable")
        ax.legend()
        ax.set_title("Fixed Points for Task-Trained")
        ax.set_xlabel("$m_1$")
        ax.set_ylabel("$m_2$")
        if xstar.shape[1] == 3:
            ax.set_zlabel("$m_3$")
        ax.set_facecolor('none')
        ax.grid(False)
        plt.show()
        
        if return_pca_model:
            return fps, pca
        
        if return_points:
            return fps, xstar, q_flag, colors
        
        return fps

    def simulate_neural_data(self, subfolder, dataset_path):
        self.simulator.simulate_neural_data(
            self.wrapper,
            self.datamodule,
            self.run_name,
            subfolder,
            dataset_path,
            seed=0,
        )

    def find_DSA_hps(
        self,
        rank_sweep=[10, 20],
        delay_sweep=[1, 5],
    ):
        id_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        splits_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        for i, rank in enumerate(rank_sweep):
            for j, delay in enumerate(delay_sweep):
                print(f"Rank: {rank}, Delay: {delay}")
                id_comp[i, j] = dsa_to_id(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
                splits_comp[i, j] = dsa_bw_data_splits(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(id_comp)
        ax.set_title("ID")
        ax.set_xticks(np.arange(len(delay_sweep)))
        ax.set_yticks(np.arange(len(rank_sweep)))
        ax.set_xticklabels(delay_sweep)
        ax.set_yticklabels(rank_sweep)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.savefig(f"{HOME_DIR}/id_comp.png")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(splits_comp)
        ax2.set_title("Splits")
        ax2.set_xticks(np.arange(len(delay_sweep)))
        ax2.set_yticks(np.arange(len(rank_sweep)))
        ax2.set_xticklabels(delay_sweep)
        ax2.set_yticklabels(rank_sweep)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig(f"{HOME_DIR}/splits_comp.png")
        return id_comp, splits_comp

    def save_latents(self, filepath):
        latents = self.get_latents().detach().numpy()
        with open(filepath, "wb") as f:
            pickle.dump(latents, f)

    def plot_scree(self, max_pcs=10):
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        pca = PCA(n_components=max_pcs)
        pca.fit(latents)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(range(1, max_pcs + 1), pca.explained_variance_ratio_ * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, max_pcs + 1), np.cumsum(pca.explained_variance_ratio_) * 100)
        ax2.set_xlabel("PC #")
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_ylabel("Explained Variance (%)")
        # Add horiz lines at 50, 90, 95, 99%
        ax2.axhline(y=50, color="r", linestyle="--")
        ax2.axhline(y=90, color="r", linestyle="--")
        ax2.axhline(y=95, color="r", linestyle="--")
        ax2.axhline(y=99, color="r", linestyle="--")
        # Add y ticks
        ax2.set_yticks([50, 90, 95, 99])
        plt.savefig(f"{HOME_DIR}/scree_plot.png")
        return pca.explained_variance_ratio_

    def get_trial_lens(self, phase="val"):
        if self.env.dataset_name != "MultiTask":
            raise NotImplementedError(
                f"get_trial_lens not implemented for '{self.env.dataset_name}'."
            )
        phase_dict = self.datamodule.extra_data["phase_dict"]
        train_inds = self.datamodule.train_ds.tensors[3].detach().numpy().astype(int)
        valid_inds = self.datamodule.valid_ds.tensors[3].detach().numpy().astype(int)
        len_list = []
        if phase == "val":
            inds = valid_inds
        elif phase == "train":
            inds = train_inds
        elif phase == "all":
            inds = np.vstack((train_inds, valid_inds))
        for i in inds:
            len_list.append(phase_dict[i]["response"][1])

        return len_list
    
    def get_targets(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_targets = torch.cat([train_ds.tensors[2], valid_ds.tensors[2]], dim=0)
        if phase == "all":
            return tt_targets
        elif phase == "train":
            return tt_targets[self.train_inds]
        elif phase == "val":
            return tt_targets[self.valid_inds]
        
    def get_conds(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_conds = torch.cat([train_ds.tensors[4], valid_ds.tensors[4]], dim=0)
        if phase == "all":
            return tt_conds
        elif phase == "train":
            return tt_conds[self.train_inds]
        elif phase == "val":
            return tt_conds[self.valid_inds]
    
    def get_loss(self, loss_func, phase="val", noiseless=False):
        """loss_func: loss function from the task environment (Decoupled Environment)"""
        
        if noiseless:
            output_dict = self.get_model_outputs_noiseless(phase=phase)
            inputs = self.get_true_inputs(phase=phase)
        else:
            output_dict = self.get_model_outputs(phase=phase)
            inputs = self.get_inputs(phase=phase)
            
        targets = self.get_targets(phase=phase)
        conds = self.get_conds(phase=phase)
        extras = self.get_extras(phase=phase)
            
        loss_dict = {
            "controlled": output_dict["controlled"],
            "actions": output_dict["actions"],
            "latents": output_dict["latents"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            "epoch": 1,
        }
        
        return loss_func(loss_dict).item()
