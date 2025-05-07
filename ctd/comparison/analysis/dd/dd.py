import io
import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm
from scipy.linalg import svd

from ctd.comparison.analysis.analysis import Analysis
from ctd.comparison.fixedpoints import find_fixed_points
from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class Analysis_DD(ABC, Analysis):
    @staticmethod
    def create(run_name, filepath, model_type="N/A"):
        if model_type == "SAE":
            return Analysis_DD_SAE(run_name, filepath, model_type)
        elif model_type == "LFADS":
            return Analysis_DD_LFADS(run_name, filepath, model_type)
        elif model_type == "External":
            return Analysis_DD_Ext(run_name, filepath)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def __init__(self, run_name, filepath, model_type):
        self.tt_or_dd = "dd"
        self.run_name = run_name
        self.model_type = model_type
        self.load_wrapper(filepath)

    def load_wrapper(self, filepath):
        if torch.cuda.is_available():
            with open(filepath + "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
        else:
            with open(filepath + "model.pkl", "rb") as f:
                self.model = CPU_Unpickler(f).load()
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = CPU_Unpickler(f).load()

    def to_device(self, device):
        self.model.to(device)
        self.datamodule.to(device)

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
        q_thresh=1e-7,
        early_stop_threshold=1e-8,
    ):
        # Compute latent activity from task trained model
        if inputs is None and noiseless:
            _, inputs = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()
        # latents = latents.to(device)
        # inputs = inputs.to(device)
        m_device = self.model.device
        fps = find_fixed_points(
            model=self.get_dynamics_model(),
            state_trajs=latents,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
            q_threshold=q_thresh,
            early_stop_threshold=early_stop_threshold,
        )
        self.model.to(m_device)
        return fps

    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cuda",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
    ):

        latents = self.get_model_outputs()[1].detach().cpu().numpy()
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
        xstar = fps.xstar
        q_vals = fps.qstar
        is_stable = fps.is_stable
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        zero_flag = q_vals == 0
        q_vals[zero_flag] = 1e-15
        ax.hist(np.log10(q_vals), bins=100)
        ax.set_xlabel("log10(q)")
        ax.set_ylabel("Count")
        q_flag = q_vals < q_thresh
        pca = PCA(n_components=3)
        xstar_pca = pca.fit_transform(xstar)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.transform(lats_flat)
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        # Make a color vector based on stability

        xstar_pca = xstar_pca[q_flag]
        is_stable = is_stable[q_flag]

        ax.scatter(
            xstar_pca[is_stable, 0],
            xstar_pca[is_stable, 1],
            xstar_pca[is_stable, 2],
            c="g",
        )
        ax.scatter(
            xstar_pca[~is_stable, 0],
            xstar_pca[~is_stable, 1],
            xstar_pca[~is_stable, 2],
            c="r",
        )

        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title(f"{self.model_type}_Fixed Points")
        plt.show()
        return fps

    def plot_trial(self, num_trials=10, scatterPlot=True):
        latents = self.get_latents().detach().numpy()
        pca = PCA(n_components=3)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.fit_transform(lats_flat)
        lats_pca = lats_pca.reshape(-1, latents.shape[1], 3)
        if scatterPlot:

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for i in range(num_trials):
                ax.plot(
                    lats_pca[i, :, 0],
                    lats_pca[i, :, 1],
                    lats_pca[i, :, 2],
                )
            ax.set_title(f"{self.model_type}_Trial Latent Activity")
        else:
            fig = plt.figure(figsize=(10, 4 * num_trials))
            for i in range(num_trials):
                ax = fig.add_subplot(num_trials, 1, i + 1)
                ax.plot(lats_pca[i, :, 0])
                ax.plot(lats_pca[i, :, 1])
                ax.plot(lats_pca[i, :, 2])
            ax.set_title(f"{self.model_type}_Trial Latent Activity")

        plt.show()
        
        
    def average_latents_by_phase_new(self, inputs, latents):
        """
        Align time series and take an average over each phase:
        - Delay: from start to stimulus onset
        - Stimulus: from stimulus onset to fixation offset
        - Response: from fixation offset to end

        Returns: average across trials for delay, stimulus, and response phases
                 or 0, 0, 0 if no valid trials
        """
        # trials, timesteps, _ = inputs.shape
        trials, timesteps, latent_dim = latents.shape

        stim_onsets = []
        fix_offs = []

        for i in range(trials):
            trial_input = inputs[i]
            stim_indices = np.where(trial_input[:, 1] == 1)[0]
            if len(stim_indices) == 0:
                return 0, 0, 0  # no stim found

            stim_onset = stim_indices[0]
            stim_onsets.append(stim_onset)

            # assume fixation off is when fixation input (e.g., input[:, 0]) goes to 0
            fix_off_candidates = np.where(trial_input[:, 0] == 0)[0]
            if len(fix_off_candidates) == 0 or fix_off_candidates[0] <= stim_onset:
                return 0, 0, 0  # no valid fixation offset
            fix_offs.append(fix_off_candidates[0])

        # Get max lengths for each phase to align them
        delay_len = max(stim_onsets)
        stim_lens = [fix_off - stim_onset for fix_off, stim_onset in zip(fix_offs, stim_onsets)]
        stim_len = max(stim_lens)
        resp_lens = [timesteps - fix_off for fix_off in fix_offs]
        resp_len = max(resp_lens)

        # Initialize with NaNs
        delay = np.full((trials, delay_len, latent_dim), np.nan)
        stim = np.full((trials, stim_len, latent_dim), np.nan)
        resp = np.full((trials, resp_len, latent_dim), np.nan)

        for i in range(trials):
            stim_on = stim_onsets[i]
            fix_off = fix_offs[i]

            # Fill delay phase
            delay_len_i = stim_on
            delay[i, :delay_len_i, :] = latents[i, :delay_len_i, :]

            # Fill stimulus phase
            stim_len_i = fix_off - stim_on
            stim[i, :stim_len_i, :] = latents[i, stim_on:fix_off, :]

            # Fill response phase
            resp_len_i = timesteps - fix_off
            resp[i, :resp_len_i, :] = latents[i, fix_off:, :]

        # Average across trials, ignoring NaNs
        delay_avg = np.nanmean(delay, axis=0)
        stim_avg = np.nanmean(stim, axis=0)
        resp_avg = np.nanmean(resp, axis=0)

        return delay_avg, stim_avg, resp_avg
    
    def compute_koopman(self, latents_ref, ortho=False):
        """
        Compute Koopman modes and eigenvalues from latent trajectories.

        Args:
            latents_ref: NumPy array of shape [time_steps, n_dim] or PyTorch tensor

        Returns:
            koopman_modes: Orthonormal modes (shape [n_dim, n_dim], rows are modes)
            eigenvalues: Koopman eigenvalues (shape [n_dim])
            mean_latent: Mean used for centering (shape [n_dim])
        """
        if isinstance(latents_ref, torch.Tensor):
            latents_ref = latents_ref.detach().numpy()
            
        if len(latents_ref.shape) != 2:
            latents_ref = latents_ref.reshape(-1, latents_ref.shape[-1])

        latents_ref = np.asarray(latents_ref, dtype=np.float64)
        mean_latent = np.mean(latents_ref, axis=0)

        # Center data
        X = (latents_ref[:-1] - mean_latent).T  # [n_dim, T-1]
        Y = (latents_ref[1:] - mean_latent).T   # [n_dim, T-1]

        # Dynamic Mode Decomposition
        U, S, Vh = svd(X, full_matrices=False)
        K = (Y @ Vh.T) @ np.diag(1.0/S) @ U.T  # Koopman operator
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Orthonormalize modes (rows)
        if ortho:
            eigenvectors = np.linalg.qr(eigenvectors.T)[0].T

        return eigenvectors, eigenvalues, mean_latent
    
    
    def transform_points_koopman(self, points, koopman_modes, mean_latent):
        """
        Transform points (e.g., fixed points) to Koopman space.

        Args:
            points: NumPy array [n_points, n_dim] or PyTorch tensor
            koopman_modes: From compute_koopman() [n_dim, n_dim]
            mean_latent: From compute_koopman() [n_dim]

        Returns:
            points_koopman: Transformed points [n_points, n_dim]
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().numpy()

        points = np.asarray(points, dtype=np.float64)
        # the output will be 2
        return (points - mean_latent) @ koopman_modes[:2].T
        
    def get_PCA_axes(self, inputs=None):
        
        # TODO: fix this function for DD
        
        # if common_basis:
        #     pc1, pc2, A = self.get_PCA_axes()  # each axis has flat shape (latent_dim)
        #     latents = latents.reshape(-1, latents.shape[-1]) @ A.T
        #     latents = latents.reshape(latents.shape[0], latents.shape[1], -1)
        #     # project onto pc1 and pc2
        #     lats_proj = latents @ np.array([pc1, pc2]).T  # shape (B, T, 2)
            
        #     if avg:
        #         lats_proj = np.mean(lats_proj, axis=0)
        #         lats_proj = lats_proj.reshape(1, lats_pca.shape[0], lats_pca.shape[1])

        #     ax = fig.add_subplot(111)
        #     for i in range(num_trials if not avg else 1):
        #         ax.plot(
        #             lats_proj[i, :, 0],
        #             lats_proj[i, :, 1],
        #             color=colors[i]
        #         )
        
        """Get two leading PCs from transforming with SVD of readout layer
           Use in DD models that use a nonlinear mapping to predict rates

        Args:
            inputs (np.ndarray) : explicitly generate many trials from the 
            relevant task to generate confident axes

        Returns:
            A (np.ndarray) : S V^T to transform latents
            pc1 (np.ndarray) : leading PC
            pc2 (np.ndarray) : leading PC
        """

        model = self.wrapper.model
        C = model.readout.weight.detach().numpy()
        U, S, VT = np.linalg.svd(C)
        S_diag = np.diag(S)

        if inputs is not None:
            saved_ics, saved_inputs, _ = self.get_model_inputs()
            inputs = torch.tensor(inputs, dtype=torch.float)
            if hasattr(model, "generator"):
                dynamics_model = model.generator
            elif hasattr(model, "cell"):
                dynamics_model = model.model.cell
            else:
                raise ValueError("No generator or cell found in model")
            latents = dynamics_model(inputs, saved_ics).detach().numpy()

        else:
            out_dict = self.get_model_outputs()
            latents = out_dict["latents"].detach().numpy()

        # Compute A = S @ VT with proper dimensions
        A = S_diag @ VT

        # Flatten batch and sequence dimensions
        batch_size, seq_len, latent_dim = latents.shape
        latents_reshaped = latents.reshape(-1, latent_dim) 
        # output (samples, lat_dim)
        latents_semi_orthog = (A @ latents_reshaped.T).T

        # Get the number of components for PCA
        n_components = latents_semi_orthog.shape[1]
        if n_components > latents_reshaped.shape[0]:
            raise ValueError(f"Not enough samples: found {latents_reshaped.shape[0]}, need at least {n_components}")

        # Do PCA on the semi-orthogonalized latents
        pca = PCA(n_components=n_components)
        pca.fit(latents_semi_orthog)
        # pcs = pca.components_  # (n_components, latent_dim)

        # # normalize first two PCs
        # if len(pcs) >= 2:
        #     pcs[0] = pcs[0] / np.linalg.norm(pcs[0])
        #     pcs[1] = pcs[1] / np.linalg.norm(pcs[1])

            # Return first two PCs and the transformation matrix
        return pca  # return the pca object only using lat 2 anyway

        
    def plot_flow_field(self, latents_range, num_points, cmap_field=plt.get_cmap('pink'),
                        scatter_trajectories=True, cmap_time=plt.get_cmap('copper')):
        
        # TODO: simplest version not plotting custom latents yet
        
        model = self.get_dynamics_model()
        
        _, inputs = self.get_model_inputs()
        input_field = torch.zeros_like(inputs)[0][0]
        input_field = torch.unsqueeze(input_field, 0)
        latents = self.get_latents().detach().numpy()
        
        fig, ax = plt.subplots()
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
        
        for i in range(latents.shape[0]):
            if scatter_trajectories:
                ax.scatter(*latents[i].T, s=6, color=cmap_time(np.linspace(0, 1, latents.shape[1])))
            # else: 
            #     norm_labels = plt.Normalize(1,39)
            #     ax.plot(*latents[i].T, linewidth=1.5, color=cmap_rate(norm_labels(labels[i,1])))
            
        ax.set_xlim(latents_range[0])
        ax.set_ylim(latents_range[1])
        
        plt.show()
        

    def get_inputs(self, phase="val"):
        _, inputs = self.get_model_inputs(phase=phase)

        return inputs

    def plot_rates(self, phase="val", neurons=[0], n_trials=5):
        gru_rates = self.get_rates(phase=phase)
        true_rates = self.get_true_rates(phase=phase)
        trial_lens = self.get_trial_lens(phase=phase)
        rates_stack = []
        true_rates_stack = []
        for i in range(len(trial_lens)):
            rates_stack.append(gru_rates[i][:].detach().cpu().numpy())
            true_rates_stack.append(true_rates[i][:].detach().cpu().numpy())

        fig, ax = plt.subplots(n_trials, len(neurons), figsize=(10, 10))
        for i in range(n_trials):
            for j in range(len(neurons)):
                neuron = neurons[j]
                if i == 0 and j == 0:
                    ax[i, j].plot(
                        rates_stack[i][:, neuron],
                        color="black",
                        label="Estimated Rates",
                    )
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron],
                        color="black",
                        linestyle="--",
                        label="True Rates",
                    )
                else:
                    ax[i, j].plot(rates_stack[i][:, neuron], color="black")
                    # Restart the color order
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron], color="black", linestyle="--"
                    )
                if i == 0:
                    ax[i, j].set_title(f"Neuron {neuron}")
        ax[0, 0].legend()
        plt.show()

    def plot_scree(self, max_pcs=10):
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        n_lats = latents.shape[-1]
        high_bound = np.min([n_lats, max_pcs])
        pca = PCA(n_components=high_bound)
        pca.fit(latents)
        exp_var = pca.explained_variance_ratio_
        exp_var_ext = np.zeros(max_pcs)
        exp_var_ext[:high_bound] = exp_var
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(range(1, max_pcs + 1), exp_var_ext * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, max_pcs + 1), np.cumsum(exp_var_ext) * 100)
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
        ax2.set_ylim(0, 105)
        plt.savefig(f"{self.run_name}_scree_plot.pdf")
        return exp_var_ext

    @abstractmethod
    def get_model_inputs(self):
        pass

    @abstractmethod
    def get_model_outputs(self):
        pass

    @abstractmethod
    def get_latents(self):
        pass

    @abstractmethod
    def get_dynamics_model(self):
        pass

    @abstractmethod
    def get_true_rates(self):
        pass

    @abstractmethod
    def get_rates(self):
        pass

    @abstractmethod
    def get_trial_lens(self):
        pass

    @abstractmethod
    def get_spiking(self):
        pass


class Analysis_DD_SAE(Analysis_DD):
    def get_model_inputs(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            dd_spiking = torch.cat(
                (dd_train_ds.tensors[0], dd_val_ds.tensors[0]), dim=0
            )
            dd_inputs = torch.cat((dd_train_ds.tensors[2], dd_val_ds.tensors[2]), dim=0)
        elif phase == "train":
            dd_spiking = self.datamodule.train_ds.tensors[0]
            dd_inputs = self.datamodule.train_ds.tensors[2]
        elif phase == "val":
            dd_spiking = self.datamodule.valid_ds.tensors[0]
            dd_inputs = self.datamodule.valid_ds.tensors[2]

        return dd_spiking, dd_inputs

    def get_model_outputs(self, phase="all"):
        dd_spiking, dd_inputs = self.get_model_inputs(phase=phase)
        dd_spiking = dd_spiking.to(self.model.device)
        dd_inputs = dd_inputs.to(self.model.device)
        log_rates, latents = self.model(dd_spiking, dd_inputs)
        return torch.exp(log_rates), latents

    def get_latents(self, phase="all"):
        _, latents = self.get_model_outputs(phase=phase)
        return latents

    def get_dynamics_model(self):
        return self.model.decoder.cell

    def get_true_rates(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            rates_train = dd_train_ds.tensors[6]
            rates_val = dd_val_ds.tensors[6]
            true_rates = torch.cat((rates_train, rates_val), dim=0)
        elif phase == "train":
            true_rates = self.datamodule.train_ds.tensors[6]
        elif phase == "val":
            true_rates = self.datamodule.valid_ds.tensors[6]
        return true_rates

    def get_rates(self, phase="all"):
        rates, _ = self.get_model_outputs(phase=phase)
        return rates

    def get_trial_lens(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            trial_lens = torch.cat(
                (dd_train_ds.tensors[3][:, -1], dd_val_ds.tensors[3][:, -1]), dim=0
            )
        elif phase == "train":
            trial_lens = self.datamodule.train_ds.tensors[3][:, -1]
        elif phase == "val":
            trial_lens = self.datamodule.valid_ds.tensors[3][:, -1]
        return trial_lens

    def get_spiking(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            dd_spiking = torch.cat(
                (dd_train_ds.tensors[1], dd_val_ds.tensors[1]), dim=0
            )
        elif phase == "train":
            dd_spiking = self.datamodule.train_ds.tensors[1]
        elif phase == "val":
            dd_spiking = self.datamodule.valid_ds.tensors[1]
        return dd_spiking


class Analysis_DD_LFADS(Analysis_DD):
    def get_trial_lens(self, phase="all"):
        dd_extra = []
        if phase == "all":
            train_dl = self.datamodule.train_dataloader(shuffle=False)
            val_dl = self.datamodule.val_dataloader()
            for batch in train_dl:
                # Move data to the right device
                train_extra = batch[1][3][:, -1]
                dd_extra.append(train_extra)
            for batch in val_dl:
                # Move data to the right device
                val_extra = batch[1][3][:, -1]
                dd_extra.append(val_extra)
        elif phase == "train":
            train_dl = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_dl:
                # Move data to the right device
                train_extra = batch[1][3][:, -1]
                dd_extra.append(train_extra)
        elif phase == "val":
            val_dl = self.datamodule.val_dataloader()
            for batch in val_dl:
                # Move data to the right device
                val_extra = batch[1][3][:, -1]
                dd_extra.append(val_extra)
        dd_extra = torch.cat(dd_extra, dim=0)
        return dd_extra

    def get_model_inputs(self, phase="all"):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            dd_inputs = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][0]
                inputs_train = batch[0][2]
                dd_spiking.append(spiking_train)
                dd_inputs.append(inputs_train)
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][0]
                inputs_val = batch[0][2]
                dd_spiking.append(spiking_val)
                dd_inputs.append(inputs_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_spiking = []
            dd_inputs = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][0]
                inputs_train = batch[0][2]
                dd_spiking.append(spiking_train)
                dd_inputs.append(inputs_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            dd_inputs = []
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][0]
                inputs_val = batch[0][2]
                dd_spiking.append(spiking_val)
                dd_inputs.append(inputs_val)
        dd_spiking = torch.cat(dd_spiking, dim=0)
        dd_inputs = torch.cat(dd_inputs, dim=0)
        return dd_spiking, dd_inputs

    def get_true_rates(self, phase="all"):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_rates = []
            for batch in train_ds:
                # Move data to the right device
                rates_train = batch[1][2]
                dd_rates.append(rates_train)
            for batch in val_dataloader:
                # Move data to the right device
                rates_val = batch[1][2]
                dd_rates.append(rates_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_rates = []
            for batch in train_ds:
                # Move data to the right device
                rates_train = batch[1][2]
                dd_rates.append(rates_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_rates = []
            for batch in val_dataloader:
                # Move data to the right device
                rates_val = batch[1][2]
                dd_rates.append(rates_val)
        dd_rates = torch.cat(dd_rates, dim=0)
        return dd_rates

    def get_inferred_inputs(self, phase="all"):
        dd_inf_inputs = []
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()

            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])

            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        dd_inf_inputs = torch.cat(dd_inf_inputs, dim=0)
        return dd_inf_inputs

    def get_model_outputs(self, phase="all"):
        dd_rates = []
        dd_latents = []
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])

            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        dd_rates = torch.cat(dd_rates, dim=0)
        dd_latents = torch.cat(dd_latents, dim=0)
        return dd_rates, dd_latents

    def get_rates(self, phase="all"):
        rates, _ = self.get_model_outputs(phase=phase)
        return rates

    def get_latents(self, phase="all"):
        rates, latents = self.get_model_outputs(phase=phase)
        return latents

    def get_dynamics_model(self):
        return self.model.decoder.rnn.cell.gen_cell

    def get_spiking(self, phase):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][1]
                dd_spiking.append(spiking_train)
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][1]
                dd_spiking.append(spiking_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_spiking = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][1]
                dd_spiking.append(spiking_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][1]
                dd_spiking.append(spiking_val)
        dd_spiking = torch.cat(dd_spiking, dim=0)
        return dd_spiking


class Analysis_DD_Ext(Analysis_DD):
    def __init__(self, run_name, filepath):
        self.tt_or_dd = "dd"
        self.run_name = run_name
        self.filepath = filepath

        self.train_true_rates = None
        self.train_true_latents = None
        self.eval_true_rates = None
        self.eval_true_latents = None

        self.load_data(filepath)

    def load_data(self, filepath):
        with h5py.File(filepath, "r") as h5file:
            # Check the fields
            print(h5file.keys())
            self.eval_rates = torch.Tensor(h5file["eval_rates"][()])
            self.eval_latents = torch.Tensor(h5file["eval_latents"][()])
            self.train_rates = torch.Tensor(h5file["train_rates"][()])
            self.train_latents = torch.Tensor(h5file["train_latents"][()])
            if "fixed_points" in h5file.keys():
                self.fixed_points = torch.Tensor(h5file["fixed_points"][()])
            else:
                self.fixed_points = None

    def get_latents(self, phase="all"):
        if phase == "train":
            return self.train_latents
        elif phase == "val":
            return self.eval_latents
        else:
            full_latents = torch.cat((self.train_latents, self.eval_latents), dim=0)
            return full_latents

    def get_rates(self, phase="all"):
        if phase == "train":
            return self.train_rates
        elif phase == "val":
            return self.eval_rates
        else:
            full_rates = torch.cat((self.train_rates, self.eval_rates), dim=0)
            return full_rates

    def get_true_rates(self, phase="all"):
        if phase == "train":
            return self.train_true_rates
        elif phase == "val":
            return self.eval_true_rates
        else:
            full_true_rates = torch.cat(
                (self.train_true_rates, self.eval_true_rates), dim=0
            )
            return full_true_rates

    def get_model_outputs(self, phase="all"):
        if phase == "train":
            return self.train_rates, self.train_latents
        elif phase == "val":
            return self.eval_rates, self.eval_latents
        else:
            return self.get_rates(), self.get_latents()

    def compute_FPs(self, latents, inputs):
        return None

    def add_true_rates(self, train_true_rates, eval_true_rates):
        self.train_true_rates = train_true_rates
        self.eval_true_rates = eval_true_rates

    def plot_fps(self):
        if self.fixed_points is None:
            print("No fixed points to plot")
            return
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        latents = self.get_latents(phase="val")
        fps = self.fixed_points
        for i in range(100):
            ax.plot(
                latents[i, :, 0],
                latents[i, :, 1],
                latents[i, :, 2],
                c="k",
                linewidth=0.1,
            )
        ax.scatter(fps[:, 0], fps[:, 1], fps[:, 2], c="r")
        ax.set_title(f"Fixed Points: {self.run_name}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        
        
    def plot_flow_field_new(self, latents_range: list, num_points: int, inputs_latents: np.array, input_field: np.array,  
                    input_latents_extra: np.array=None, custom_n_timesteps: int=None, n_trials=10, 
                    scatter_trajectories=False, xstar=None, is_stable=None,  plot_wrapper_trajs=False, 
                    filter_pc_rate:int=None, avg_per_rate=False, lint_plot_style=False, 
                    cmap_field=plt.get_cmap('pink'), cmap_time=plt.get_cmap('copper'), cmap_rate=plt.get_cmap('coolwarm'),
                    ics_noise=None, pca=True, koopman=False, phase='all', ortho_koopman=False, timescales=None, 
                    cluster_centers=None, cluster_is_stable=None, t_min=5.0, t_max=10.0,timescale_cmap=plt.get_cmap('cool'), 
                    pca_obj=None, field_color_only=False, **kwargs):

#         if hasattr(self.wrapper.model, "generator"):
#             model = self.wrapper.model.generator
#         elif hasattr(self.wrapper.model, "cell"):
#             model = self.wrapper.model.cell
#         else:
#             raise ValueError("No generator or cell found in model")
            
        # other model would output rates, this evolves hiddens
        model = self.get_dynamics_model()
        
        spiking, correct_inputs = self.get_model_inputs()
        # input_field = torch.zeros_like(correct_inputs)[0][0]
        # input_field = torch.unsqueeze(input_field, 0)
        latents = self.get_latents().detach().numpy()
        
        # log_rates, latents = self.model(dd_spiking, dd_inputs)
        
        # input shape should match n_dimension

        if inputs_latents.shape[-1] != correct_inputs.shape[-1]: 
            raise ValueError("inputs_latents should have last dimension: ", correct_inputs.shape[-1])
        elif input_field.shape[0] != correct_inputs.shape[-1]:
            raise ValueError("input_field should have last dimension: ", correct_inputs.shape[-1])
        else:
            inputs_latents = torch.tensor(inputs_latents, dtype=torch.float32)  #from numpy to tensor
            input_field = torch.tensor(input_field, dtype=torch.float32)  #from numpy to tensor
        # get the latents for as many trials as in inputs_latents
       # inputs_to_env = self.get_inputs_to_env(phase="all")  # TODO: is inputs_to_env, tt_ics an issue?
        # same number of trials
        n, t, _ = inputs_latents.shape
        # tt_ics = tt_ics[:n]
        # inputs_to_env = inputs_to_env[:n]
        
        if plot_wrapper_trajs:
            latents = self.get_latents().detach().numpy()
            # labels = self.get_extra_inputs().detach().numpy()
        else:
            if custom_n_timesteps is not None and custom_n_timesteps > t:
                raise ValueError("got more timesteps than in inputs_latents. Reduce custom_n_timesteps")
            # run inference with custom value
            log_rates, latents = self.model(spiking, inputs_latents)
            #out_dict = self.wrapper(tt_ics, inputs_latents, inputs_to_env, custom_n_timesteps=custom_n_timesteps)
            #latents = out_dict["latents"].detach().numpy()
            if input_latents_extra is not None:
                labels = input_latents_extra
            
        #if latents.shape[-1] > 3 and not pca:
        #    raise ValueError("Latents have more than 3 dimensions. Not supported without PCA")
        if latents.shape[-1] != len(latents_range) and not pca:
            raise ValueError("Adjust latents_range to dimension ", latents.shape[-1])
        
        input_field = torch.unsqueeze(input_field, 0)
        
        if pca:
            # get only the pca object always on the wrapper latents
            latents_ref = self.get_latents().detach().numpy()
            flattened_latents = latents_ref.reshape(-1, latents_ref.shape[-1])
            if pca_obj is None:
                pca_obj = PCA(n_components=2)
                pca_obj.fit(flattened_latents)
            
            # Calculate PCA space limits for creating the grid
            latents_pca = pca_obj.transform(flattened_latents)
            
            # Get explained variance
            explained_variance = pca_obj.explained_variance_ratio_
            print(f"PCA explained variance: PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")
            print(f"Total variance explained: {np.sum(explained_variance):.4f}")
            
            # Determine appropriate range in PCA space
            x_min, x_max = np.min(latents_pca[:, 0])-0.5, np.max(latents_pca[:, 0])+0.5
            y_min, y_max = np.min(latents_pca[:, 1])-0.5, np.max(latents_pca[:, 1])+0.5
            
            # Override latents_range with PCA ranges
            pca_range = [[x_min, x_max], [y_min, y_max]]
            
        elif koopman:
            latents_ref = self.get_latents()  # PyTorch tensor
            koopman_modes, eigenvalues, mean_latent = self.compute_koopman(latents_ref, ortho=ortho_koopman)
        
        else:
            pca_range = latents_range
        
        fig, ax = plt.subplots()
        
        if lint_plot_style and (latents.shape[-1] == 2 or pca):
            # Use PCA ranges if PCA is enabled
            ranges_to_use = pca_range if pca else latents_range
            
            x = np.linspace(ranges_to_use[0][0], ranges_to_use[0][1], num_points+1)
            y = np.linspace(ranges_to_use[1][0], ranges_to_use[1][1], num_points+1)
            x_mpts = (x[1:] + x[:-1]) / 2
            y_mpts = (y[1:] + y[:-1]) / 2
            field = np.zeros((num_points, num_points, 2))
            X, Y = np.meshgrid(x_mpts, y_mpts)
            
            for i, x_val in enumerate(x_mpts):
                for j, y_val in enumerate(y_mpts):
                    if pca:
                        # Transform PCA point back to original space
                        state_pca = np.array([[x_val, y_val]])
                        state_orig = pca_obj.inverse_transform(state_pca)
                        state = torch.tensor(state_orig, dtype=torch.float32)
                    else:
                        state = torch.tensor([[x_val, y_val]], dtype=torch.float32)
                    
                    # NOTE: keep the indexing like ji
                    velocity = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy()
                    
                    if pca:
                        # Transform velocity to PCA space
                        velocity = (pca_obj.components_ @ velocity.reshape(-1, 1)).flatten()
                        
                    elif koopman: 
                        velocity_centered = velocity - (state.numpy().squeeze() - mean_latent)
                        velocity = (koopman_modes[:2] @ velocity_centered.reshape(-1, 1)).flatten()
                        # origin = np.zeros((2, 2))
                        # ax.quiver(*origin, koopman_modes[:, 0], koopman_modes[:, 1], color=['r', 'b'], scale=10)  
                    
                    field[j, i, :] = velocity
                    
            if not field_color_only:
                ax.streamplot(x_mpts, y_mpts, field[:, :, 0], field[:, :, 1], color='white', density=1., arrowsize=1.,
                              linewidth=1.*.5)
#             else:
            norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)   
            # divide magnitude of norm field by its max
            norm_field = norm_field / np.max(norm_field)
            
            mappable = ax.pcolor(X, Y, norm_field, cmap=cmap_field)
            # plot a colormap for the normalized field
            # fig.colorbar(mappable, ax=ax)
            
        else:
            num_points = int(num_points / 3)
            # Calculate velocities over a grid using a double for loop implementation
            
            # Use PCA ranges if PCA is enabled
            ranges_to_use = pca_range if pca else latents_range
            
            x = np.linspace(ranges_to_use[0][0], ranges_to_use[0][1], num_points)
            y = np.linspace(ranges_to_use[1][0], ranges_to_use[1][1], num_points)
            
            if len(latents_range) == 3 and not pca:
                z = np.linspace(latents_range[2][0], latents_range[2][1], num_points)
                
            if len(latents_range) == 2 or pca:
                U = np.zeros([num_points, num_points])
                V = np.zeros([num_points, num_points])
            else:
                U = np.zeros([num_points, num_points, num_points])
                V = np.zeros([num_points, num_points, num_points])
                W = np.zeros([num_points, num_points, num_points])
                
            for i in range(num_points):
                for j in range(num_points):
                    if pca:
                        # Transform PCA point back to original space
                        state_pca = np.array([[x[i], y[j]]])
                        state_orig = pca_obj.inverse_transform(state_pca)
                        state = torch.tensor(state_orig, dtype=torch.float32)
                        
                        # Get velocity in original space
                        velocity = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy()
                        
                        # Transform to PCA space
                        velocity_pca = pca_obj.transform(velocity.reshape(1, -1)).flatten()
                        U[i, j], V[i, j] = velocity_pca
                    elif len(latents_range) == 2:
                        state = torch.tensor([[x[i], y[j]]], dtype=torch.float)
                        # NOTE: this is only applicable to a NODE w/ leak term
                        U[i, j], V[i, j] = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy().flatten()
                    else:
                        for k in range(num_points):
                            state = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float)
                            U[i, j, k], V[i, j, k], W[i, j, k] = (model(input_field, state).squeeze() - state.squeeze()).detach().numpy().flatten()
                            
            # Create a colormap based on the normalized magnitude
            if len(latents_range) == 2 or pca:
                magnitude = np.sqrt(U**2 + V**2)
            else:
                magnitude = np.sqrt(U**2 + V**2 + W**2)
                
            normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            colors_map = cmap_field(normalized_magnitude.flatten())
            
            # Plot the velocity field
            if len(latents_range) == 2 or pca:
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
                if plot_wrapper_trajs:
                    inputs_latents = correct_inputs
                delay, stim, resp = self.average_latents_by_phase_new(inputs_latents[indices], latents[indices])

                # if all inputs were zero (no stim - don't need phases)
                if type(delay) == int:  # if all returned zero
                    print("Plotting for all zero inputs - no phase averaging")
                    cat = np.mean(latents[indices], axis=0)
                else:  
                    print("Plotting for non-trivial inputs with phase averaging")
                    if phase == "all":
                        cat = np.concatenate((delay, stim, resp), axis=0)
                    elif phase == "delay":
                        cat = delay
                    elif phase == "stim":
                        cat = stim
                    elif phase == "resp":
                        cat = resp
                    
                if pca:
                    cat = pca_obj.transform(cat)
                    
                elif koopman:
                    # already in shape (n_(time)points, n_dim)
                    cat = self.transform_points_koopman(cat, koopman_modes, mean_latent)
                    
                # plot all the latents for this label
                if scatter_trajectories:
                    # can color each phase different
                    c = np.linspace(0, 1, cat.shape[0])
                    ax.scatter(*cat.T, s=10, color=cmap_time(c))
                else: 
                    norm_labels = plt.Normalize(1,39)
                    # concatenate to make a continuous time series and color by label
                    ax.plot(*cat.T, linewidth=1.5, color=cmap_rate(norm_labels(label)))
                        
        else:
            # plot all trials separately
            for i in range(min(n_trials, latents.shape[0])):
                latents_to_plot = latents[i]
                if pca:
                    latents_to_plot = pca_obj.transform(latents_to_plot)
                elif koopman:
                    b,t,d = latents_to_plot.shape
                    latents_to_plot = latents_to_plot.reshape(-1, latents.shape[-1])
                    cat = self.transform_points_koopman(latents_to_plot, koopman_modes, mean_latent)
                    #reshape back
                    # TODO: does koopman ever reduce dimensions? if so fix
                    cat = cat.reshape(b,t,d)
                    
                if scatter_trajectories:
                    ax.scatter(*latents_to_plot.T, s=6, color=cmap_time(np.linspace(0, 1, latents_to_plot.shape[0])))
                else: 
                    norm_labels = plt.Normalize(1,39)
                    ax.plot(*latents_to_plot.T, linewidth=1.5, color=cmap_rate(norm_labels(labels[i,1])))
        
        # Set the plot limits
        if pca:
            ax.set_xlim(pca_range[0])
            ax.set_ylim(pca_range[1])
        else:
            ax.set_xlim(latents_range[0])
            ax.set_ylim(latents_range[1])
            
        # plot fixed points  
        if xstar is not None and is_stable is not None and not np.all(np.isnan(xstar)):
                print(f"Plotting {xstar.shape[0]} fixed points")
                
                if pca:
                    xstar_pca = pca_obj.transform(xstar)
                    #z_fixed = pca_obj.transform(x_fixed.reshape(1, -1))[0]
                    stable_points = xstar_pca[is_stable]
                    unstable_points = xstar_pca[~is_stable]
                    
                elif koopman:
                    xstar_koopman = self.transform_points_koopman(xstar, koopman_modes, mean_latent)
                    stable_points = xstar_koopman[is_stable]
                    unstable_points = xstar_koopman[~is_stable]
                    
                else:
                    stable_points = xstar[is_stable]
                    unstable_points = xstar[~is_stable]
                
                # color by timescale (only decaying modes)
                if len(timescales) > 0:
                    if pca:
                        centers = pca_obj.transform(cluster_centers)
                    elif koopman:
                        centers = self.transform_points_koopman(cluster_centers, koopman_modes, mean_latent)
                    else:
                        centers = cluster_centers
                        
                    stable_points = centers[cluster_is_stable]
                    unstable_points = centers[~cluster_is_stable]
                    timescales_stable = timescales[cluster_is_stable]
                    timescales_unstable = timescales[~cluster_is_stable]

                    # Color by timescale (both decaying and growing modes)
                    is_special = timescales_unstable == -1
                    is_valid_stable = ~np.isnan(timescales_stable)
                    is_valid_unstable = ~np.isnan(timescales_unstable) & ~is_special

                    # Combine all valid timescales for normalization
                    all_valid_timescales = []
                    if np.any(is_valid_stable):
                        all_valid_timescales.extend(timescales_stable[is_valid_stable])
                    if np.any(is_valid_unstable):
                        all_valid_timescales.extend(timescales_unstable[is_valid_unstable])

                    if len(all_valid_timescales) > 0:
                        # Create a symmetric normalization around 0

                        # Use LogNorm for logarithmic normalization
                        norm = LogNorm(vmin=t_min, vmax=t_max)  # Replace 1e-3 with your desired minimum value

                        # Plot valid stable points with continuous colormap
                        if np.any(is_valid_stable):
                            sc_stable = ax.scatter(
                                *stable_points[is_valid_stable].T, 
                                c=timescales_stable[is_valid_stable], 
                                cmap=timescale_cmap, 
                                marker='o', 
                                s=60, 
                                norm=norm, 
                                edgecolors='white', # edge color
                                linewidths=1.5
                            )

                        # Plot valid unstable points with continuous colormap
                        if np.any(is_valid_unstable):
                            sc_unstable = ax.scatter(
                                *unstable_points[is_valid_unstable].T, 
                                c=-1.0*timescales_unstable[is_valid_unstable], 
                                cmap=timescale_cmap, 
                                marker='s', 
                                s=60, 
                                norm=norm, 
                                edgecolors='white', # edge color
                                linewidths=1.5
                            )

                        # Plot special points in black
                        if np.any(is_special):
                            ax.scatter(
                                *unstable_points[is_special].T, 
                                c='black', 
                                marker='o', 
                                s=60, 
                                edgecolor=timescale_cmap,
                                linewidth=0.5,
                                edgecolors='white', # edge color
                                linewidths=2
                            )
                        # add the colorbar
                        if np.any(is_valid_stable) or np.any(is_valid_unstable):
                            sc = sc_stable if np.any(is_valid_stable) else sc_unstable
                            plt.colorbar(sc, ax=ax, label=r'Mean $\tau$ (s)')

                else:
                    # color by stability
                    ax.scatter(*stable_points.T, c='limegreen', marker='o', s=30)
                    ax.scatter(*unstable_points.T, c='red', marker='x', s=30)
        
        # Set labels based on PCA or original space
        if pca:
            ax.set_xlabel(f"$PC_1$ ({explained_variance[0]:.1%} var)", fontsize=20)
            ax.set_ylabel(f"$PC_2$ ({explained_variance[1]:.1%} var)", fontsize=20)
        elif koopman:
            ax.set_xlabel("$K_1$", fontsize=20)
            ax.set_ylabel("$K_2$", fontsize=20)
        else:        
            ax.set_ylabel("$D_2$", fontsize=20)
            ax.set_xlabel("$D_1$", fontsize=20)
            
        # make the ticks big
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        # Return PCA object and explained variance if requested
        if pca and 'return_pca' in kwargs and kwargs['return_pca']:
            return fig, ax, pca_obj, explained_variance
        elif koopman and 'return_koopman' in kwargs and kwargs['return_koopman']:
            return fig, ax, koopman_modes, eigenvalues, mean_latent
            
        return fig, ax
