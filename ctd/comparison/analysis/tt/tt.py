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
    
    def plot_trial_latents(self, num_trials=10, pca=True, tsne=False, 
                           reduce_3_latents=False, n_components = 3):
        """
        Plot latent trajectories for trials ran during training, with
        predetermined train/val inputs
        """
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        fig = plt.figure(figsize=(10, 10))
        
        # Use a colormap to plot the trials
        colors = cm.viridis(np.linspace(0, 1, num_trials))
  
        # reduce
        if latents.shape[-1] > 3 and pca:
            pca = PCA(n_components=3)
            lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
            lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        if latents.shape[-1] > 3 and tsne:
            tsne = TSNE(n_components=3)
            lats_tsne = tsne.fit_transform(latents.reshape(-1, latents.shape[-1]))
            lats_tsne = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
            
        # special case with 3 dimensional latents but want 2D plot
        if (latents.shape[-1] == 3 and reduce_3_latents) or (latents.shape[-1] >= 3 and n_components == 2):
            pca = PCA(n_components=2)
            lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
            lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 2)
            # 2 axis with raw latents
            ax= fig.add_subplot(111)
            ax_list = [ax]
            for i in range(num_trials):
                ax.plot(
                    lats_pca[i, :, 0],
                    lats_pca[i, :, 1],
                    color=colors[i]
                )
            plt.show()
            return

        # plot
        if latents.shape[-1] == 2:
            # 2 axis with raw latents
            ax= fig.add_subplot(111)
            ax_list = [ax]
            for i in range(num_trials):
                ax.plot(
                    latents[i, :, 0],
                    latents[i, :, 1],
                    color=colors[i]
                )
            
        elif latents.shape[-1] == 3:
            # 3 axis with raw latents
            ax = fig.add_subplot(111, projection="3d")
            ax_list = [ax]
            for i in range(num_trials):
                ax.plot(
                    latents[i, :, 0],
                    latents[i, :, 1],
                    latents[i, :, 2],
                    color=colors[i]
                )
        else:
            # each axis will be a reduced output
            if (pca and not tsne) or (tsne and not pca):
                ax = fig.add_subplot(111, projection="3d")
                ax_list = [ax]
                for i in range(num_trials):
                    if pca:
                        ax.plot(
                            lats_pca[i, :, 0],
                            lats_pca[i, :, 1],
                            lats_pca[i, :, 2],
                            color=colors[i]
                        )
                    else:
                        ax.plot(
                            lats_tsne[i, :, 0],
                            lats_tsne[i, :, 1],
                            lats_tsne[i, :, 2],
                            color=colors[i]
                        )
            elif pca and tsne:
                ax1 = fig.add_subplot(121, projection="3d")
                ax2 = fig.add_subplot(122, projection="3d")
                ax_list = [ax1, ax2]
                for i in range(num_trials):
                    ax1.plot(
                        lats_pca[i, :, 0],
                        lats_pca[i, :, 1],
                        lats_pca[i, :, 2],
                        color=colors[i]
                    )
                    ax2.plot(
                        lats_tsne[i, :, 0],
                        lats_tsne[i, :, 1],
                        lats_tsne[i, :, 2],
                        color=colors[i]
                    )

        # Set grid color to white
        for a in ax_list:
            a.tick_params(axis='both', which='major', labelsize=16)
        
        # TODO: adjust axis labels

        plt.show()
        
    def plot_flow_fieldLR(self, latents_range: list, num_points:int, inputs: torch.Tensor, vec1, vec2, 
                          orth=False, sizes=1.):
        """
        Plot 2d flow field and eventually fixed points for a rank 2 network. Can plot the affine flow field in presence of a
        constant input with argument input.
        
        :param vec1: None or a numpy array of shape (hidden_size). If None, will be taken as vector m1 of the network
        :param vec2: same with m2
        :param input: None or torch tensor of shape (n_inputs), provides constant input for plotting affine flow field
        :param orth: bool, if True, start by orthogonalizing (vec1, vec2)
        :param sizes: float, general scaling factor for arrows
        
        """
        pass
    
    # def plot_field(net, vec1=None, vec2=None, xmin=-3, xmax=3, ymin=-3, ymax=3, input=None, res=50,
    #            ax=None, add_fixed_points=False, fixed_points_trials=10, fp_save=None, fp_load=None, nojac=False,
    #            orth=False, sizes=1.):
    # """
    # Plot 2d flow field and eventually fixed points for a rank 2 network. Can plot the affine flow field in presence of a
    # constant input with argument input.
    # :param net: a LowRankRNN
    # :param vec1: None or a numpy array of shape (hidden_size). If None, will be taken as vector m1 of the network
    # :param vec2: same with m2
    # :param xmin: float
    # :param xmax: float
    # :param ymin: float
    # :param ymax: float
    # :param input: None or torch tensor of shape (n_inputs), provides constant input for plotting affine flow field
    # :param res: int, grid resolution
    # :param ax: None or matplotlib axes
    # :param add_fixed_points: bool
    # :param fixed_points_trials: int, number of simulations to launch to find fixed points
    # :param fp_save: None or filename, to save found fixed points instead of plotting them
    # :param fp_load: None or filename, to load fixed points instead of recomputing them
    # :param nojac: bool, if True, use root solver without jacobian matrix
    # :param orth: bool, if True, start by orthogonalizing (vec1, vec2)
    # :param sizes: float, general scaling factor for arrows
    # :return: axes, mappable (for colorbar)
    # """
    # if ax is None:
    #     fig, ax = plt.subplots()
    # adjust_plot(ax, xmin, xmax, ymin, ymax)
    # if vec1 is None:
    #     vec1 = net.m[:, 0].squeeze().detach().numpy()
    # if vec2 is None:
    #     vec2 = net.m[:, 1].squeeze().detach().numpy()
    # if add_fixed_points:
    #     n1 = net.n[:, 0].squeeze().detach().numpy()
    #     n2 = net.n[:, 1].squeeze().detach().numpy()
    # m = net.m.detach().numpy()
    # n = net.n.detach().numpy()

    # # Plotting constants
    # nx, ny = res, res
    # marker_size = 50 * sizes

    # # Orthogonalization of the basis vec1, vec2, I
    # if orth:
    #     vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
    # if input is not None:
    #     I = (input @ net.wi_full).detach().numpy()
    #     I_orth = I - (I @ vec1) * vec1 / (vec1 @ vec1) - (I @ vec2) * vec2 / (vec2 @ vec2)
    # else:
    #     I = np.zeros(net.hidden_size)
    #     I_orth = np.zeros(net.hidden_size)

    # # rescaling factors (for transformation euclidean space / overlap space)
    # # here, if one wants x s.t. overlap(x, vec1) = alpha, x should be r1 * alpha * vec1
    # # with the overlap being defined as overlap(u, v) = u.dot(v) / sqrt(hidden_size)
    # r1 = net.hidden_size / (vec1 @ vec1)
    # r2 = net.hidden_size / (vec2 @ vec2)

    # # Defining the grid
    # xs_grid = np.linspace(xmin, xmax, nx + 1)
    # ys_grid = np.linspace(ymin, ymax, ny + 1)
    # xs = (xs_grid[1:] + xs_grid[:-1]) / 2
    # ys = (ys_grid[1:] + ys_grid[:-1]) / 2
    # field = np.zeros((nx, ny, 2))
    # X, Y = np.meshgrid(xs, ys)

    # # Recurrent function of dx/dt = F(x, I)
    # def F(x, I):
    #     return -x + m @ (n.T @ np.tanh(x)) / net.hidden_size + I

    # # Compute flow in each point of the grid
    # for i, x in enumerate(xs):
    #     for j, y in enumerate(ys):
    #         h = r1 * x * vec1 + r2 * y * vec2 + I_orth
    #         delta = F(h, I)
    #         field[j, i, 0] = delta @ vec1
    #         field[j, i, 1] = delta @ vec2
    # ax.streamplot(xs, ys, field[:, :, 0], field[:, :, 1], color='white', density=0.5, arrowsize=sizes,
    #               linewidth=sizes*.8)
    # norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)
    # mappable = ax.pcolor(X, Y, norm_field)

    # # Look for fixed points
    # if add_fixed_points:
    #     if fp_load is None:
    #         stable_sols = []
    #         saddles = []
    #         sources = []

    #         # initial conditions are dispersed over a grid
    #         X_grid, Y_grid = np.meshgrid(np.linspace(xmin, xmax, int(sqrt(fixed_points_trials))),
    #                                      np.linspace(ymin, ymax, int(sqrt(fixed_points_trials))))

    #         # Parallelized root solver
    #         x0s = [r1 * X_grid.ravel()[i] * vec1 + r2 * Y_grid.ravel()[i] * vec2 + I_orth for i in range(X_grid.size)]
    #         with mp.Pool(mp.cpu_count()) as pool:
    #             args = [(x0, m, n, net.hidden_size, I, nojac) for x0 in x0s]
    #             sols = pool.starmap(fixedpoint_task, args)

    #         for sol in sols:
    #             # if solution found
    #             if sol.success == 1:
    #                 kappa_sol = [(sol.x @ vec1) / net.hidden_size, (sol.x @ vec2) / net.hidden_size]
    #                 # Computing stability
    #                 pseudoJac = np.zeros((2, 2))
    #                 phiPr = phi_prime(sol.x)
    #                 n1_eff = n1 * phiPr
    #                 n2_eff = n2 * phiPr
    #                 pseudoJac[0, 0] = vec1 @ n1_eff / net.hidden_size
    #                 pseudoJac[0, 1] = vec2 @ n1_eff / net.hidden_size
    #                 pseudoJac[1, 0] = vec1 @ n2_eff / net.hidden_size
    #                 pseudoJac[1, 1] = vec2 @ n2_eff / net.hidden_size
    #                 eigvals = np.linalg.eigvals(pseudoJac)
    #                 if np.all(np.real(eigvals) <= 1):
    #                     stable_sols.append(kappa_sol)
    #                 elif np.any(np.real(eigvals) <= 1):
    #                     saddles.append(kappa_sol)
    #                 else:
    #                     sources.append(kappa_sol)
    #     # Load fixed points stored in a file
    #     else:
    #         arrays = np.load(fp_load)
    #         arr = arrays['arr_0']
    #         stable_sols = [arr[i] for i in range(arr.shape[0])]
    #         arr = arrays['arr_1']
    #         saddles = [arr[i] for i in range(arr.shape[0])]
    #         arr = arrays['arr_2']
    #         sources = [arr[i] for i in range(arr.shape[0])]
    #     if fp_save is not None:
    #         np.savez(fp_save, np.array(stable_sols), np.array(saddles), np.array(sources))
    #     else:
    #         ax.scatter([x[0] for x in stable_sols], [x[1] for x in stable_sols], facecolors='white', edgecolors='white',
    #                    s=marker_size, zorder=1000)
    #         ax.scatter([x[0] for x in saddles], [x[1] for x in saddles], facecolors='black', edgecolors='white',
    #                    s=marker_size, zorder=1000)
    #         ax.scatter([x[0] for x in sources], [x[1] for x in sources], facecolors='black', edgecolors='white',
    #                    s=marker_size, zorder=1000)
    # return ax, mappable


# def plot_trajectories(net, inputs, vec1=None, vec2=None, ax=None, labels=None, **plot_kws):
#     # Getting m1 and m2, orthogonalize basis
#     if vec1 is None:
#         vec1 = net.m[:, 0].squeeze().detach().numpy()
#     if vec2 is None:
#         vec2 = net.m[:, 1].squeeze().detach().numpy()
#     vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)

#     out, traj = net.forward(inputs, return_dynamics=True)
#     traj = traj.detach().numpy()

#     traj1 = traj @ vec1 / net.hidden_size
#     traj1 = traj1.squeeze()
#     traj2 = traj @ vec2 / net.hidden_size
#     traj2 = traj2.squeeze()

#     if ax is None:
#         fig, ax = plt.subplots()
#     xmin = np.min(traj1)
#     xmax = np.max(traj1)
#     ymin = np.min(traj2)
#     ymax = np.max(traj2)
#     adjust_plot(ax, xmin, xmax, ymin, ymax)

#     n_trials = inputs.shape[0]
#     for i in range(n_trials):
#         if labels is not None:
#             ax.plot(traj1[i], traj2[i], label=labels[i], **plot_kws)
#         else:
#             ax.plot(traj1[i], traj2[i], **plot_kws)

#     if labels is not None:
#         fig.legend(loc='center right', borderaxespad=0.1)
#         plt.subplots_adjust(right=.6)

#     return ax
        
        
    def plot_flow_field(self, latents_range: list, num_points: int, inputs_latents: np.array, input_field: np.array,  
                        custom_task_env: DecoupledEnvironment=None, n_trials=10, scatter_trajectories=False,
                        xstar=None, q_flag=None, colors_fps=None,  cmap=plt.cm.pink, plot_saved_trajs=False):
        """
        Plot the velocity flow field for a previously trained model. 

        Args:
            latents_range (list): range of each axis on the grid
            num_points (int): to set the grid
            inputs_latents (np.array):(n_trials, n_timesteps, input_dim) array to draw trajectories 
            input_field (np.array): flat array (input_dim) - fixed inputs to get the velocities
            custom_task_env (DecoupledEnvironment, optional): Custom task environment with desired parameters.
            n_trials (int, optional): Number of trials to plot. Defaults to 10.
            scatter_trajectories (bool, optional): True to plot the trajectories with a colormap indicating time evolution. Defaults to False.
            xstar (None, optional): Fixed points to plot. Defaults to None.
            q_flag (None, optional): Flag to indicate which fixed points to plot. Defaults to None.
            colors_fps (None, optional): Colors for the fixed points. Defaults to None.
            cmap (colormap, optional): Colormap for the flow field plot. Defaults to plt.cm.pink.
            plot_saved_trajs (bool, optional): True to plot the trajectories saved during training. Defaults to False.
        """
        
        if hasattr(self.wrapper.model, "generator"):
            model = self.wrapper.model.generator
        elif hasattr(self.wrapper.model, "cell"):
            model = self.wrapper.model.cell
        else:
            raise ValueError("No generator or cell found in model")
        
        # load the task_env to modify it to get latents, if necessary
        if custom_task_env is not None:
            self.wrapper.task_env = custom_task_env
        
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
        tt_ics = 5.0*torch.ones_like(tt_ics[:n])
        inputs_to_env = inputs_to_env[:n]
        
        if plot_saved_trajs:
            latents = self.get_latents().detach().numpy()
        else:
            # run the model in the wrapper with the custom_env
            out_dict = self.wrapper(tt_ics, inputs_latents, inputs_to_env)
            latents = out_dict["latents"].detach().numpy()
            
        if latents.shape[-1] > 3:
            raise ValueError("Latents have more than 3 dimensions. Not supported now")
        elif latents.shape[-1] != len(latents_range):
            raise ValueError("Adjust latents_range to dimension ", latents.shape[-1])
            
        fig, ax = plt.subplots()
            
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
            
        input_field = torch.unsqueeze(input_field, 0)
            
        for i in range(num_points):
            for j in range(num_points):
                state = torch.tensor([[x[i], y[j]]], dtype=torch.float)
                if len(latents_range) == 2:
                    # NOTE: need to multiply by the time constant
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
        colors_map = cmap(normalized_magnitude.flatten())

        # Plot the velocity field
        if len(latents_range) == 2:
            ax.quiver(*np.meshgrid(x, y, indexing='ij'), U, V, color=colors_map)
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(*np.meshgrid(x, y, z, indexing='ij'), U, V, W, color=colors_map)
        
        colors_time = plt.cm.copper(np.linspace(0, 1, latents.shape[1]))
            
        if n_trials > latents.shape[0]:
            n_trials = latents.shape[0]
        for i in range(n_trials):
            if scatter_trajectories:
                ax.scatter(*latents[i].T, s=7, color=colors_time)
            else: 
                ax.plot(*latents[i].T, linewidth=0.25, color='black')
            ax.set_xlim(latents_range[0])
            ax.set_ylim(latents_range[1])
            
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
        
    def get_extras(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_extras = torch.cat([train_ds.tensors[5], valid_ds.tensors[5]], dim=0)
        if phase == "all":
            return tt_extras
        elif phase == "train":
            return tt_extras[self.train_inds]
        elif phase == "val":
            return tt_extras[self.valid_inds]
    
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
