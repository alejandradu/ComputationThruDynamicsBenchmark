"""This defines the simplified training loop for the LINT method
   based on the training loop in low_rank_rnns.modules in LINT
"""

import torch
from low_rank_rnns.helpers import *
import torch.nn as nn
from math import sqrt, floor
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def loss_mse(output, target, mask=None):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    if mask is None:
        mask = torch.ones_like(target)
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, early_stop=None, keep_best=False, cuda=False, resample=False,
          initial_states=None):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param early_stop: None or float, set to target loss value after which to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors if desired
    :return: nothing
    # NOTE: here example = trial
    """
    print("Training...")
    # what are we doing?? Outline the steps
    # state the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            if cuda == True:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda}')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device, dtype=torch.float32)   # TODO do we need _input
    target = _target.to(device=device, dtype=torch.float32)
    mask = _mask.to(device=device, dtype=torch.float32)
    if initial_states is not None:
        initial_states = initial_states.to(device=device, dtype=torch.float32)

    # Initialize setup to keep best network
    with torch.no_grad():
        output = net(input, initial_states=initial_states)
        initial_loss = loss_mse(output, target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        losses = []  # losses over the whole epoch
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            if initial_states is not None:
                output = net(batch, initial_states=initial_states[random_batch_idx])
            else:
                output = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()
            if mask_gradients:
                net.m.grad = net.m.grad * net.m_mask
                net.n.grad = net.n.grad * net.n_mask
                net.wi.grad = net.wi.grad * net.wi_mask
                net.wo.grad = net.wo.grad * net.wo_mask
                net.unitn.grad = net.unitn.grad * net.unitn_mask
                net.unitm.grad = net.unitm.grad * net.unitm_mask
                net.unitwi.grad = net.unitwi.grad * net.unitwi_mask
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())