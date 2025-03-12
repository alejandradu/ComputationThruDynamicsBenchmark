"""Implement the LINT method for fitting a model to the trajectories 
produced by a task-optimized model"""
 
import sys
import os
import h5py
import numpy as np
import torch
from sklearn.metrics import r2_score

# import the torch models
from ctd.task_modeling.model.rnn import LowRankRNN, FullRankRNN
from train_loop import train

# GLOBAL VARS TO TUNE WITH THE SLURM SCRIPT
# NOTE: later optimize this with ray
EPOCHS = 200
BATCH_SIZE = 64

def load_trajs(path, seed=0):
    """
    Load trajectories and inputs for traj fitting
    
    path: path to hdf5 file saved from task-training
    NOTE: is it saved in the same way from data-training exps?
    """
    
    with h5py.File(path, 'r') as h5_file:
        train_lat = h5_file["train_latents"][:]
        val_lat = h5_file["valid_latents"][:]
        train_inputs = h5_file["train_inputs"][:]
        val_inputs = h5_file["valid_inputs"][:]
        # output = h5_file["reaodut"][:]

    return train_lat, train_inputs, val_lat, val_inputs
    
    
def traj_fit(path, n_ranks=1):
    """Fit low rank RNNs to the trajectories output by a 
    task-optimized model (either LR or FR, or node??)
    
    path: path to saved model of a task-optimized nn to fit
    NOTE: not putting any mask now - counting all timesteps """
    
    # load the latents and ignore the very first timestep
    latents, inputs, val_latents, val_inputs, output = load_trajs(path)
    N_TRIALS, N_TIMESTEPS, INPUT_SIZE = inputs.shape
    LATENT_SIZE = latents.shape[2]
    
    # pass through non-linearity (rectify?)
    target = torch.tanh(latents[:, 1:])
    # BUG: might throw bc val_latents is array and not torch tensor
    val_output = torch.tanh(val_latents[:, 1:])
    print("val output shape", target.shape)
    
    # they train on the trajs from the inputs, but then check 
    # against the val trajs??
    for rank in range(1, n_ranks+1):         
        net = LowRankRNN(LATENT_SIZE,rank,input_size=LATENT_SIZE,output_size=LATENT_SIZE,noise_level=0.05,gamma=0.2)
        # TODO: adjust the params for new train func - cuda??
        train(net, inputs, target, mask=None, n_epochs, lr=1e-2, clip_gradient=1, keep_best=True, cuda=2)
        # save the model
        net.to('cpu')
        torch.save(net.state_dict(), f'../models/lowRankRNN_fitted_r{rank}.pt')
        # get the forward step of the model
        # TODO: get the initial conditions!!
        fit_outdict = net.forward(ics, val_inputs)
        fit_latents = fit_outdict["latents"]
        fit_output = torch.tanh(fit_latents[:, 1:])
        print("val output shape", target.shape)
        
        # TODO: integrate checking for task accuracy here?
        
        # get the r2 score
    print("finished fitting all ranks")
        
        # how can we compute task performance now?
    
    
    # define low or full rank RNN model 
    
    # prob define a training function??
    
    # 1 define another train function
    
