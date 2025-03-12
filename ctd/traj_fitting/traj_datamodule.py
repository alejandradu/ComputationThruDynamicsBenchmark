import logging
import os
import pickle

import dotenv
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from gymnasium import Env
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class lintDataModule(pl.LightningDataModule):
    """Create training and validation datasets for trajectory fitting"""
    
    def __init__(self, batch_size=64, num_workers=1):