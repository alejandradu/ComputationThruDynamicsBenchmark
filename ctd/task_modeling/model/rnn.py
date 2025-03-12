import torch
from torch import nn
from torch.nn import GRUCell, RNNCell
import numpy as np

"""
All models must meet a few requirements
    1. They must have an init_model method that takes
    input_size and output_size as arguments
    2. They must have a forward method that takes inputs and hidden
    as arguments and returns output and hidden for one time step
    3. They must have a cell attribute that is the recurrent cell
    4. They must have a readout attribute that is the output layer
    (mapping from latent to output)

    Optionally,
    1. They can have an init_hidden method that takes
    batch_size as an argument and returns an initial hidden state
    2. They can have a model_loss method that takes a loss_dict
    as an argument and returns a loss (L2 regularization on latents, etc.)

"""


class GRU_RNN(nn.Module):
    def __init__(
        self, latent_size, input_size=None, output_size=None, latent_ic_var=0.05
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class NoisyGRU(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_RNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_LatentL2(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
        l2_wt=1e-2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.l2_wt = l2_wt
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden

    def model_loss(self, loss_dict):
        latents = loss_dict["latents"]
        lats_flat = latents.view(latents.shape[0], -1)
        latent_l2_loss = self.l2_wt * torch.norm(lats_flat, p=2, dim=1).mean()
        return latent_l2_loss


class DriscollRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        # BUG
        # hidden = (1 - self.gamma) * self.recW(hidden) + self.gamma * self.act_func(
        #     self.recW(hidden) + self.inpW(inputs) + self.bias + noise
        # )
        hidden = (1 - self.gamma) * hidden + self.gamma * self.act_func(
            self.recW(hidden) + self.inpW(inputs) + self.bias + noise
        )
        return output, hidden
    
    
# TODO: change name, since now it works for FR and LR
class FullRankRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,    # delta(t) / tau
        rank=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()
        self.rank=rank
        self.cell = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)
        self.cell = RNNGeneralCell(self.input_size, self.latent_size, gamma=self.gamma, 
                                   noise=self.noise_level, rank=self.rank)

    def forward(self, inputs, hidden):
        # NOTE: leaving the addition of noise inside the cell but it seems to be outside for nODEs?
        output = self.readout(hidden)
        # hidden = (1 - self.gamma) * hidden + self.gamma * (self.recW(self.act_func(hidden)) +
        #             self.inpW(inputs) + self.bias + noise)
        hidden = self.cell(inputs, hidden)
        return output, hidden
    
    
# Cell that exclusively evolves the hidden state
class RNNGeneralCell(nn.Module):
    def __init__(self, input_size, latent_size, gamma=0.1, noise=0.05, rank=None):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.rank = rank
        if rank:
            self.recW1 = nn.Linear(self.latent_size, rank, bias=False)
            self.recW2 = nn.Linear(rank, self.latent_size, bias=False)
        else:
            self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.act_func = nn.Tanh()
        self.gamma = gamma
        self.noise_level = noise

    def forward(self, inputs, hidden):
        # note that the dimension of the inputs can't be changed in this model
        noise = torch.randn_like(hidden) * self.noise_level
        if self.rank:
            return (1 - self.gamma) * hidden + self.gamma * (self.recW2(self.recW1(self.act_func(hidden))) +
                    self.inpW(inputs) + self.bias + noise)
        else:
            return (1 - self.gamma) * hidden + self.gamma * (self.recW(self.act_func(hidden)) +
                    self.inpW(inputs) + self.bias + noise)
    
    
class LowRankRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        rank,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,    # delta(t) / tau
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()
        self.rank = rank
        self.cell = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW1 = nn.Linear(self.latent_size, self.rank, bias=False)
        self.recW2 = nn.Linear(self.rank, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = (1 - self.gamma) * hidden + self.gamma * (self.recW2(self.recW1(self.act_func(hidden))) +
                    self.inpW(inputs) + self.bias + noise)
        return output, hidden

class Vanilla_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden=None):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden
