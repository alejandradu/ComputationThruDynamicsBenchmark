import torch
from torch import nn


# TODO: Rename lowercase
class gNODE(nn.Module):
    def __init__(
        self,
        dynamics_num_layers,
        dynamics_hidden_size,
        gating_num_layers,
        gating_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
    ):
        super().__init__()
        self.dynamics_num_layers = dynamics_num_layers
        self.dynamics_hidden_size = dynamics_hidden_size
        self.gating_num_layers = gating_num_layers
        self.gating_hidden_size = gating_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.generator = gatedMLPCell(
            input_size, self.dynamics_num_layers, self.dynamics_hidden_size, self.latent_size,
            self.gating_hidden_size, self.gating_num_layers
        )
        self.readout = nn.Linear(self.latent_size, output_size)
        # Initialize weights and biases for the readout layer
        nn.init.normal_(
            self.readout.weight, mean=0.0, std=0.01
        )  # Small standard deviation
        nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, self.latent_size), device=dev)
        hidden = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class gatedMLPCell(nn.Module):
    def __init__(self, input_size, dynamics_num_layers, dynamics_hidden_size, latent_size,
            gating_hidden_size, gating_num_layers):
        super().__init__()
        self.input_size = input_size
        self.dynamics_num_layers = dynamics_num_layers
        self.dynamics_hidden_size = dynamics_hidden_size
        self.latent_size = latent_size
        self.gating_hidden_size = gating_hidden_size
        self.gating_num_layers = gating_num_layers
        
        layers_dyn = nn.ModuleList()
        layers_gat = nn.ModuleList()
        
        for i in range(dynamics_num_layers):
            if i == 0 and dynamics_num_layers == 1:
                layers_dyn.append(nn.Linear(input_size + latent_size, latent_size))
            elif i == 0:
                layers_dyn.append(nn.Linear(input_size + latent_size, dynamics_hidden_size))
                layers_dyn.append(nn.ReLU())
            elif i == dynamics_num_layers - 1:
                layers_dyn.append(nn.Linear(dynamics_hidden_size, latent_size))
            else:
                layers_dyn.append(nn.Linear(dynamics_hidden_size, dynamics_hidden_size))
                layers_dyn.append(nn.ReLU())
                
        for i in range(gating_num_layers):
            if i == 0 and gating_num_layers == 1:
                layers_gat.append(nn.Linear(input_size + latent_size, latent_size))
                layers_gat.append(nn.ReLU())
            elif i == 0:
                layers_gat.append(nn.Linear(input_size + latent_size, gating_hidden_size))
                layers_gat.append(nn.ReLU())
            elif i == gating_num_layers - 1:
                layers_gat.append(nn.Linear(gating_hidden_size, latent_size))
            else:
                layers_gat.append(nn.Linear(gating_hidden_size, gating_hidden_size))
                layers_gat.append(nn.ReLU())
                
        self.vf_net_dyn = nn.Sequential(*layers_dyn)
        self.vf_net_gat = nn.Sequential(*layers_gat)

    # get the sigmoid
    # put a non-linearity otuside the gating to make it positive
    # but also applying tanh outside of F makes the grad behave better
    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        tanh = nn.Tanh()
        return hidden * (1 - 0.1 * torch.sigmoid(self.vf_net_gat(input_hidden))) + 0.1 * (torch.sigmoid(self.vf_net_gat(input_hidden)) * tanh(self.vf_net_dyn(input_hidden)))
