import torch
from torch import nn

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


class NODE(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
        leak=False,
        alpha=0.1,  # delta(t)/tau
        output_nonlinearity=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.leak = leak
        self.alpha = alpha
        
        if self.alpha > 1.0:
            print("Warning: running alpha > 1.0, not biological")
            
        self.output_nonlinearity = output_nonlinearity

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        if self.leak:
            self.generator = MLPCellLeak(
            input_size, self.num_layers, self.layer_hidden_size, self.latent_size, self.alpha
        )
        else:
            self.generator = MLPCell(
                input_size, self.num_layers, self.layer_hidden_size, self.latent_size, self.alpha
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
        # some models were trained w/o this
        if hasattr(self, "output_nonlinearity") and callable(self.output_nonlinearity):
            if self.output_nonlinearity:
                output = self.output_nonlinearity(output)
        return output, hidden


class MLPCell(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.alpha = alpha
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        # adjust for very old models without any time parameter
        if hasattr(self, 'alpha'):
            return hidden + self.alpha * self.vf_net(input_hidden)
        else:
            return hidden + 0.1 * self.vf_net(input_hidden)
    
    
class MLPCellLeak(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.alpha = alpha
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        if hasattr(self, "alpha"):
            return (1-self.alpha) * hidden + self.alpha * self.vf_net(input_hidden)
        else:
            return 0.9 * hidden + 0.1 * self.vf_net(input_hidden)


# Fixed implementation for Gated Neural ODE
class GatedMLPLeak(nn.Module):
    """
    MLP Cell for Gated Neural ODE implementing the leaky eqn:
    (tau)h˙ = G(h, x) hadamard [-h + F(h, x)]
    """
    def __init__(self, input_size, num_layers, gating_num_layers, layer_hidden_size, latent_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.gating_num_layers = gating_num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.alpha = alpha
        
        # F neural network (flow field)
        flow_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                flow_layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                flow_layers.append(nn.ReLU())
            elif i == num_layers - 1:
                flow_layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                flow_layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                flow_layers.append(nn.ReLU())
        self.flow_net = nn.Sequential(*flow_layers)
        
        # G neural network (gating function)
        gate_layers = nn.ModuleList()
        # Fixed implementation to ensure consistent dimensions
        if gating_num_layers == 1:
            # Single layer case
            gate_layers.append(nn.Linear(input_size + latent_size, latent_size))
            gate_layers.append(nn.Sigmoid())  # Ensure values between 0 and 1
        else:
            # Multi-layer case
            for i in range(gating_num_layers):
                if i == 0:
                    gate_layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                    gate_layers.append(nn.ReLU())
                elif i == gating_num_layers - 1:  # Fixed indexing to use gating_num_layers
                    gate_layers.append(nn.Linear(layer_hidden_size, latent_size))
                    gate_layers.append(nn.Sigmoid())  # Final sigmoid for 0-1 range
                else:
                    gate_layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                    gate_layers.append(nn.ReLU())
        
        self.gate_net = nn.Sequential(*gate_layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        
        # Calculate flow field
        flow = self.flow_net(input_hidden)
        
        # Calculate gate values
        gate = self.gate_net(input_hidden)
        
        # τh˙ = Gφ(h, x) ⊙ [−h + Fθ(h, x)]
        # discretizes to: h(t+dt) = h(t) + (dt/τ) * G_φ(h, x) ⊙ [−h + F_θ(h, x)]
        update = gate * (-hidden + flow)
        
        return hidden + self.alpha * update


class gNODE(nn.Module):
    """
    Gated Neural ODE implementation that follows the structure of NODE
    """
    def __init__(
        self,
        num_layers,
        gating_num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
        alpha=0.1,  # delta(t)/tau
        output_nonlinearity=None,
        leak=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gating_num_layers = gating_num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.alpha = alpha
        self.leak = leak
        
        if self.alpha > 1.0:
            print("Warning: running alpha > 1.0, not biological")
            
        self.output_nonlinearity = output_nonlinearity

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Create gated cell that implements both F_θ and G_φ networks
        if self.leak:
            self.generator = GatedMLPLeak(
                input_size, self.num_layers, self.gating_num_layers, self.layer_hidden_size, self.latent_size, self.alpha
            )
        else:
            raise NotImplementedError("gNODE only supports leaky model yet")
        
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
        
        # Get next hidden state using the gated dynamics
        hidden = self.generator(inputs, hidden)
        
        # Apply readout layer
        output = self.readout(hidden)
        
        # Apply output nonlinearity if specified
        if hasattr(self, "output_nonlinearity") and callable(self.output_nonlinearity):
            if self.output_nonlinearity:
                output = self.output_nonlinearity(output)
                
        return output, hidden

