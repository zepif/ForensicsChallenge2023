import torch
import torch.nn as nn

class NeRF(nn.Module):
    
    def __init__(self,
        num_layers: int = 3,
        num_hidden: int = 256,
        input_size: int = 5,
        output_size: int = 4,
        use_leaky_relu: bool = False,
        ):
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_leaky_relu = use_leaky_relu
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(input_size, num_hidden)])
        self.layers.extend([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)])
        self.layers.extend([nn.Linear(num_hidden, output_size)])

    def forward(self, x):
        for i in range(self.num_layers - 1):
            if self.use_leaky_relu:
                x = torch.nn.LeakyReLU(self.layers[i](x))
            else:
                x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
