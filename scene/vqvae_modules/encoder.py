
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from scene.vqvae_modules.residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, num_layers, bias=True):
        super(Encoder, self).__init__()
        self.dim_in = in_dim
        self.dim_out = out_dim
        self.dim_hidden = h_dim
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            # net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
            linear_layer = nn.Linear(self.dim_in if l == 0 else self.dim_hidden, 
                                     self.dim_out if l == num_layers - 1 else self.dim_hidden, 
                                     bias=bias)
            # Initialize weights to zero
            init.constant_(linear_layer.weight, 0)
            if bias:
                init.constant_(linear_layer.bias, 0)
            
            net.append(linear_layer)
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)

        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
