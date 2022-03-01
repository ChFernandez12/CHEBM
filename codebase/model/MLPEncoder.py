import torch

from model.modules import *
from model.Encoder import Encoder


class MLPEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, factor)

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = spectral_norm(nn.Linear(n_hid, 1))

        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims] #CFL 128, 5, 49, 4
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node #CFL 128,5,256
        x = self.node2edge(x, rel_rec, rel_send) #CFL matmul(rel_*, x) and concat sender and receivers for each atom,  NRI 'returns for each edge the concatenation of the receiver and sender features'
        x = self.mlp2(x)
        x_skip = x

        if self.factor: #CFL Using factor graph MLP encoder
            
            x = self.edge2node_adj(x, rel_rec, adj)
            
            # x = self.edge2node(x, rel_rec, rel_send) #CFL Matmul rec^T and divide by n_atoms, NRI 'accumulates all incoming edge features via a sum'
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection 
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        x = x.sum(1) #Sum node featues
        return self.fc_out(x)
