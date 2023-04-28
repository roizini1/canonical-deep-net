from stg import StochasticGates
import torch.nn as nn
from subnets import f, g


class MyModel(nn.Module):
    """my version of sdcca updated model"""
    def __init__(self, hp):
        super().__init__()
        # hyper-parameters:
        self.gates_learning = hp.gates_learning
        self.x_features = hp.layers.in_layer
        self.y_features = hp.layers.in_layer
        self.sigma_x = hp.sigma_x
        self.sigma_y = hp.sigma_y
        self.lam_x = hp.lambda_x
        self.lam_y = hp.lambda_y

        # Basic layers:
        self.flat = nn.Flatten(start_dim=2, end_dim=-1)

        # Stochastic gates siam nets:
        self.S_left_gate = StochasticGates(self.x_features, self.sigma_x, self.lam_x, self.gates_learning)
        self.S_right_gate = StochasticGates(self.y_features, self.sigma_y, self.lam_y, self.gates_learning)

        # Dimension reduction siam nets:
        self.left_net = nn.Sequential(self.flat,
                                      f(hp.layers),
                                      nn.BatchNorm1d(num_features=1, affine=False))

        self.right_net = nn.Sequential(self.flat,
                                       g(hp.layers),
                                       nn.BatchNorm1d(num_features=1, affine=False))

    def forward(self, x, y):
        left = nn.Sequential(self.S_left_gate, self.left_net)
        right = nn.Sequential(self.S_right_gate, self.right_net)
        return left(x), right(y)
