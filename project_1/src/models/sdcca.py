from stg import StochasticGates
import torch.nn as nn
from subnets import f, g


class MyModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.x_features = hp.layers.in_layer
        self.y_features = hp.layers.in_layer
        self.sigma_x = hp.sigma_x
        self.sigma_y = hp.sigma_y
        self.lam_x = hp.lambda_x
        self.lam_y = hp.lambda_y
        self.S_left_gate = StochasticGates(self.x_features, self.sigma_x, self.lam_x)
        self.S_right_gate = StochasticGates(self.y_features, self.sigma_y, self.lam_y)
        self.left_net = nn.Sequential(self.S_left_gate,
                                      nn.Flatten(start_dim=2, end_dim=-1),
                                      f(hp.layers),
                                      nn.BatchNorm1d(1))

        self.right_net = nn.Sequential(self.S_right_gate,
                                       nn.Flatten(start_dim=2, end_dim=-1),
                                       g(hp.layers),
                                       nn.BatchNorm1d(1))

    def forward(self, x, y):
        left, right = self.left_net(x), self.right_net(y)
        return left, right
