import torch.nn as nn
import math


class f(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.f_net = nn.Sequential(nn.Linear(math.prod(hp.in_layer), 100, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(100, 20, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(20, hp.out_layer, bias=False),)

    def forward(self, x):
        return self.f_net(x)


class g(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.g_net = nn.Sequential(nn.Linear(math.prod(hp.in_layer), 100, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(100, 20, bias=False),
                                   nn.ReLU(),
                                   nn.Linear(20, hp.out_layer, bias=False),)

    def forward(self, x):
        return self.g_net(x)
