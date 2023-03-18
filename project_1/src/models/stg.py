import torch
import torch.nn as nn
import math


class StochasticGates(nn.Module):
    def __init__(self, size, sigma, lam):
        super().__init__()
        self.mus = nn.Parameter(0.5 * torch.ones(size[:]))
        self.sigma = sigma
        self.lam = lam

    def forward(self, x):
        gaussian = self.sigma * torch.randn_like(self.mus)
        shifted_gaussian = self.mus + gaussian
        z = torch.clamp(shifted_gaussian, min=0.0, max=1.0)
        gated_output = x * z
        return gated_output

    def get_reg(self):
        """ given mus output is const """
        return self.lam * torch.sum((1 + torch.erf((self.mus / self.sigma) / math.sqrt(2))) / 2)

    def get_gates(self):
        """ given mus output is const """
        return torch.clamp(self.mus, min=0.0, max=1.0)
