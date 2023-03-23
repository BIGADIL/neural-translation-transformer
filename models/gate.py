import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import FloatTensor


class Gate(nn.Module):
    def __init__(self,
                 num_gates: int,
                 init: float = 2.0,
                 beta: float = 0.33,
                 gamma: float = -0.1,
                 zeta: float = 1.1):
        super().__init__()
        self.num_gates = num_gates
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        self.gate = nn.Parameter(torch.zeros(num_gates, ).fill_(init))
        self.threshold = 0.5
        self.eps = 1e-6
        self.loss_constant = (beta * math.log(-gamma / zeta))
        self.distributor = torch.distributions.uniform.Uniform(self.eps, 1 - self.eps)

    def forward(self) -> Tuple[FloatTensor, FloatTensor]:
        if self.training:
            u = self.distributor.sample((self.num_gates, )).to(self.gate.device)
            s = torch.log(u) - torch.log(1.0 - u)
            s = (s + self.gate) / self.beta
            s = torch.sigmoid(s)
        else:
            s = torch.sigmoid(self.gate)
        s = s * (self.zeta - self.gamma) + self.gamma
        out = torch.clamp(s, self.eps, 1.0)
        out_hard = torch.greater_equal(out, self.threshold).float()
        out = out + (out_hard - out).detach()
        l0_loss = torch.sigmoid(self.gate - self.loss_constant)
        l0_loss = torch.clamp(l0_loss, self.eps, 1.0 - self.eps).sum()
        return out, l0_loss
