from typing import List, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, widths: List[int] | Tuple[int], use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.layers = torch.nn.ModuleList([nn.Linear(dim, widths[0])])
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
        if use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=widths[i]) for i in range(len(widths))])
        self.output = nn.Linear(widths[-1], dim)

    def forward(self, x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = nn.functional.silu(x)
            if self.use_bn:
                x = self.bn[i](x)
        x = self.output(x)
        return x
